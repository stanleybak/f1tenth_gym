"""gap follower with adaptive gain"""

import os
import pickle
import numpy as np
import time

class MyGapFollower:
    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    CORNERS_SPEED = 5.0
    STRAIGHTS_SPEED = 8.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    # for tuning, can't be instance variables since we reset
    LAST_CRASH_POS = (0, 0)
    SAME_POS_CRASH_COUNT = 0

    def __init__(self, tuning_gains=False):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

        self.SPEED_GAIN = 20.0

        # dictionation of positions -> gains
        self.target_gain = 20.0
        self.tuning_gains = tuning_gains
        self.last_crash = (0, 0)
        self.gain_dict = {}

        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'gains_gap.pkl')

        try:
            with open(path, "rb") as f:
                self.gain_dict = pickle.load(f)
                print(f"loaded {len(self.gain_dict)} gains from file")
        except FileNotFoundError:
            print("no gain file found")
            assert self.tuning_gains

        self.pos_history = [] # list of pairs

    def crashed(self):
        """ car crashed, update gains """

        assert self.tuning_gains, "crashed not while tuning gains"

        last_pos = self.pos_history[-1]

        time.sleep(0.5)

        if last_pos != MyGapFollower.LAST_CRASH_POS:
            MyGapFollower.LAST_CRASH_POS = last_pos
            MyGapFollower.SAME_POS_CRASH_COUNT = 1

            print(f"Crash was in a new location: {last_pos}")
        else:
            MyGapFollower.SAME_POS_CRASH_COUNT += 1
            count = MyGapFollower.SAME_POS_CRASH_COUNT

            print(f"Crashed in same location as last time ({last_pos}), crash_count = {count}")

        # update <count> historic gains
        count = MyGapFollower.SAME_POS_CRASH_COUNT

        index = len(self.pos_history) - 1

        for i in range(count):
            if index - i < 0:
                break
            
            pos = self.pos_history[index - i]

            # reduce gain
            min_gain = 7.0 # 7.5
            # SAME_POS_CRASH_COUNT=4:
            # 5/8 -> 100?
            # 6/8 (3/4) -> 93 lap time
            # 7/8 -> 99 sec lap time

            # SAME_POS_CRASH_COUNT=3:
            # 6/8 (3/4) -> 89

            # SAME_POS_CRASH_COUNT=2:
            # 6/8 (3/4) -> 88.5
            # raw-6/8 (3/4) -> 82!!! worth doing predictive if possible

            # SAME_POS_CRASH_COUNT=1:
            # 6/8 (3/4) -> 87.9
            # raw-6/8 (3/4) -> 80!!!
            # raw-33/40 -> 81.6
            # raw-27/40 -> 80.7
            # raw-25/40 -> 88

            # SAME_POS_CRASH_COUNT=1:
            # min-gain=7.0, raw-6/8 (3/4) -> 80.1
            # min-gain=6.0, raw-6/8 (3/4) -> 82
            # min-gain=8.0, raw-6/8 (3/4) -> 80.8
            # min-gain=9.0 -> fail
            # min-gain=7.75, raw-6/8 (3/4) -> 80.8
            # min-gain=7.25, raw-6/8 (3/4) -> 81.0
         
            new_gain = (3*self.gain_dict[pos] + min_gain) / 4
            if new_gain < min_gain + 0.1:
                new_gain = min_gain
                
            self.gain_dict[pos] = new_gain
            print(f"reduced gain for {pos} to {self.gain_dict[pos]}")

        # save to file
        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'gains_gap.pkl')

        raw = pickle.dumps(self.gain_dict)
        
        with open(path, "wb") as f:
            f.write(raw)

    def update_params(self, ego_odom):
        """update gains based on position"""

        x = int(round(ego_odom['pose_x'])) // 10
        y = int(round(ego_odom['pose_y'])) // 10
        pt = (x, y)
        first = False

        if len(self.pos_history) < 10:
            # start
            pt = (pt[0], pt[1], 'start')

        if not self.pos_history:
            first = True
            self.pos_history.append(pt)

            if pt in self.gain_dict:
                self.SPEED_GAIN = self.gain_dict[pt]

        if pt != self.pos_history[-1] or first:
            # position was updated
            
            self.pos_history.append(pt)

            if pt not in self.gain_dict:
                cur_gain = 20 if self.tuning_gains else 7.75
            else:
                cur_gain = self.gain_dict[pt]

            #print(f"{pt} -> {cur_gain}")
                
            self.target_gain = cur_gain
            self.gain_dict[pt] = cur_gain

        # update gain to move towards target_gain
        self.SPEED_GAIN = self.target_gain
        #if self.SPEED_GAIN > self.target_gain:
        #    self.SPEED_GAIN -= 0.01
        #elif self.SPEED_GAIN < self.target_gain:
        #    self.SPEED_GAIN += 0.005

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
            free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        
        #steering_angle = lidar_angle / 2
        steering_angle = lidar_angle / 2
        
        return steering_angle

    def process_observation(self, ranges, ego_odom):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        self.update_params(ego_odom)
        
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))

        speed = self.SPEED_GAIN
        
        #if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
        #    speed *= 0.75
        #    speed = self.CORNERS_SPEED
        #else:
        #    speed = self.STRAIGHTS_SPEED
            
        #print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle


