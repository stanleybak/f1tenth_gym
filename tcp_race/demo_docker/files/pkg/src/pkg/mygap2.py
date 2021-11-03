"""gap follower with adaptive gain"""

import os
import pickle
import numpy as np
import time

import pyglet
from pyglet.gl import *

class MyGap2Follower:
    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    #BEST_POINT_CONV_SIZE = 20 # 80
    MAX_LIDAR_DIST = 3000000
    CORNERS_SPEED = 5.0
    #STRAIGHTS_SPEED = 8.0
    #STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    # for tuning, can't be instance variables since we reset
    LAST_CRASH_POS = (0, 0)
    SAME_POS_CRASH_COUNT = 0

    SCORE_LABEL = None

    def __init__(self, tuning_gains=False):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

        # the step for steering
        self.real_steer_predicted = 0
        self.steer_step = 0.032


        self.real_steer_integral = 0
        self.desired_steer_integral = 0
        self.desired_steer = 0

        self.duty_cycle = 10
        self.duty_cycle_index = 0
        self.duty_cycle_on = 0
        
        self.SPEED_GAIN = 20.0
        self.x = 0
        self.y = 0
        self.vel = 0

        # dictionation of positions -> gains
        self.target_gain = 20.0
        self.tuning_gains = tuning_gains
        self.last_crash = (0, 0)
        self.gain_dict = {}

        self.BEST_POINT_CONV_SIZE = 20 # sets to 80 after first turn

        self.slow_steering = False

        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'gains_gap2.pkl')

        try:
            with open(path, "rb") as f:
                self.gain_dict = pickle.load(f)
                print(f"loaded {len(self.gain_dict)} gains from file")
        except FileNotFoundError:
            assert self.tuning_gains, "no gain file found"

        self.pos_history = [] # list of pairs

    def render(self, x, y, env):
        "extra render func"

        cur_gain = self.target_gain

        if MyGap2Follower.SCORE_LABEL is None:
            MyGap2Follower.SCORE_LABEL = pyglet.text.Label(
                f'Gain: {self.target_gain}',
                font_size=24,
                x=x,
                y=y,
                anchor_x='center',
                anchor_y='center',
                # width=0.01,
                # height=0.01,
                color=(255, 255, 255, 255),
                batch=env.batch)
        else:
            sl = MyGap2Follower.SCORE_LABEL
            sl.x = x
            sl.y = y + 100
            sl.text = f'Gain: {round(self.target_gain, 1)}'

    def crashed(self):
        """ car crashed, update gains """

        assert self.tuning_gains, "crashed not while tuning gains"

        last_pos = self.pos_history[-1]

        print(f"crashed with vel {round(self.vel, 2)}")

        time.sleep(0.5)

        if last_pos != MyGap2Follower.LAST_CRASH_POS:
            MyGap2Follower.LAST_CRASH_POS = last_pos
            MyGap2Follower.SAME_POS_CRASH_COUNT = 4

            print(f"Crash was in a new location: {last_pos}")
        else:
            MyGap2Follower.SAME_POS_CRASH_COUNT += 1
            count = MyGap2Follower.SAME_POS_CRASH_COUNT

            print(f"Crashed in same location as last time ({last_pos}), crash_count = {count}")

        # update <count> historic gains
        count = MyGap2Follower.SAME_POS_CRASH_COUNT

        index = len(self.pos_history) - 1

        for i in range(count):
            if index - i < 0:
                break

            if index - i < 10 and self.BEST_POINT_CONV_SIZE == 80:
                # don't tune start anymore
                break
            
            pos = self.pos_history[index - i]

            # reduce gain
            min_gain = 7.5 # was just 7.5
            old_gain = self.gain_dict[pos] 
            new_gain = 4 * old_gain / 5

            #if new_gain < self.vel - 0.5:
            #    new_gain = self.vel - 0.5
            
            if new_gain < min_gain + 0.1:
                new_gain = min_gain
                
            self.gain_dict[pos] = new_gain
            print(f"reduced gain for {pos} to {self.gain_dict[pos]}")

        # save to file
        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'gains_gap2.pkl')

        raw = pickle.dumps(self.gain_dict)
        
        with open(path, "wb") as f:
            f.write(raw)

    def update_params(self, ego_odom):
        """update gains based on position"""

        self.x = ego_odom['pose_x']
        self.y = ego_odom['pose_y']
        self.vel = ego_odom['linear_vel_x']
        self.ang_vel = ego_odom['angular_vel_z']

        if self.x < -40 and self.BEST_POINT_CONV_SIZE < 80:
            self.BEST_POINT_CONV_SIZE += 1

        x = int(round(self.x)) // 10
        y = int(round(self.y)) // 10
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
                cur_gain = 20 if self.tuning_gains else max(10, self.target_gain - 1.0) # was 20
                
                if not self.tuning_gains:
                    self.slow_steering = True
            else:
                cur_gain = self.gain_dict[pt]

            #print(f"{pt} -> {cur_gain}")
                
            self.target_gain = cur_gain
            self.gain_dict[pt] = cur_gain

        # update gain to move towards target_gain
        self.SPEED_GAIN = self.target_gain

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

        if self.slow_steering:
            steering_angle = lidar_angle / 4
        else:
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
        steer = self.get_angle(best, len(proc_ranges))

        speed = self.SPEED_GAIN

        ####################################333
        # okay, update steer integrators

        self.desired_steer = steer

        steer_base = self.steer_step * round(steer / self.steer_step)
        steer_remainder = self.desired_steer - steer_base

        # set duty-cycle low and high
        self.duty_cycle_on = abs(round(self.duty_cycle * steer_remainder / self.steer_step))

        print(f"steer_remainder: {steer_remainder}, duty-cycle: {self.duty_cycle_on}/{self.duty_cycle}")

        if self.duty_cycle_index < self.duty_cycle_on:
            steer_bonus = self.steer_step
            steer_bonus *= -1 if self.desired_steer < 0 else 1
        else:
            steer_bonus = 0

        steer = steer_base + steer_bonus

        print(f"index={self.duty_cycle_index}, steer_bouns={steer_bonus}, steer= {steer}")
            
        self.duty_cycle_index = (self.duty_cycle_index + 1) % self.duty_cycle

        ###############
        # update real_steer_predicted

        steer_diff = steer - self.real_steer_predicted
            
        #steer_u = steering_constraint(self.real_steer_predicted, steer)       

        if np.fabs(steer_diff) > 1e-4:
            sv = (steer_diff / np.fabs(steer_diff)) * self.steer_step
        else:
            sv = 0.0

        steer_u = steering_constraint(self.real_steer_predicted, sv)
            
        self.real_steer_predicted += steer_u

        ###############

        #print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steer


def steering_constraint(steering_angle, steering_velocity, s_min=-0.4189, s_max=0.4189, sv_min=-3.2, sv_max=3.2):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity
