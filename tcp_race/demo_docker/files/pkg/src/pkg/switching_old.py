'''
Switching controller strategy
'''

import os
from copy import deepcopy

import numpy as np
import time

from pkg.mygap import MyGapFollower

from pkg.unc_disparity_extender import DisparityExtenderUNCDriver
from pkg.my_laser_models import MyScanSimulator2D
from pkg.my_racecar import MyRaceCar

def smooth_lidar(ranges):
    """smooth out the lidar scan like gap follower"""

    MAX_LIDAR_DIST = 3000000
    PREPROCESS_CONV_SIZE = 3

    # we won't use the LiDAR data from directly behind us
    proc_ranges = np.array(ranges[135:-135])
    # sets each value to the mean over a given window
    proc_ranges = np.convolve(proc_ranges, np.ones(PREPROCESS_CONV_SIZE), 'same') / PREPROCESS_CONV_SIZE
    proc_ranges = np.clip(proc_ranges, 0, MAX_LIDAR_DIST)

    return proc_ranges

class SwitchingDriver:

    def __init__(self, racetrack='SOCHI'):

        self.gap = None#MyGapFollower(tuning_gains=False, filename='gains_lookahead.pkl')
        self.unc = DisparityExtenderUNCDriver(tuning_gains=False)

        self.drivers = []#[self.gap, self.unc_obstacle]
        
        self.driver_index = 0
        self.old_driver_index = 1

        parent = os.path.dirname(os.path.realpath(__file__))
        map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
        yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
        self.scanner = MyScanSimulator2D(map_path, yaml_path)
        self.racecar = None
        self.steps = 0

        self.last_speed = 0
        self.last_steer = 0
        self.switched_count = 0
        self.switch_frames = 500
        self.switch_interpolate_frames = 50 # was 50

        self.started_outside = True

    def process_observation(self, ranges, ego_odom):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        orig_scan = ranges
        ranges = None
        
        x = ego_odom['pose_x']
        y = ego_odom['pose_y']
        theta = ego_odom['pose_theta']

        pose = np.array([x, y, theta], dtype=float)

        expected_scan = self.scanner.scan(pose)
        expected_smoothed = smooth_lidar(expected_scan)
        orig_smoothed = smooth_lidar(orig_scan)

        scan_diff = np.max(np.abs(orig_smoothed - expected_smoothed))

        angular_vel_z = ego_odom['angular_vel_z']

        # don't switch to frequently or in the middle of a turn
        if self.switched_count == 0 and abs(angular_vel_z) < 0.034 and self.steps > 250:
            # switch drivers maybe
            
            if scan_diff >= 1.0 and self.driver_index == 0:
                print(f"switching to de! {ego_odom}")
                #time.sleep(0.5)

                #self.gap.slow_steer_count = self.switch_frames

                self.old_driver_index = 0
                self.driver_index = 1
                self.switched_count = self.switch_frames
            elif scan_diff < 2.0 and self.driver_index == 1: 
                # disparity extender -> gap
                self.switched_count = self.switch_frames / 4

                print(f"switching to gap! {ego_odom}")
                #time.sleep(0.5)

                self.old_driver_index = 1
                self.driver_index = 0

        if self.racecar is None:
            # start: init virtual racecar
            self.racecar = MyRaceCar()
            pose = np.array([x, y, theta])
            self.racecar.reset(pose)
            self.steps = 0

            # figure out if we're start on inside or outside using lidar
            lidar_len = len(expected_scan)

            right_index = lidar_len // 4
            left_index = 3 * lidar_len // 4

            right_reading = expected_scan[right_index]
            left_reading = expected_scan[left_index]

            if left_reading * 2 < right_reading:
                self.started_outside = True
                gain_filename = 'gains_gap_outside79.pkl'
            else:
                self.started_outside = False
                gain_filename = 'gains_gap_outside79.pkl'

            self.gap = MyGapFollower(tuning_gains=False, filename=gain_filename)
            self.drivers = [self.gap, self.unc]

        state_incorrect = False

        if abs(self.racecar.state[0] - x) > 1e-4:
            print("virtual racecasr state is incorrect")
            self.racecar.state[0] = x
            self.racecar.state[1] = y
            self.racecar.state[4] = theta
            state_incorrect = True

        ranges = orig_scan
        speed_unc, steer_unc = self.drivers[1].process_observation(ranges, ego_odom)
        ranges = None

        if state_incorrect:
            ranges = orig_scan
            speed_gap, steer_gap = self.drivers[0].process_observation(ranges, ego_odom)
            speed_gap *= 0.9 # incorrect
        else:
            # compare scan with expected scan to see if there's an obstacle
            virtual_driver = deepcopy(self.drivers[self.driver_index])
            virtual_racecar = deepcopy(self.racecar)

            speed, steer = virtual_driver.process_observation(expected_scan, ego_odom) # gf is stateless

            # 2 -> 83

            for _ in range(2):
                virtual_racecar.update_pose(steer, speed)

                virtual_pose = np.array([virtual_racecar.state[0], virtual_racecar.state[1], virtual_racecar.state[4]])
                virtual_scan = self.scanner.scan(virtual_pose)

                virtual_ego_odom = {}
                virtual_ego_odom['pose_x'] = virtual_racecar.state[0]
                virtual_ego_odom['pose_y'] = virtual_racecar.state[1]
                virtual_ego_odom['pose_theta'] = virtual_racecar.state[4]

                speed, steer = virtual_driver.process_observation(virtual_scan, virtual_ego_odom) # gf is stateless

            virtual_pose = np.array([virtual_racecar.state[0], virtual_racecar.state[1], virtual_racecar.state[4]])
            virtual_scan = self.scanner.scan(virtual_pose)

            # use the predicted scan from the virtual car for decision making
            ranges = virtual_scan

            speed_gap, steer_gap = self.drivers[0].process_observation(ranges, ego_odom)

        # interpolated between speed_unc and speed_gap

        if self.switched_count > 0:
            self.switched_count -= 1

        if self.switched_count <= self.switch_frames - self.switch_interpolate_frames:
            if self.driver_index == 0:
                speed = speed_gap
                steer = steer_gap
            else:
                speed = speed_unc
                steer = steer_unc
        else:
            f = self.switched_count - (self.switch_frames - self.switch_interpolate_frames)
            
            # blend between the two controllers when switching
            frac = f / self.switch_interpolate_frames

            if self.driver_index == 0:
                assert self.old_driver_index == 1
                new_steer = steer_gap
                new_speed = speed_gap

                old_steer = steer_unc
                old_speed = speed_unc
            else:
                assert self.old_driver_index == 0
                new_steer = steer_unc
                new_speed = speed_unc

                old_steer = steer_gap
                old_speed = speed_gap
            
            speed = frac * old_speed + (1-frac) * new_speed
            #steer = frac * old_steer + (1-frac) * new_steer
            steer = new_steer #(3 * new_steer + self.last_steer) / 4

        # race start
        if self.steps <= 250:
            # start of race, go as fast as you can to mess up other guy's lidar
            steer = 0
            speed = 20
            
        if self.driver_index == 1 and self.steps < 330:
            speed = 0
            steer = 0
            # slow down before de switch

        if x < -130: # slow down more near these corners
            speed *= 0.95

        # update virtual racecar with applied command to keep it in sync
        self.racecar.update_pose(steer, speed)
        
        self.last_speed = speed
        self.last_steer = steer

        self.steps += 1

        #print(f"driver: {self.driver_index}, {speed}, {steer}")
        
        return speed, steer
