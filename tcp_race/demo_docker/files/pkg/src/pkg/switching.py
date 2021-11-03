'''
Switching controller strategy
'''

import os
from copy import deepcopy

import numpy as np
import time

from pkg.mygap import MyGapFollower

from pkg.pp import MyPurePursuitPlanner
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

        #self.gap = None#MyGapFollower(tuning_gains=False, filename='gains_lookahead.pkl')
        self.unc = DisparityExtenderUNCDriver(tuning_gains=False)
        self.pp = MyPurePursuitPlanner(tuning_gains=False)

        self.drivers = [self.pp, self.unc]
        
        self.driver_index = 0
        self.old_driver_index = 1

        parent = os.path.dirname(os.path.realpath(__file__))
        map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
        yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
        self.scanner = MyScanSimulator2D(map_path, yaml_path)
        self.steps = 0

        self.last_speed = 0
        self.last_steer = 0
        self.switched_count = 0
        self.switch_frames = 1000
        self.switch_interpolate_frames = 100 # was 50

        self.slow_down_steps = -1


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

        max_diff = np.max(np.abs(orig_smoothed - expected_smoothed))
        scan_diff = max_diff

        print(max_diff)

        angular_vel_z = ego_odom['angular_vel_z']

        if scan_diff >= 1.0 and self.driver_index == 1:
            # keep DE longer
            self.switched_count = max(self.switched_count, self.switch_frames - self.switch_interpolate_frames)
            print(f"resetting switched count to {self.switched_count}")

        # don't switch to frequently or in the middle of a turn
        if self.switched_count == 0 and abs(angular_vel_z) < 0.1 and self.steps > 250:
            # switch drivers maybe
            
            if scan_diff >= 1.0 and self.driver_index == 0:
                print(f"switching to de! max_diff = {max_diff}, {ego_odom}")
                #time.sleep(0.5)

                #self.gap.slow_steer_count = self.switch_frames

                self.old_driver_index = 0
                self.driver_index = 1
                self.switched_count = self.switch_frames
            elif scan_diff < 2.0 and self.driver_index == 1: 
                # disparity extender -> gap
                self.switched_count = self.switch_frames

                print(f"switching to normal! {ego_odom}")
                #self.pp.longer_lookahead_count = 5
                #time.sleep(0.5)

                self.old_driver_index = 1
                self.driver_index = 0

        ranges = orig_scan
        speed_unc, steer_unc = self.drivers[1].process_observation(ranges, ego_odom)

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

            print(frac)

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
            steer = frac * old_steer + (1-frac) * new_steer
            #steer = new_steer #(3 * new_steer + self.last_steer) / 4

        # fast race start logic
        if self.steps <= 1000:
            if y > -18:
                # start of race, go as fast as you can to mess up other guy's lidar
                steer = 0
                speed = 20
            else:
                if self.slow_down_steps == -1:
                    if self.driver_index == 1:
                        self.slow_down_steps = 80
                    else:
                        self.slow_down_steps = 40

                if self.slow_down_steps > 0:
                    self.slow_down_steps -= 1

                    print(f"slowdown steps: {self.slow_down_steps}")

                    # slow down before de switch
                    speed = 0
                    steer = 0

        #if x < -130: # slow down more near these corners
        #    speed *= 0.95

        if self.driver_index == 1:
            speed *= 0.8
        
        self.last_speed = speed
        self.last_steer = steer

        self.steps += 1

        print(f"step: {self.steps}, driver: {self.driver_index}, cmd: {speed}, {steer}. {ego_odom}")
        
        return speed, steer
