"""
eval a specific scenario for overtaking
"""

import sys
import time
import os

import numpy as np
import pyglet
import gym

from my_laser_models import MyScanSimulator2D
from gap_driver import GapFollower

def pack_odom(obs, i):
    """create single-car odometry from multi-car odometry"""

    keys = {
        'poses_x': 'pose_x',
        'poses_y': 'pose_y',
        'poses_theta': 'pose_theta',
        'linear_vels_x': 'linear_vel_x',
        'linear_vels_y': 'linear_vel_y',
        'ang_vels_z': 'angular_vel_z',
    }
    return {single: obs[multi][i] for multi, single in keys.items()}

class LidarDrawer:
    """class for drawing lidar data of a car"""

    def __init__(self):

        self.vertex_list = None

        self.x_y_theta_scan = None

        self.color = (0, 255, 0, 32) + (0, 255, 0, 64) #(64, 64, 64)

        self.fov = 4.7

        self.counter = 0
        self.downsample_step = 5

        self.min_ind = None
        self.max_ind = None
        self.min_max_ind_color = [255, 0, 0, 255]
        self.ind_vertex_list = None

        self.full_scan = None

        self.last_left = None
        self.last_top = None
        self.frames = 0

    def render_callback(self, w):
        """render callback, draws lidar data"""

        self.frames += 1

        # update camera to follow car
        x = w.cars[0].vertices[::2]
        y = w.cars[0].vertices[1::2]

        # hmm? looks like it was for multiple cars at some point
        top, bottom, left, right = max(y), min(y), min(x), max(x)

        if self.frames < 5 or (self.last_left == w.left and self.last_top == w.top):
            z = w.zoom_level * 2

            (width, height) = w.get_size()
            w.left = left - z * width/2
            w.right = right + z * width/2
            w.bottom = bottom - z * height/2
            w.top = top + z * height/2

            self.last_left = w.left
            self.last_top = w.top

        #########################################3

        if self.x_y_theta_scan is not None:
            car_x, car_y, car_theta, scan = self.x_y_theta_scan

            num_pts = 2 * len(scan)

            if self.ind_vertex_list is None and self.min_ind is not None:
                num_inds = len(self.min_ind)
                
                self.ind_vertex_list = w.batch.add(4 * num_inds, pyglet.gl.GL_LINES, None,
                        ('v3f/stream', np.zeros(3 * 4 * num_inds)),
                        ('c4B/static', self.min_max_ind_color * 4 * num_inds)
                    )

            if self.vertex_list is None:
                pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
                pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    
                # make initial self.vertex_list

                self.vertex_list = w.batch.add(num_pts, pyglet.gl.GL_LINES, None,
                        ('v3f/stream', np.zeros(3 * num_pts)),
                        ('c4B/static', self.color * len(scan))
                    )

            #pt_list = np.zeros(3 * num_pts) # list of x, y, 0.0 for the lines
            offset = 0

            theta_arr = np.linspace(car_theta - self.fov / 2, car_theta + self.fov / 2, len(scan))
            sines = np.sin(theta_arr)
            cosines = np.cos(theta_arr)

            for dist, sin, cos in zip(scan, sines, cosines):
                x = 50*car_x + 50*dist * cos
                y = 50*car_y + 50*dist * sin

                self.vertex_list.vertices[offset:offset + 6] = (50*car_x, 50*car_y, 0, x, y, 0)
                offset += 6

            if self.min_ind is None and self.ind_vertex_list is not None:
                self.ind_vertex_list.vertices[:] = 0

            if self.min_ind is not None and self.ind_vertex_list is not None:
                #print(f"inds: {self.min_ind, self.max_ind}")

                # optimization: remove these arrays
                full_theta_arr = np.linspace(car_theta - self.fov / 2, car_theta + self.fov / 2, len(self.full_scan))
                full_sines = np.sin(full_theta_arr)
                full_cosines = np.cos(full_theta_arr)
                index = 0
                
                for min_ind, max_ind in zip(self.min_ind, self.max_ind):
                    for i in [min_ind, max_ind + 1]:

                        if i >= len(self.full_scan):
                            i = len(self.full_scan) - 1

                        dist = self.full_scan[i]
                        cos = full_cosines[i]
                        sin = full_sines[i]

                        x = 50*car_x + 50*dist * cos
                        y = 50*car_y + 50*dist * sin

                        pts = (50*car_x, 50*car_y, 0, x, y, 0)
                        self.ind_vertex_list.vertices[index:index+6] = pts
                        index += 6

            #self.vertex_list.vertices = pt_list

            #self.counter += 1
            #theta = self.counter / 50

            #x = 200 * np.cos(theta)
            #y = 200 * np.sin(theta)

            #self.vertex_list.vertices[3:5] = 2*x, y

    def update_min_max_ind(self, min_ind, max_ind):
        """update min and max ind"""
        
        self.min_ind = min_ind
        self.max_ind = max_ind

    def update_pose_lidar(self, x, y, theta, scan):
        """update car pose and lidar data"""

        if self.downsample_step > 1:
            self.full_scan = scan
            downsampled = []

            for i, d in enumerate(scan):
                if i % self.downsample_step == 0:
                    downsampled.append(d)

            scan = downsampled

        self.x_y_theta_scan = x, y, theta, scan

def get_pose(obs, i):
    """extra pose from observation"""

    x = obs['poses_x'][i]
    y = obs['poses_y'][i]
    theta = obs['poses_theta'][i]

    return x, y, theta

def main():
    #(ego_start_tup, opp_start_tup, env, racetrack, overtake_timeout=60):
    """run an evaluation from the given positions

    returns one of: 'overtake', 'crash', 'overtake_timeout'
    """

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    racetrack = 'SOCHI'

    # original start_poses
    start_poses = [[0.8007017, -0.2753365, 4.1421595]]#, [0.8162458, 1.1614572, 4.1446321]]

    start_poses.append([-16.768584709303845, -26.283063196557773, 4.05576782845423])

    start_poses.append([-16.768584709303845, -28.283063196557773, 4.05576782845423])

    drivers = [GapFollower(8), GapFollower(2), GapFollower(0.1)]
    num_drivers = len(drivers)

    env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, racetrack),
                       map_ext=".png", num_agents=num_drivers)

    lidar_drawer = LidarDrawer()
    env.add_render_callback(lidar_drawer.render_callback)

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
    scanner = MyScanSimulator2D(map_path, yaml_path)



    obs, step_reward, done, info = env.reset(poses=np.array(start_poses))
    env.render(mode='human_fast')

    cur_frame = 0
    max_frames = 100 * 7
    paused = False

    new_scan_time = 0
    old_scan_time = 0
    clean_scan_time = 0
     
    while cur_frame < max_frames:

        actions = np.zeros((num_drivers, 2))

        for i in range(num_drivers):
            odom = pack_odom(obs, i)
            scan = obs['scans'][i]

            if i == 0:
                x, y, theta = ego_pose = get_pose(obs, 0)
                opp_pose1 = get_pose(obs, 1)
                opp_pose2 = get_pose(obs, 2)

                ego_scan = scan

                #start = time.perf_counter()
                #scanner.scan(ego_pose)
                #clean_scan_time += time.perf_counter() - start

                start = time.perf_counter()
                new_scan, min_inds, max_inds = scanner.scan(ego_pose, [opp_pose1, opp_pose2])
                new_scan_time += time.perf_counter() - start

                #start = time.perf_counter()
                #scanner.scan(ego_pose, [opp_pose1, opp_pose2], old=True)
                #old_scan_time += time.perf_counter() - start

                # compare gym scan with new scan code
                diff_scan = np.max(np.abs(ego_scan - new_scan))

                if diff_scan > 0.1 and not paused:
                    print(f"scan result was different, diff_scan: {diff_scan}")
                    paused = True

                lidar_drawer.update_pose_lidar(x, y, theta, scan)
                lidar_drawer.update_min_max_ind(min_inds, max_inds)

            driver = drivers[i]
                
            if hasattr(driver, 'process_observation'):
                speed, steer = driver.process_observation(ranges=scan, ego_odom=odom)
            else:
                assert hasattr(driver, 'process_lidar')
                speed, steer = driver.process_lidar(scan)

            actions[i, 0] = steer
            actions[i, 1] = speed

        if not paused:
            obs, step_reward, done, info = env.step(actions)
            cur_frame += 1
            
        env.render(mode='human_fast')

        if obs['collisions'][0] or obs['collisions'][1]:
            print("Crashed!")
            break

        assert not done, "Finished race? (done was True)"

    print(f"old_scan time: {old_scan_time}")
    print(f"new_scan time: {new_scan_time}")
    print(f"clean_scan time: {clean_scan_time}")

if __name__ == "__main__":
    main()
