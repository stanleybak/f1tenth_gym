"""
parallel working func for concurrent race
"""

import os
import sys
import time

import numpy as np
import gym

from my_laser_models import MyScanSimulator2D

from networking import send_object, recv_object
from gap_driver import GapFollower

def init_worker(func, racetrack, start_poses):
    """init func for multiprocessing pool"""

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')

    func.scanner = MyScanSimulator2D(map_path, yaml_path)

    opp_start = [-16.768584709303845, -26.283063196557773, 4.05576782845423]

    start_poses_array = np.array([start_poses[0], opp_start])

    func.env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=start_poses_array.shape[0])

    func.obs, step_reward, done, info = func.env.reset(poses=start_poses_array)

def get_pose(obs, i):
    """extra pose from observation"""

    x = obs['poses_x'][i]
    y = obs['poses_y'][i]
    theta = obs['poses_theta'][i]

    return x, y, theta

def worker_func(param):
    """parallel working

    returns dict with entries:
        'pose_history': list of poses at each step
        'computation_time': sum of controller computation time
    """

    if param is None:
        return 'param-was-None-crashed?'

    sock, driver_name = param

    # pre-initialized objects
    scanner = worker_func.scanner
    obs = worker_func.obs
    env = worker_func.env

    #pose = np.array([odom['pose_x'], odom['pose_y'], odom['pose_theta']], dtype=float)

    #msg = {'type': 'obs', 'odom': odom, 'scan': list(scan)}
    #send_object(sock, msg)

    #obj = recv_object(sock)

    num_agents = 2
    crashed = [False] * num_agents

    gap_driver = GapFollower(7)
    
    pose_history = []
    total_computation_time = 0.0
    total_env_step_time = 0.0
    loop_start = time.perf_counter()

    while True:

        if len(pose_history) > 2000:
            break
        
        pose_history.append((get_pose(obs, 0), get_pose(obs, 1)))
        actions = np.zeros((num_agents, 2))

        for i in range(num_agents):
            if obs['collisions'][i] and not crashed[i]:
                crashed[i] = True

            if crashed[i]:
                continue

        # send obs and get actions back for ego
        odom = pack_odom(obs, 0)
        scan = obs['scans'][0]

        send_object(sock, {'type':'obs', 'odom': odom, 'scan': list(scan)})
        obj = recv_object(sock)

        if obj is None:
            print(f"driver #{i} disconnected")
            break

        assert obj['type'] == 'actions'

        actions[0, 0] = obj['steer']
        actions[0, 1] = obj['speed']

        computation_time = obj['computation_time']
        total_computation_time += computation_time

        # compute actions for other cars
        # use clean scan for gap follower opponent
        opp_pose = get_pose(obs, 1)
        scan = scanner.scan(opp_pose)
        speed, steer = gap_driver.process_lidar(scan)
        actions[1, 0] = steer
        actions[1, 1] = speed

        # step simulation
        start = time.perf_counter()
        obs, step_reward, done, info = env.step(actions)
        total_env_step_time += time.perf_counter() - start

        #env.render(mode='human_fast')

        if True in info['checkpoint_done']:
            # race completed
            break

        if all(crashed):
            # all cars crashed
            break

    loop_time = time.perf_counter() - loop_start
    other_time = loop_time - total_env_step_time - total_computation_time

    sim_time = len(pose_history)/100
    ratio = sim_time / loop_time
    print(f"{driver_name} finished {round(sim_time, 2)}s sim time in {round(loop_time, 2)}s wall time " + \
          f"({round(ratio, 3)}x real-time) with {round(total_computation_time, 1)}s controller time and " + \
          f"{round(total_env_step_time, 1)} env step time.")

    return {'pose_history': pose_history,
            'computation_time': total_computation_time,
            'env_step_time': total_env_step_time,
            'other_time': other_time}

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
