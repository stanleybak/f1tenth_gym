"""
parallel working func for concurrent race
"""

from typing import List, Union

import os
import sys
import time
from copy import deepcopy

import numpy as np
import gym

from my_laser_models import MyScanSimulator2D

from networking import send_object, recv_object
from util import pack_odom, get_pose

def worker_func(param):
    """parallel working

    returns dict with entries:
        'name': driver_name
        'results': results of all test cases
    """

    start = time.perf_counter()

    racetrack, start_poses, sock, driver_name, opp_start_tuples = param

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')

    scanner = MyScanSimulator2D(map_path, yaml_path)

    start_poses_array = np.array(start_poses)
    num_agents = start_poses_array.shape[0]

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=num_agents)

    obs, step_reward, done, info = env.reset(poses=start_poses_array)
    #env.render('human_fast')

    opp_index = 0
    results: List[Union[str, int]] = [] # "crashed" or frames to pass

    env.sim.agents[1] = deepcopy(opp_start_tuples[opp_index][0])
    opp_driver = opp_start_tuples[opp_index][1]
    # stop vehicle
    env.sim.agents[1].state[3] = 0.0
    env.sim.agents[1].state[5:] = 0.0

    state = "detecting" # detecting or racing
    last_detected_in_back = False
    overtake_timeout = 30
    max_frames = 100 * overtake_timeout
    start_frame = 0
    frame = 0

    replay_list = []
    cur_replay = []

    while True:
        frame += 1
        
        actions = np.zeros((num_agents, 2))
        
        if state == "detecting":
            pose = get_pose(obs, 0)
            scan = obs['scans'][0]
            clean_scan = scanner.scan(pose)

            max_diff = np.max(np.abs(scan - clean_scan))

            if max_diff > 0.1:
                start_frame = frame
                end_frame = frame + max_frames

                last_detected_in_back = False
                env.sim.agents[1] = deepcopy(opp_start_tuples[opp_index][0])
                state = "racing"
                cur_replay = []

        # run controllers
        for i in range(num_agents):            
            if obs['collisions'][i]:
                continue

            scan = obs['scans'][i]
            odom = pack_odom(obs, i)

            if i == 0:
                # get next command over tcp
                send_object(sock, {'type':'obs', 'odom': odom, 'scan': list(scan)})
                obj = recv_object(sock)

                if obj is None:
                    print(f"driver #{i} disconnected")
                    break

                assert obj['type'] == 'actions'

                actions[i, 0] = obj['steer']
                actions[i, 1] = obj['speed']

                if state == 'racing':
                    cur_replay.append(get_pose(obs, 0))
                
            elif i == 1 and state == "racing":
                if hasattr(opp_driver, 'process_observation'):
                    speed, steer = opp_driver.process_observation(ranges=scan, ego_odom=odom)
                else:
                    assert hasattr(opp_driver, 'process_lidar')
                    speed, steer = opp_driver.process_lidar(scan)

                actions[i, 0] = steer
                actions[i, 1] = speed

        # advance environment
        obs, step_reward, done, info = env.step(actions)
        #env.render(mode='human_fast')

        # check result
        result: Union[None, int, str] = None

        if obs['collisions'][0]:
            result = "crash"

        if result is None and state == "racing":
            # compare scan with clean scan to see if overtake completed
            # overtake completed if back 1/4 of scan is not clean (opponent is behind) and rest is clean
            f = len(scan) // 4
            pose = get_pose(obs, 0)
            ego_scan = obs['scans'][0]
            ego_clean_scan = scanner.scan(pose)

            back_scan = np.append(ego_scan[-f:], ego_scan[:f])
            back_clean_scan = np.append(ego_clean_scan[-f:], ego_clean_scan[:f])

            max_diff_back = np.max(np.abs(back_scan - back_clean_scan))
            max_diff_front = np.max(np.abs(ego_scan[f:-f] - ego_clean_scan[f:-f]))

            detected_in_front = max_diff_front > 0.1
            detected_in_back = max_diff_back > 0.1

            if last_detected_in_back and not detected_in_back and not detected_in_front:
                # passed opponent
                result = frame - start_frame
            elif frame >= end_frame:
                result = "timeout"

            last_detected_in_back = detected_in_back

        if result is not None:
            opp_index += 1
            replay_list.append(cur_replay)

            if isinstance(result, int):
                print(f"{driver_name} Result: Overtake ({result / 100}s). {opp_index}/{len(opp_start_tuples)}")
            else:
                print(f"{driver_name} Result: {result}. {opp_index}/{len(opp_start_tuples)}")
                
            results.append(result)

            if opp_index >= len(opp_start_tuples):
                print(f"{driver_name} Done!")
                break
            
            # reset with next opponent
            state = "detecting"
            send_object(sock, {'type':'reset'})

            obs, step_reward, done, info = env.reset(poses=start_poses_array)
            #env.render('human_fast')
            
            env.sim.agents[1] = deepcopy(opp_start_tuples[opp_index][0])
            opp_driver = opp_start_tuples[opp_index][1]
            # stop vehicle
            env.sim.agents[1].state[3] = 0.0
            env.sim.agents[1].state[5:] = 0.0


    runtime = time.perf_counter() - start
            
    return {'name': driver_name, 'results': results, 'replay_list': replay_list, 'runtime': runtime}
