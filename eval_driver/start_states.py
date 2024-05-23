"""
logic to compute opponent start states
"""

from typing import List, Tuple
import os
from copy import deepcopy
import time
import pickle

import numpy as np

from f110_gym.envs.base_classes import RaceCar

from my_laser_models import MyScanSimulator2D
from gap_driver import Driver

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

def get_opp_start_states(env, racetrack, poses, opp_driver_orig, num_overtake_scenarios, cache_filename=None) \
                            -> List[Tuple[RaceCar, Driver]]:
    """compute opponent start states for the given driver, possible loading from a file based on cache_filename

    <num_overtake_scenarios> starting positions will be returned
    """

    rv = None

    if cache_filename is not None:
        try:
            with open(cache_filename, "rb") as f:
                rv = pickle.load(f)
                print(f"loaded {len(rv)} starting positions from {cache_filename}")
        except FileNotFoundError:
            print(f"no cached starting positions found at {cache_filename}, re-running computation")

    if rv is None:
        rv = compute_opp_start_states(env, racetrack, poses, opp_driver_orig, num_overtake_scenarios)

        if cache_filename is not None:
            # save to file
            raw = pickle.dumps(rv)
        
            with open(cache_filename, "wb") as f:
                f.write(raw)

    return rv

def get_ego_start_states(env, racetrack, poses, opp_driver_orig, opp_start_tuples, cache_filename=None) \
                            -> List[Tuple[RaceCar, Driver]]:
    """compute ego start states for the given driver, possible loading from a file based on cache_filename

    one position (tuple of RaceCar, Driver) is returned for each opp_start_state.
    """

    rv = None

    if cache_filename is not None:
        try:
            with open(cache_filename, "rb") as f:
                rv = pickle.load(f)
                print(f"loaded {len(rv)} starting positions from {cache_filename}")
        except FileNotFoundError:
            print(f"no cached starting positions found at {cache_filename}, re-running computation")

    if rv is None:
        rv = compute_ego_start_states(env, racetrack, poses, opp_driver_orig, opp_start_tuples)

        if cache_filename is not None:
            # save to file
            raw = pickle.dumps(rv)
        
            with open(cache_filename, "wb") as f:
                f.write(raw)

    return rv

def compute_opp_start_states(env, racetrack, poses, opp_driver_orig, num_overtake_scenarios) \
                            -> List[Tuple[RaceCar, Driver]]:
    """compute opponent start states for the given driver

    <num_overtake_scenarios> starting positions will be returned
    """

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
    scanner = MyScanSimulator2D(map_path, yaml_path)

    obs, step_reward, done, info = env.reset(poses=poses)
    env.render(mode='human_fast')

    driver = deepcopy(opp_driver_orig)

    # sample every half second
    samples: List[Tuple[RaceCar, Driver]] = []
    frames_per_sample = 50
    current_frame = 0
    start = time.perf_counter()

    # drive opp_driver from front position until it senses the stopped_driver (lap is completed)
    while True:
        actions = np.zeros((2, 2))
     
        odom = pack_odom(obs, 0)
        scan = obs['scans'][0]

        if hasattr(driver, 'process_observation'):
            speed, steer = driver.process_observation(ranges=scan, ego_odom=odom)
        else:
            assert hasattr(driver, 'process_lidar')
            speed, steer = driver.process_lidar(scan)

        actions[0, 0] = steer
        actions[0, 1] = speed

        obs, step_reward, done, info = env.step(actions)
        env.render(mode='human_fast')

        assert not obs['collisions'][0], "vehicle crashed in single-car race"
        assert not done, "vehicle finished race without sensing opponent"

        current_frame += 1

        if current_frame % frames_per_sample == 0:
            # try to sense the opponent car
            x = odom['pose_x']
            y = odom['pose_y']
            theta = odom['pose_theta']

            pose = np.array([x, y, theta], dtype=float)

            clean_scan = scanner.scan(pose)

            # take the middle third of the scan (what's in front of the vehicle)
            third_index = len(scan) // 3

            # compare scan with clean scan
            max_diff = np.max(np.abs(scan[third_index:2*third_index] - clean_scan[third_index:2*third_index]))
            scan_diff = max_diff

            if scan_diff >= 1.0:
                # sensed opponent
                break
            
            tup = (deepcopy(env.sim.agents[0]), deepcopy(driver))
            samples.append(tup)

    assert len(samples) >= num_overtake_scenarios, f"sensed opponent too early (recorded {len(samples)} samples)"

    diff = time.perf_counter() - start
    print(f'Got {len(samples)} start positions in {round(diff, 1)} sec')

    # downsample start positions
    rv = []
    first = True

    for index_float in np.linspace(0, len(samples) - 1, num_overtake_scenarios + 1):
        if first:
            # skip index 0 (almost certainly a collision since start state is the same)
            first = False
            continue

        index = int(round(index_float))
        
        rv.append(samples[index])

    return rv

def compute_ego_start_states(env, racetrack, poses, ego_driver_orig, opp_start_tuples) \
                            -> List[Tuple[RaceCar, Driver]]:
    """compute opponent start states for the given driver

    <num_overtake_scenarios> starting positions will be returned
    """

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
    scanner = MyScanSimulator2D(map_path, yaml_path)

    driver = deepcopy(ego_driver_orig)

    rv: List[Tuple[RaceCar, Driver]] = []

    # move the opponent vehicle into the next position (at opp_index)
    opp_index = 0

    start = time.perf_counter()

    prev_two_states = []
    snapshot_frames = 10
    cur_frame = 0

    obs, step_reward, done, info = env.reset(poses=poses)
    env.render(mode='human_fast')

    env.sim.agents[1] = deepcopy(opp_start_tuples[opp_index][0])
    # stop vehicle
    env.sim.agents[1].state[3] = 0.0
    env.sim.agents[1].state[5:] = 0.0

    odom = pack_odom(obs, 0)
    x = odom['pose_x']
    y = odom['pose_y']
    theta = odom['pose_theta']
    pose = np.array([x, y, theta], dtype=float)

    #initial scan
    clean_scan = scanner.scan(pose)

    # drive opp_driver from front position until it senses the stopped_driver (lap is completed)
    while True:

        if len(prev_two_states) < 2 or cur_frame % snapshot_frames == 0:
            tup = (deepcopy(env.sim.agents[0]), deepcopy(driver))
            
            prev_two_states.append(tup)

            if len(prev_two_states) > 2:
                prev_two_states.pop(0)
        

        # note: driving is done using clean scan
        if hasattr(driver, 'process_observation'):
            speed, steer = driver.process_observation(ranges=clean_scan, ego_odom=odom)
        else:
            assert hasattr(driver, 'process_lidar')
            speed, steer = driver.process_lidar(clean_scan)

        actions = np.zeros((2, 2))

        actions[0, 0] = steer
        actions[0, 1] = speed
        
        obs, step_reward, done, info = env.step(actions)

        odom = pack_odom(obs, 0)
        x = odom['pose_x']
        y = odom['pose_y']
        theta = odom['pose_theta']
        pose = np.array([x, y, theta], dtype=float)

        scan = obs['scans'][0]
        clean_scan = scanner.scan(pose)

        env.render(mode='human_fast')

        assert not obs['collisions'][0], "ego vehicle crashed in single-car race"

        if done:
            print("Finished race? (done was True)")
            break

        # compare scan with clean scan
        max_diff = np.max(np.abs(scan - clean_scan))
        scan_diff = max_diff

        if scan_diff >= 1.0:
            # sensed opponent, move on to next index
            opp_index += 1

            # get state from a little bit ago
            rv.append(prev_two_states[0])
            
            if opp_index >= len(opp_start_tuples):
                break

            env.sim.agents[1] = deepcopy(opp_start_tuples[opp_index][0])
            # stop vehicle
            env.sim.agents[1].state[3] = 0.0
            env.sim.agents[1].state[5:] = 0.0 

    diff = time.perf_counter() - start
    print(f'Got {len(rv)} ego start positions in {round(diff, 1)} sec')

    assert len(rv) == len(opp_start_tuples), f"expected {len(opp_start_tuples)} ego start positions, got {len(rv)}"

    return rv
    
