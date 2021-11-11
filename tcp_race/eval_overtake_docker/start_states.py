"""
start states generation for tcp_race

Stanley Bak, Nov 2021
"""

from typing import List, Tuple

import sys
import os
import pickle
from copy import deepcopy

import gym
import numpy as np
import time

from f110_gym.envs.base_classes import RaceCar

from gap_driver import GapFollower, Driver
from my_laser_models import MyScanSimulator2D
from util import pack_odom

def render_callback(env_renderer):
    'custom extra drawing function'

    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]

    # hmm? looks like it was for multiple cars at some point
    top, bottom, left, right = max(y), min(y), min(x), max(x)

    #e.left = left - 800
    #e.right = right + 800
    #e.top = top + 800
    #e.bottom = bottom - 800

    z = env_renderer.zoom_level

    (width, height) = env_renderer.get_size()
    e.left = left - z * width/2
    e.right = right + z * width/2
    e.bottom = bottom - z * height/2
    e.top = top + z * height/2


def get_opp_start_states(racetrack, start_poses, gain=7, num_overtake_scenarios=10) -> List[Tuple[RaceCar, Driver]]:
    """compute opponent start states for the given driver, possible loading from a file based on cache_filename

    <num_overtake_scenarios> starting positions will be returned
    """

    assert len(start_poses) == 2

    os.makedirs('cache', exist_ok=True)
    cache_filename = f"cache/opp_starts_{num_overtake_scenarios}_gap{gain}.pkl"

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=2)

    env.add_render_callback(render_callback)

    rv = None

    if cache_filename is not None:
        try:
            with open(cache_filename, "rb") as f:
                rv = pickle.load(f)
                print(f"loaded {len(rv)} starting positions from {cache_filename}")
        except FileNotFoundError:
            print(f"no cached starting positions found at {cache_filename}, re-running computation")

    if rv is None:
        poses = np.array(start_poses)
        opp_driver = GapFollower(gain)
            
        rv = compute_opp_start_states(env, racetrack, poses, opp_driver, num_overtake_scenarios)

        if cache_filename is not None:
            # save to file
            raw = pickle.dumps(rv)
        
            with open(cache_filename, "wb") as f:
                f.write(raw)
                print(f"wrote {len(rv)} starting positions to {cache_filename}")

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
    #env.render(mode='human_fast')

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
        #env.render(mode='human_fast')

        assert not obs['collisions'][0], "vehicle crashed in single-car race"
        assert not done, "vehicle finished race without sensing opponent"

        current_frame += 1

        if current_frame % frames_per_sample == 0:
            print(".", end='', flush=True)
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

    print()
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
