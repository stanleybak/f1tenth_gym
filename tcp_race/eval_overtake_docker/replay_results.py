"""
replay results
"""

import os
import sys
from copy import deepcopy

import numpy as np
import gym
import pyglet

from my_laser_models import MyScanSimulator2D
from util import get_pose, pack_odom

class Renderer:
    """custom renderer"""

    def __init__(self, agent_names):

        self.agent_names = agent_names
        self.labels = []
        self.positions = []

    def callback(self, env_renderer):
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

        # update name labels
        if self.positions:

            # need to delete and re-create to draw moving text correctly
            while self.labels:
                self.labels[-1].delete()
                del self.labels[-1]
            
            for i, (name, (x, y)) in enumerate(zip(self.agent_names, self.positions)):

                r, g, b = e.rgbs[i % len(e.rgbs)]
                
                l = pyglet.text.Label(f'{name}',
                        font_size=22,
                        x=50 * x,
                        y=50 * y - 40,
                        anchor_x='center',
                        anchor_y='center',
                        color=(r, g, b, 255),
                        batch=e.batch)

                self.labels.append(l)

def replay_results(race_results, racetrack, start_poses, opp_start_tuples):
    """replay race results"""

    num_agents = len(race_results)
    
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    start_poses_array = np.array([start_poses[0]] * (num_agents + 1))
    
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=num_agents+1, num_beams=3, should_check_collisions=False)

    gap_env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=1)

    agent_names = ['Opponent'] + [rr['name'] for rr in race_results]

    renderer = Renderer(agent_names)
    env.add_render_callback(renderer.callback)

    gap_obs = gap_env.reset(poses=start_poses_array[:1])[0]
    obs = env.reset(poses=start_poses_array)[0]
    env.render(mode='human_fast')

    renderer.positions = [(start_poses[0][0], start_poses[0][1])] * (num_agents + 1)
    
    frame = 0
    opp_index = 0
    extra_frames = -1

    gap_env.sim.agents[0] = deepcopy(opp_start_tuples[opp_index][0])
    opp_driver = opp_start_tuples[opp_index][1]

    while True:
        frame += 1

        if extra_frames >= 0:
            extra_frames -= 1
        
        #run gap follower logic
        actions = np.zeros((1, 2))
        
        gap_odom = pack_odom(gap_obs, 0)
        gap_scan = gap_obs['scans'][0]

        if hasattr(opp_driver, 'process_observation'):
            speed, steer = opp_driver.process_observation(ranges=gap_scan, ego_odom=gap_odom)
        else:
            assert hasattr(opp_driver, 'process_lidar')
            speed, steer = opp_driver.process_lidar(gap_scan)

        actions[0, 0] = steer
        actions[0, 1] = speed

        gap_obs = gap_env.step(actions)[0]

        pose = get_pose(gap_obs, 0)
        renderer.positions[0] = pose[0], pose[1]
        env.render_obs['poses_x'][0] = pose[0]
        env.render_obs['poses_y'][0] = pose[1]
        env.render_obs['poses_theta'][0] = pose[2]

        ## update other cars
        non_timeout_exists = False
        
        for i, rr in enumerate(race_results):
            # for each driver
            
            replay = rr['replay_list'][opp_index]
            # current replay

            if frame >= len(replay):
                if rr['results'][opp_index] == 'crash':
                    pose = replay[-1]
                else:
                     # overtake makes car disappear
                    pose = (1000, 1000, 0)
            else:
                pose = replay[frame]

                if rr['results'][opp_index] != 'timeout':
                    non_timeout_exists = True

            env.render_obs['poses_x'][i+1] = pose[0]
            env.render_obs['poses_y'][i+1] = pose[1]
            env.render_obs['poses_theta'][i+1] = pose[2]

            renderer.positions[i+1] = (pose[0], pose[1])

        env.render(mode='human_fast')

        if not non_timeout_exists and extra_frames < 0:
            extra_frames = 200
            
        if extra_frames == 0:
            # move on to next one
            frame = 0
            opp_index += 1

            if opp_index >= len(opp_start_tuples):
                print("Done!")
                break
            
            print(f"advancing to next race {opp_index+1}/{len(opp_start_tuples)}")

            gap_env.sim.agents[0] = deepcopy(opp_start_tuples[opp_index][0])
            opp_driver = opp_start_tuples[opp_index][1]
