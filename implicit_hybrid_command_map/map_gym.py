"""
code to make implicit hybrid command map in gym environment
"""

from math import sqrt
from copy import deepcopy

import yaml
from argparse import Namespace

import numpy as np

from f110_gym.envs import F110Env

from map_gui import MapGui

class MapGymSim:
    'simulation state for mapping'

    map_config_dict = None
    render_on = True

    def __init__(self, ego_planner, opp_planner, use_lidar, config_file):
        self.ego_planner = ego_planner
        self.opp_planner = opp_planner

        self.render_follow_ego = False

        # config
        with open(config_file) as f:
            conf_dict = yaml.load(f, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        # env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)

        num_beams = 1080 if use_lidar else 32
        env = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=2, num_beams=num_beams)

        map_config_path = f"{conf.map_path}.yaml"

        with open(map_config_path) as f:
            self.map_config_dict = yaml.load(f, Loader=yaml.FullLoader)

        env.add_render_callback(self.render_callback)

        lanes = np.loadtxt(conf.lanes_path, delimiter=conf.lanes_delim, skiprows=conf.wpt_rowskip)
        center_lane_index = 1
        self.center_lane = lanes[:, center_lane_index*3:center_lane_index*3+2]

        self.next_cmds = []
        
        #env.render()
        start_list = [[conf.sx, conf.sy, conf.stheta]]
        start_list.append([conf.sx2, conf.sy2, conf.stheta2])
        self.start_positions = np.array(start_list, dtype=float)

        # doing this will assign render_obs in environment, which is needed at the root
        env.reset(self.start_positions)
        self.first_step = True

        self.map = conf.map_path
        self.error = False
        self.num_steps = 0
        self.env = env

        self.finished_initial_lap = False
        self.planner_position_list = [] # list of x, y, planner_obj

    def step_initial_lap(self):
        """run one step during the initial lap"""

        self.step_sim(freeze_ego = True)

        assert not self.error, "Crash during initial lap (single car)"

        opp_x = self.env.render_obs['poses_x'][0]
        opp_y = self.env.render_obs['poses_y'][0]
        opp_percent = self.percent_completed(opp_x, opp_y)

        tup = (opp_x, opp_y, deepcopy(self.opp_planner))
        self.planner_position_list.append(tup)

        if opp_percent > 95:
            self.finished_initial_lap = True

        return opp_x, opp_y

    def step_sim(self, substeps=100, freeze_ego=False):
        """step the simulation state"""

        assert not self.error

        for _ in range(substeps):

            if self.first_step:
                self.first_step = False

                # reset again, to get obs
                obs, _step_reward, done, _info = self.env.reset(self.start_positions)
                speed, steer = self.ego_planner.plan(obs, 0)
                self.next_cmds = [[steer, speed]]

                opp_speed, opp_steer = self.opp_planner.plan(obs, 1)
                self.next_cmds.append([opp_steer, opp_speed])

            assert self.next_cmds is not None
            obs, _step_reward, done, _info = self.env.step(np.array(self.next_cmds))

            if MapGymSim.render_on:
                self.env.render(mode='human_fast')

            if np.any(obs['collisions']):
                self.error = True # someone crashed
                break

            speed, steer = self.ego_planner.plan(obs, 0)

            if freeze_ego:
                steer, speed = 0, 0
            
            self.next_cmds = [[steer, speed]]

            opp_speed, opp_steer = self.opp_planner.plan(obs, 1)
            self.next_cmds.append([opp_steer, opp_speed])

        self.num_steps += 1

    def get_status(self):
        "get simulation status. element of ['ok', 'stop', 'error']"

        # key is obs['collisions'][1] is true if opponent crashes

        ego_x = self.env.render_obs['poses_x'][0]
        ego_y = self.env.render_obs['poses_y'][0]
        ego_percent = self.percent_completed(ego_x, ego_y)

        if self.error:
            rv = 'error'
        elif ego_percent > 95:
            rv = 'stop'
        else:
            rv = 'ok'

        return rv

    def get_map_pos(self)->np.ndarray:
        """get the map position the state

        returns a np.array of float objects
        """

        ego_x = self.env.render_obs['poses_x'][0]
        ego_y = self.env.render_obs['poses_y'][0]

        return np.array([ego_x, ego_y], dtype=float)

    def percent_completed(self, own_x, own_y):
        'find the percent completed in the race just using the closest waypoint'

        num_waypoints = self.center_lane.shape[0]

        min_dist_sq = np.inf
        rv = 0

        # min_dist_sq is the squared sum of the two legs of a triangle
        # considering two waypoints at a time

        for i in range(len(self.center_lane) - 2):
            x1, y1 = self.center_lane[i]
            x2, y2 = self.center_lane[i + 1]

            dx1 = (x1 - own_x)
            dy1 = (y1 - own_y)

            dx2 = (x2 - own_x)
            dy2 = (y2 - own_y)
            
            dist_sq1 = dx1*dx1 + dy1*dy1
            dist_sq2 = dx2*dx2 + dy2*dy2
            dist_sq = dist_sq1 + dist_sq2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                # 100 * min_index / num_waypoints
                
                rv = 100 * i / num_waypoints

                # add the fraction completed betwen the waypints
                dist1 = sqrt(dist_sq1)
                dist2 = sqrt(dist_sq2)
                frac = dist1 / (dist1 + dist2)

                assert 0.0 <= frac <= 1.0
                rv += frac / num_waypoints

        return rv

    def render_callback(self, env_renderer):
        'custom extra drawing function'

        e = env_renderer

        # update camera to follow car
        if self.render_follow_ego:
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
        else:
            x = e.cars[1].vertices[::2]
            y = e.cars[1].vertices[1::2]

        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

def make_map(driver_class, config="config.yaml", use_lidar=True):
    """make the map images"""

    ego_driver = driver_class()
    opp_driver = driver_class()

    gym_sim = MapGymSim(ego_driver, opp_driver, use_lidar, config)

    gui = MapGui(gym_sim)

    gui.show_gui()
