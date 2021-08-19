'''
Interface for fuzz tester using gym environment
'''

import time
import abc

from math import sqrt
from argparse import Namespace
from PIL import Image


import multiprocessing
import numpy as np
import yaml
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

from f110_gym.envs import F110Env

from fuzz_test_generic import SimulationState, run_fuzz_testing

class Driver(abc.ABC):
    """abstract class for planner"""

    @abc.abstractmethod
    def plan(self, obs, ego_index):
        """returns speed, steer, obs_list is a list of observations for each car"""

class F110GymSim(SimulationState):
    'simulation state for fuzzing'

    render_on = True
    map_config_dict = None
    pool = multiprocessing.Pool(2) # for parallel execution of planners

    @staticmethod
    def get_cmds():
        'get a list of commands (strings) that can be passed into step_sim'

        return ['opp_faster', 'opp_slower']

    @staticmethod
    def get_obs_data():
        '''get labels and ranges on observations

        returns:
            list of 3-tuples, label, min, max
        '''

        return ('Ego Completed Percent', 0, 20), ('Opponent Behind Percent', -2.5, 2.5)

    def __init__(self, ego_planner, opp_planner, use_lidar):
        self.ego_planner = ego_planner
        self.opp_planner = opp_planner

        # config
        with open('config.yaml') as f:
            conf_dict = yaml.load(f, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        # env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)

        num_beams = 1080 if use_lidar else 32
        env = F110Env(map=conf.map_path, map_ext=conf.map_ext, num_agents=2, num_beams=num_beams)

        map_config_path = f"{conf.map_path}.yaml"

        with open(map_config_path) as f:
            F110GymSim.map_config_dict = yaml.load(f, Loader=yaml.FullLoader)

        env.add_render_callback(render_callback)

        lanes = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        center_lane_index = 1
        self.center_lane = lanes[:, center_lane_index*3:center_lane_index*3+2]

        #env.render()
        self.start_positions = np.array([[conf.sx, conf.sy, conf.stheta],
                                         [conf.sx2, conf.sy2, conf.stheta2]])

        # doing this will assign render_obs in environment, which is needed at the root
        env.reset(self.start_positions)
        self.first_step = True

        self.map = conf.map_path
        self.error = False
        self.num_steps = 0
        self.next_cmds = None
        self.env = env

    def get_pickle_name(self):
        """get the name use for pickling to prevent class conflicts"""

        return type(self.ego_planner).__name__

    @staticmethod
    def make_map_artist(ax):
        """make an artist for plotting the map, usingthe passed-in axis (optional)"""

        map_config_dict = F110GymSim.map_config_dict

        # map static map artist
        image_path = map_config_dict['image']
        res = map_config_dict['resolution']
        origin = map_config_dict['origin']

        img = Image.open(image_path)
        img = img.convert("RGBA")

        pixdata = img.load()

        width, height = img.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (255, 255, 255, 255):
                    pixdata[x, y] = (255, 255, 255, 0)

        img = np.array(img)

        xsize, ysize = img.shape[0:2]
        
        x1 = origin[0]
        y1 = origin[1]
        assert origin[2] == 0

        xsize *= res
        ysize *= res

        box = Bbox.from_bounds(x1, y1, xsize, ysize)
        tbox = TransformedBbox(box, ax.transData)
        box_image = BboxImage(tbox, zorder=2)

        box_image.set_data(img)
        box_image.zorder = 0

        ax.add_artist(box_image)

        ax.set_xlim(x1, x1 + xsize)
        ax.set_ylim(y1, y1 + ysize)
        
        return box_image

    def render(self):
        'display visualization'

        self.env.render(mode='human')
        time.sleep(0.1)
            
    def step_sim(self, cmd):
        'step the simulation state'

        assert not self.error

        for _ in range(100):

            if self.first_step:
                self.first_step = False

                # reset again, to get obs
                obs, _step_reward, done, _info = self.env.reset(self.start_positions)
                speed, steer = self.ego_planner.plan(obs, 0)
                opp_speed, opp_steer = self.opp_planner.plan(obs, 1)
                self.next_cmds = [[steer, speed], [opp_steer, opp_speed]]

            assert self.next_cmds is not None
            obs, _step_reward, done, _info = self.env.step(np.array(self.next_cmds))

            if F110GymSim.render_on:
                self.env.render(mode='human_fast')

            if done:
                self.error = True # crashed!
                break

            speed, steer = self.ego_planner.plan(obs, 0)
            opp_speed, opp_steer = self.opp_planner.plan(obs, 1)

            if cmd == 'opp_slower':
                opp_speed *= 0.8
            elif cmd == 'opp_faster':
                opp_speed *= 1.2

            self.next_cmds = [[steer, speed], [opp_steer, opp_speed]]

        self.num_steps += 1

    def get_status(self):
        "get simulation status. element of ['ok', 'stop', 'error']"

        ego_x, _opp_x = self.env.render_obs['poses_x']
        ego_y, _opp_y = self.env.render_obs['poses_y']
        ego_percent = self.percent_completed(ego_x, ego_y)

        if self.error:
            rv = 'error'
        elif ego_percent > 95:
            rv = 'stop'
        else:
            rv = 'ok'

        return rv

    def get_obs(self):
        '''get observation of current state

        currently this is a pair, [perent_completed_ego, dist_opp_percent]
        '''

        ego_x, opp_x = self.env.render_obs['poses_x']
        ego_y, opp_y = self.env.render_obs['poses_y']

        ego_percent = self.percent_completed(ego_x, ego_y)
        opp_percent = self.percent_completed(opp_x, opp_y)

        opp_behind_percent = ego_percent - opp_percent

        return np.array([ego_percent, opp_behind_percent], dtype=float)

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

def render_callback(env_renderer):
    'custom extra drawing function'

    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800
    
def fuzz_test_gym(planner_class, use_rrt=True, use_lidar=True, render_on=True):
    'main entry point'

    F110GymSim.render_on = render_on

    ego_driver = planner_class()
    opp_driver = planner_class()

    gym_sim = F110GymSim(ego_driver, opp_driver, use_lidar)

    run_fuzz_testing(gym_sim, always_from_start=False)

