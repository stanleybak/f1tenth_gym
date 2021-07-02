import time
import yaml
import gym
import numpy as np
import math
from argparse import Namespace

from pyglet.gl import GL_POINTS, GL_LINES

from pure_pursuit import nearest_point_on_trajectory, get_actuation, first_point_on_trajectory_intersecting_circle

class SmoothPurePursuitPlanner:
    """
    Example Planner
    """

    MAX_COUNTER = 50 # number of frames to smooth trajectories when setting waypoints (100 = 1 second)
    
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.

        # load waypoints
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.old_waypoints = self.waypoints.copy()
        self.lane_switch_counter = SmoothPurePursuitPlanner.MAX_COUNTER

    def update_waypoints(self, waypoints):
        self.old_waypoints = self.waypoints
        self.waypoints = waypoints
        self.lane_switch_counter = 0 

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            # return fixed value
            return 4.0, 0

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)
        self.lane_switch_counter += 1

        if self.lane_switch_counter < SmoothPurePursuitPlanner.MAX_COUNTER:
            lookahead_point_old = self._get_current_waypoint(self.old_waypoints, lookahead_distance, position, pose_theta)

            lookahead_point = np.array(lookahead_point, dtype=float)
            lookahead_point_old = np.array(lookahead_point_old, dtype=float)

            frac = self.lane_switch_counter / SmoothPurePursuitPlanner.MAX_COUNTER
            lookahead_point = lookahead_point * frac + lookahead_point_old * (1 - frac)

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle

class LaneSwitcherPlanner:
    'planner that combines lane switcher with pure pursuit'

    def __init__(self, conf, lanes, current_lane=1):

        num_lanes = int(lanes.shape[1] / 3 - 1)

        self.ls = LaneSwitcher(lanes, num_lanes, current_lane)
        self.ppp = SmoothPurePursuitPlanner(conf, conf.wheelbase)
        
        self.ppp.update_waypoints(lanes[:, current_lane*3:current_lane*3+3])

        self.lookahead_distance = conf.lookahead_distance * 2
        self.vgain = conf.vgain

        self.drawn_waypoints = []
        self.wp_color = [183, 193, 222]
        self.prev_decision = -1

    def plan(self, x, y, theta):
        'return speed, steer'

        return self.ppp.plan(x, y, theta, self.lookahead_distance, self.vgain)

    def update(self, ego_pose, opp_pose):
        'update state based on observations'

        ego_switcher = self.ls
        lanes = ego_switcher.lanes
        planner = self.ppp
        
        decision = ego_switcher.decision(*ego_pose, *opp_pose)

        if decision != self.prev_decision:
            self.prev_decision = decision
            
            # update current waypoint being followed
            if decision == 2:
                _, d, _, _ = nearest_point_on_trajectory(np.array([ego_pose[0], ego_pose[1]]), lanes[:, -3:-1])
                if d <= ego_switcher.switch_thresh:
                    planner.update_waypoints(lanes[:, -3:])
            elif decision == 3:
                opp_lane = ego_switcher._pose2lane(opp_pose[0], opp_pose[1])
                # print('opp_lane', opp_lane)
                planner.update_waypoints(lanes[:, 3*opp_lane:3*opp_lane+3])
            elif decision == -1 and ego_switcher.current_lane != 0:
                ego_switcher.current_lane -= 1
                planner.update_waypoints(lanes[:, 3*ego_switcher.current_lane:3*ego_switcher.current_lane+3])
            elif decision == 1 and ego_switcher.current_lane < ego_switcher.num_lanes:
                ego_switcher.current_lane += 1
                planner.update_waypoints(lanes[:, 3*ego_switcher.current_lane:3*ego_switcher.current_lane+3])

            

    def render_waypoints(self, env_renderer):
        'draw waypoints using EnvRenderer'

        e = env_renderer
        #points = self.waypoints

        points = self.ppp.waypoints[:, :2]

        #points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                #b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                #    ('c3B/stream', [183, 193, 222]))
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', self.wp_color))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

class LaneSwitcher:
    def __init__(self, lanes, num_lanes, current_lane, proximity_distance=2.5):
        # Assumptions for lanes vector layout: each lane occupies 3 columns (x, y, vel), last three columns is the raceline, previous are lanes from inner to outer, corresponding to increase in indexing
        self.lanes = lanes
        self.num_lanes =  num_lanes
        self.current_lane = current_lane
        self.pd = proximity_distance
        self.switch_thresh = 0.2

    def _rotation_matrix(self, angle, direction, point=None):
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self._unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.array(((cosa, 0.0,  0.0),
                      (0.0,  cosa, 0.0),
                      (0.0,  0.0,  cosa)),
                      dtype=np.float64)
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array((( 0.0,         -direction[2],  direction[1]),
                       ( direction[2], 0.0,          -direction[0]),
                       (-direction[1], direction[0],  0.0)),
                       dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M

    def _unit_vector(self, data, axis=None, out=None):
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def _transform(self, x, y, th, oppx, oppy, oppth):
        """
        Transforms oppponent world coordinates into ego frame

        Args:
            x, y, th (float): ego pose
            oppx, oppy, oppz (float): opponent position

        Returns:
            oppx_new, oppy_new, oppth_new (float): opponent pose in ego frame
        """
        rot = self._rotation_matrix(th, (0, 0, 1))
        homo = np.array([[oppx - x], [oppy - y], [0.], [1.]])
        # inverse transform
        rotated = rot.T @ homo
        rotated = rotated / rotated[3]
        return rotated[0], rotated[1]

    def _pose2quadrant(self, oppx, oppy):
        """
        Returns the quadrant the opponent is in
             x
         1 | 0 | 2
        y----*-----
         3 | 0 | 4

        Args:
            oppx, oppy (float): opponent position in ego frame

        Returns:
            quadrant (int): which area opponent is w.r.t. ego (check figure)
        """
        if oppx >= -0.65:
            if oppy >= 0.1:
                return 1
            elif oppy <= -0.1:
                return 2
            else:
                return 0
        else:
            if oppy >= 1.5:
                return 3
            elif oppy <= -1.5:
                return 4
            else:
                return 0

    def _pose2lane(self, oppx, oppy):
        """
        Returns the lane the opponent is in

        Args:
            oppx, oppy (float): opponent position in world frame

        Returns:
            lane (int): which lane the opponent is on
        """
        nearest_dist = np.inf
        nearest_lane = -1
        for i in range(self.num_lanes):
            lane_test = self.lanes[:, i*3:i*3+2]
            _, test_dist, _, _ = nearest_point_on_trajectory(np.array([oppx, oppy]), lane_test)
            if test_dist < nearest_dist:
                nearest_lane = i
                nearest_dist = test_dist
        return nearest_lane

    def decision(self, x, y, th, oppx, oppy, oppth):
        """
        Makes laneswitch decision, left or right right now

        Args:
            x, y, th (float): ego pose
            oppx, oppy, oppth (float): opponent pose

        Returns:
            move (int): -1, 0, or 1, indicating moving lanes left, stay, or right;
                        2, indicating switching to raceline when no opponent around
                        3, indicating switching to opponent's lane
        """

        # Check if opponent close by
        radius = np.linalg.norm(np.array([x - oppx, y - oppy]))
        if radius <= self.pd:
            # check quadrant in ego frame
            oppx_new, oppy_new = self._transform(x, y, th, oppx, oppy, oppth)
            quadrant = self._pose2quadrant(oppx_new, oppy_new)
            # print('quadrant: ', quadrant)
            # move to inner
            if quadrant == 1:
                return -1 if self.current_lane != 0 else 0
            # move to outer
            elif quadrant == 2:
                return 1 if self.current_lane != self.num_lanes - 1 else 0
            # behind, block according to opponent lane
            else:
                return 3
        else:
            # if opponent not in range, follow race line
            return 2

def center_screen(env_renderer):
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

def main():
    'main entry point'
    
    # config
    with open('config.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    lanes = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    # assuming 3 columns per lane (x, y, vel)
    #num_lanes = int(lanes.shape[1] / 3 - 1)
    # start on race line
    # current_lane = num_lanes + 1
    #current_lane = 1
    #ego_switcher = LaneSwitcher(lanes, num_lanes, current_lane)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2)

    for a in env.sim.agents:
        a.use_scan = False # scan is used for collision detection
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta],
                                                       [conf.sx2, conf.sy2, conf.stheta2]]))
    env.render()
    
    #planner = PurePursuitPlanner(conf, conf.wheelbase)
    #planner.update_waypoints(lanes[:, current_lane*3:current_lane*3+3])
    
    #opp_planner = PurePursuitPlanner(conf, conf.wheelbase)
    # use centerline for opp in this example
    #opp_planner.update_waypoints(lanes[:, 3:6])

    ego_planner = LaneSwitcherPlanner(conf, lanes)
    opp_planner = LaneSwitcherPlanner(conf, lanes)
    opp_planner.wp_color = [0, 255, 0]

    env.add_render_callback(center_screen)
    env.add_render_callback(ego_planner.render_waypoints)
    env.add_render_callback(opp_planner.render_waypoints)

    laptime = 0.0
    start = time.time()

    frames = 0
    # 1: opponent goes faster, -1: opponent goes slower, 0: opponent speed unmodified
    input_vector_scale_speed = [-1, -1, 1, 1, 0, 0, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

    while not done:
        # lane switch decision

        ego_pose = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
        opp_pose = obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]

        ego_planner.update(ego_pose, opp_pose)
        opp_planner.update(opp_pose, ego_pose)

        # print('decision', decision, 'current lane', ego_switcher.current_lane)

        speed, steer = ego_planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0])
        opp_speed, opp_steer = opp_planner.plan(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1])

        frames += 1
        input_index = frames // 100

        if input_index >= len(input_vector_scale_speed):
            print("finished processing input vector")
            break

        scale_speed = input_vector_scale_speed[input_index]
        
        opp_speed_multiplier = 1.0

        if scale_speed == -1:
            opp_speed_multiplier = 0.9
        elif scale_speed == 1:
            opp_speed_multiplier = 1.1
        
        obs, step_reward, done, info = env.step(np.array([[steer, speed],
                                                          [opp_steer, opp_speed * opp_speed_multiplier]]))
        laptime += step_reward
        
        
        env.render(mode='human_fast')
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()

