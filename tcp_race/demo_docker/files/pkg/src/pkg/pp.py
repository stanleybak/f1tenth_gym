"""
pure pursuit
"""

import os
import pickle
import math
import time

import yaml

import numpy as np

from numba import njit

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase, vgain):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2] #lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    # print(speed, steering_angle)
    return speed*vgain, steering_angle

class MyPurePursuitPlanner:
    """
    Example Planner
    """
    # for tuning, can't be instance variables since we reset
    LAST_CRASH_POS = (0, 0)
    SAME_POS_CRASH_COUNT = 0

    def __init__(self, tuning_gains, wb=0.17145+0.15875):
        self.wheelbase = wb

        # dictionation of positions -> gains
        self.target_gain = 1.5
        self.tuning_gains = tuning_gains
        self.last_crash = (0, 0)
        self.gain_dict = {}
        self.crashes_waypoints ={}
        self.pos_history = [] # list of pairs
        self.SPEED_GAIN = self.target_gain

        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'Sochi_raceline.csv')
        path_gains = os.path.join(parent, 'gains_pp.pkl')
        path_crashes = os.path.join(parent, 'gains_pp_crashes.pkl')

        waypoints = [(0.0, 0.0)]
        
        try:
            with open(path, "rb") as f:
                waypoints = np.loadtxt(path, delimiter=';', skiprows=1)
                print(f"loaded {len(waypoints)} from file")
        except FileNotFoundError:
            print("no waypoint file found")
            assert False

        try:
            with open(path_gains, "rb") as f:
                self.gain_dict = pickle.load(f)
                print(f"loaded {len(self.gain_dict)} gains from file")
        except FileNotFoundError:
            print("no gain file found")
            # assert self.tuning_gains

        try:
            with open(path_crashes, "rb") as f:
                self.crashes_waypoints = pickle.load(f)
                print(f"loaded {len(self.crashes_waypoints)} Crashes counts from file")
        except FileNotFoundError:
            print("no gain file found")

        self.waypoints = waypoints
        
        self.max_reacquire = 20.

        self.drawn_waypoints = []

        self.tuning_gains = tuning_gains

        self.lookahead_distance = 1.3

        self.longer_lookahead_count = 0

    def render_waypoints(self, e):
        """slow_s
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, 0, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        
        wpts = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T
        
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        current_waypoint = np.empty((3, ))

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, 5]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, 5])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)
        self.update_params(lookahead_point)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase, self.SPEED_GAIN)

        # print(speed, steering_angle)
        return speed, steering_angle

    def process_observation(self, ranges, ego_odom):
        """ process observation"""
        x = ego_odom['pose_x']
        y = ego_odom['pose_y']
        theta = ego_odom['pose_theta']

        lookahead_distance = self.lookahead_distance

        if self.longer_lookahead_count > 0:
            self.longer_lookahead_count -= 1
            self.lookahead_distance = 2.5
            print("longer lookahead!")

        return self.plan(x, y, theta, lookahead_distance, self.SPEED_GAIN)

    def crashed(self):
        """ car crashed, update gains """

        assert self.tuning_gains, "crashed not while tuning gains"

        last_pos = self.pos_history[-1]

        time.sleep(0.5)

        if last_pos not in self.crashes_waypoints:
            MyPurePursuitPlanner.LAST_CRASH_POS = last_pos
            MyPurePursuitPlanner.SAME_POS_CRASH_COUNT = 20
            self.crashes_waypoints[last_pos] = 20

            print(f"Crash was in a new location: {last_pos}")
        else:
            self.crashes_waypoints[last_pos] +=50
            count = self.crashes_waypoints[last_pos]
            print(f"Crashed in same location as last time ({last_pos}), crash_count = {count}")
        print(self.crashes_waypoints)
        count = self.crashes_waypoints[last_pos]

        # update <count> historic gains

        index = len(self.pos_history) - 1

        for i in range(count):
            if index - i < 0:
                break

            pos = self.pos_history[index - i]

            # reduce gain
            min_gain = 1.0
            new_gain = (9 * self.gain_dict[pos] + min_gain) / 10
            if new_gain < min_gain + 0.001:
                new_gain = min_gain

            self.gain_dict[pos] = new_gain
            print(f"reduced gain for {pos} to {self.gain_dict[pos]}")

        # save to file
        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'gains_pp.pkl')

        raw = pickle.dumps(self.gain_dict)

        with open(path, "wb") as f:
            f.write(raw)

        raw2 = pickle.dumps(self.crashes_waypoints)
        path2 = os.path.join(parent, 'gains_pp_crashes.pkl')

        with open(path2, "wb") as f:
            f.write(raw2)

    def update_params(self, waypoint):
        """update gains based on position"""


        # if self.x < -20 and self.BEST_POINT_CONV_SIZE == 20:
        #    self.BEST_POINT_CONV_SIZE = 80

        pt = (waypoint[0], waypoint[1])
        first = False

        if len(self.pos_history) < 10:
            # start
            pt = (pt[0], pt[1], 'start')

        if not self.pos_history:
            first = True
            self.pos_history.append(pt)

            if pt in self.gain_dict:
                self.SPEED_GAIN = self.gain_dict[pt]

        if pt != self.pos_history[-1] or first:
            # position was updated

            self.pos_history.append(pt)

            if pt not in self.gain_dict:
                cur_gain = 1.5
                self.slow_steering = True
            else:
                cur_gain = self.gain_dict[pt]

            # print(f"{pt} -> {cur_gain}")

            self.target_gain = cur_gain
            self.gain_dict[pt] = cur_gain

        # update gain to move towards target_gain
        self.SPEED_GAIN = self.target_gain
