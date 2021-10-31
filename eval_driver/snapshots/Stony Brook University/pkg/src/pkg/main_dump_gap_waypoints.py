import time
import gym
import numpy as np
import concurrent.futures
import os
import pickle
import sys
from math import sqrt

# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# import your drivers here
from pkg.mygap import MyGapFollower

# choose your drivers here (1-4)
#drivers = [DisparityExtender()]

# choose your racetrack here (SOCHI, SOCHI_OBS)
RACETRACK = 'SOCHI'

def _pack_odom(obs, i):
    keys = {
        'poses_x': 'pose_x',
        'poses_y': 'pose_y',
        'poses_theta': 'pose_theta',
        'linear_vels_x': 'linear_vel_x',
        'linear_vels_y': 'linear_vel_y',
        'ang_vels_z': 'angular_vel_z',
    }
    return {single: obs[multi][i] for multi, single in keys.items()}


class GymRunner(object):

    def __init__(self, racetrack):
        self.racetrack = racetrack

    def run(self):
        # load map
        driver_count = 1 #len(drivers)
                
        env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, RACETRACK),
                       map_ext=".png", num_agents=driver_count)

        # specify starting positions of each agent

        
        if driver_count == 1:
            #poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
            poses = np.array([[0.8162458, 1.1614572, 4.1446321]])
        elif driver_count == 2:
            poses = np.array([
                [0.8007017, -0.2753365, 4.1421595],
                [0.8162458, 1.1614572, 4.1446321],
            ])
        else:
            raise ValueError("Max 2 drivers are allowed")

        env.add_render_callback(render_callback)

        obs, step_reward, done, info = env.reset(poses=poses)

        gf = MyGapFollower(tuning_gains=False)
        drivers = [gf]
        #drivers = [GapFollower()]

        env.render()

        laptime = 0.0
        start = time.time()

        crashed = False
        step = 0

        waypoints = []

        while not done:
            actions = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                odom_0, odom_1 = _pack_odom(obs, 0), None
                if len(drivers) > 1:
                    odom_1 = _pack_odom(obs, 1)

                for i, driver in enumerate(drivers):
                    if i == 0:
                        ego_odom, opp_odom = odom_0, odom_1
                    else:
                        ego_odom, opp_odom = odom_1, odom_0

                    scan = obs['scans'][i]
                    if hasattr(driver, 'process_observation'):
                        futures.append(executor.submit(driver.process_observation, ranges=scan, ego_odom=ego_odom))
                    elif hasattr(driver, 'process_lidar'):
                        futures.append(executor.submit(driver.process_lidar, scan))

            for future in futures:
                speed, steer = future.result()
                actions.append([steer, speed])
                
            actions = np.array(actions)
            obs, step_reward, done, info = env.step(actions)
            
            laptime += step_reward
            env.render(mode='human_fast')

            if obs['collisions'][0]:
                crashed = True

            step += 1

            if step % 10 == 0:
                # save x / y for waypoints
                x = obs['poses_x'][0]
                y = obs['poses_y'][0]
                pt = x, y
                
                waypoints.append(pt)

            # spinning round and round
            if abs(obs['ang_vels_z'][0]) > 20:
                crashed = True
                done = True

        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

        if crashed:
            print("Crashed!")
            time.sleep(1)

        parent = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(parent, 'waypoints_gap.pkl')

        raw = pickle.dumps(waypoints)
        
        with open(path, "wb") as f:
            f.write(raw)

        print(f"saved {path}")

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

if __name__ == '__main__':
    runner = GymRunner(RACETRACK)
    runner.run()
