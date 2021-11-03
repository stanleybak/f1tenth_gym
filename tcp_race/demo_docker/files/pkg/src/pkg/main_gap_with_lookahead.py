import time
import gym
import numpy as np
import concurrent.futures
import os
import sys

from copy import deepcopy

# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# import your drivers here
from pkg.mygap import MyGapFollower
from pkg.my_racecar import MyRaceCar
from pkg.my_laser_models import MyScanSimulator2D

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

        loop = 0
        num_success = 0

        ######### lidar
        racetrack = 'SOCHI'
        parent = os.path.dirname(os.path.realpath(__file__))
        map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
        yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')
        scanner = MyScanSimulator2D(map_path, yaml_path)

        # list of tuples: (env, driver, racecar, step)
        saved_states = []
        env = None

        start = time.time()
        
        while num_success < 10:
            loop += 1

            if env is None or num_success != 0:
                # start a new simulation
                
                env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, RACETRACK),
                       map_ext=".png", num_agents=driver_count)

                env.add_render_callback(render_callback)
                        
                env.sim.agents[0].seed = loop
                obs, step_reward, done, info = env.reset(poses=poses)
                
                driver = MyGapFollower(tuning_gains=True)
                drivers = [driver]
                step = 0

                a = env.sim.agents[0]
                x = a.state[0]
                y = a.state[1]
                theta = a.state[4]

                racecar = MyRaceCar()
                pose = np.array([x, y, theta])
                racecar.reset(pose)
                car_pose = np.array([racecar.state[0], racecar.state[1], racecar.state[4]])
                    
                assert np.allclose(car_pose, pose), f"pose was {pose}, racecar state was {car_pose}"

                tup = (deepcopy(env), deepcopy(driver), deepcopy(racecar), deepcopy(obs), step)
                saved_states = [tup]
            else:
                # load saved env and driver
                backtrack_secs = 1 + MyGapFollower.SAME_POS_CRASH_COUNT // 2

                backtrack_secs = min(backtrack_secs, 5)

                for _ in range(backtrack_secs): # go back a few seconds
                    if len(saved_states) == 1: # don't pop first item
                        print("using init state")
                        tup = saved_states[0]
                        break
                    
                    tup = saved_states.pop()

                env, driver, racecar, obs, step = tup
                drivers[0] = driver
                driver.load_gain_dict() # load updated gains

                a = env.sim.agents[0]
                print(f"loaded env at step {step}")

            env.render()

            crashed = False
            done = False

            while not done:

                if step % 100 == 0:
                    tup = (deepcopy(env), deepcopy(driver), deepcopy(racecar), deepcopy(obs), step)
                    saved_states.append(tup)
                
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

                        old_scan = obs['scans'][i]
                        a = env.sim.agents[0]
                        env_pose = np.array([a.state[0], a.state[1], a.state[4]])                        
                        car_pose = np.array([racecar.state[0], racecar.state[1], racecar.state[4]])

                        assert np.allclose(car_pose, env_pose)
                        cur_scan = scanner.scan(car_pose)

                        scan_diff = np.max(np.abs(cur_scan - old_scan))

                        if scan_diff > 0.5:
                            scan = old_scan
                        else:
                            # copy and advance racecar
                            virtual_racecar = deepcopy(racecar)
                            virtual_driver = deepcopy(driver)
                            speed, steer = virtual_driver.process_observation(cur_scan, ego_odom)

                            # 2 -> 83
                            for _ in range(2):
                                virtual_racecar.update_pose(steer, speed)

                                virtual_pose = np.array([virtual_racecar.state[0], virtual_racecar.state[1], virtual_racecar.state[4]])
                                virtual_scan = scanner.scan(virtual_pose)

                                virtual_ego_odom = {}
                                virtual_ego_odom['pose_x'] = virtual_racecar.state[0]
                                virtual_ego_odom['pose_y'] = virtual_racecar.state[1]
                                virtual_ego_odom['pose_theta'] = virtual_racecar.state[4]

                                speed, steer = virtual_driver.process_observation(virtual_scan, virtual_ego_odom)

                            virtual_pose = np.array([virtual_racecar.state[0], virtual_racecar.state[1], virtual_racecar.state[4]])
                            virtual_scan = scanner.scan(virtual_pose)
                            scan = virtual_scan

                        if hasattr(driver, 'process_observation'):
                            futures.append(executor.submit(driver.process_observation, ranges=scan, ego_odom=ego_odom))
                        elif hasattr(driver, 'process_lidar'):
                            futures.append(executor.submit(driver.process_lidar, scan))

                for future in futures:
                    speed, steer = future.result()
                    
                    actions.append([steer, speed])
                    
                actions = np.array(actions)
                
                obs, step_reward, done, info = env.step(actions)

                # update racecar
                racecar.update_pose(*actions[0])
                
                env.render(mode='human_fast')

                if obs['collisions'][0]:
                    crashed = True

                step += 1

                # spinning round and round
                if abs(obs['ang_vels_z'][0]) > 20:
                    crashed = True
                    done = True

                #if obs['poses_x'][0] < -140:
                #    done = True

            print('Sim elapsed time:', step * 0.01, 'Real elapsed time:', time.time() - start)

            if crashed:
                driver.crashed() # update gains for next try

                num_success = 0
            else:
                num_success += 1
                print(f"Num success: {num_success}!")

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
