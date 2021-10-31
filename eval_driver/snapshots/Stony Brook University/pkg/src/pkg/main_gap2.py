import time
import gym
import numpy as np
import concurrent.futures
import os
import sys
from math import sqrt


import matplotlib.pyplot as plt

# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# import your drivers here
from pkg.mygap2 import MyGap2Follower

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
            poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
            #poses = np.array([[0.8162458, 1.1614572, 4.1446321]])
        elif driver_count == 2:
            poses = np.array([
                [0.8007017, -0.2753365, 4.1421595],
                [0.8162458, 1.1614572, 4.1446321],
            ])
        else:
            raise ValueError("Max 2 drivers are allowed")

        extra_render_funcs = []

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

            x = (left + right) / 2
            y = (top + bottom) / 2
            
            for func in extra_render_funcs:
                func(x, y, env_renderer)

        env.add_render_callback(render_callback)

        loop = 0
        num_success = 0
            
        while num_success < 10:
            loop += 1

            if num_success != 0:
                env.sim.agents[0].seed = loop

            obs, step_reward, done, info = env.reset(poses=poses)

            gf = MyGap2Follower(tuning_gains=True)
            drivers = [gf]
            extra_render_funcs.clear()
            extra_render_funcs.append(gf.render)
            
            #drivers = [GapFollower()]

            env.render()

            laptime = 0.0
            start = time.time()

            crashed = False
            step = 0

            steers = []
            ang_vels = []
            steps = []
            steer_integrator_list = []
            real_steer_list = []
            steer_diff_list = []
            desired_steer_list = []

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

                real_steer_list.append(env.sim.agents[0].state[2])
                
                obs, step_reward, done, info = env.step(actions)
                laptime += step_reward

                assert gf.real_steer_predicted == env.sim.agents[0].state[2], \
                      f"predicted: {gf.real_steer_predicted}, actual: {env.sim.agents[0].state[2]}"

                steers.append(steer)
                ang_vels.append(ego_odom['angular_vel_z'])
                steps.append(step)
                steer_integrator_list.append(gf.desired_steer_integral)
                desired_steer_list.append(gf.desired_steer)
                
                #real_steer_list.append(env.sim.agents[0].state[2])
                steer_diff_list.append(env.sim.agents[0].steer_diff)

                env.render(mode='human_fast')

                if obs['collisions'][0]:
                    crashed = True

                step += 1

                if obs['poses_x'][0] < -60:
                    done = True

                #if step > 100:
                #    break

                # spinning round and round
                if abs(obs['ang_vels_z'][0]) > 20:
                    crashed = True
                    done = True

            print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

            #plt.plot(steps, np.array(ang_vels) / 10, label='ang_vel')
            plt.plot(steps, steers, 'k-o', label='steer cmd')
            plt.plot(steps, real_steer_list, label='real_steer')
            #plt.plot(steps, steer_diff_list, label='steer_diff')
            #plt.plot(steps, steer_integrator_list, label='desired steer integral')
            plt.plot(steps, desired_steer_list, label='desired steer')
           
                        
            plt.legend()
            #plt.show()
            #exit(1)
            plt.savefig('response.png')
            plt.clf()

            if crashed:
                gf.crashed() # update gains for next try

                num_success = 0
            else:
                num_success += 1
                print(f"Num success: {num_success}")

                break

if __name__ == '__main__':
    runner = GymRunner(RACETRACK)
    runner.run()
