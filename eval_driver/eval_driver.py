"""
Code to evaluate a driver.

Provides a score for:

overtakes (% successful overtakes):
crashes (% crashes during overtaking):
"""

from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import os
import time

import numpy as np

import gym

from gap_driver import GapFollower
from f110_gym.envs.base_classes import RaceCar
from gap_driver import Driver
from start_states import get_opp_start_states, get_ego_start_states
from eval_scenario import eval_scenario

def compute_scores(ego_driver, opp_driver, num_overtake_scenarios):
    """compute the scores for the driver"""

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    racetrack = 'SOCHI'

    poses = np.array([
                [0.8007017, -0.2753365, 4.1421595],
                [0.8162458, 1.1614572, 4.1446321],
            ])

    env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, racetrack),
                       map_ext=".png", num_agents=2)

    custom_score_text_list = []

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

        e.score_label.x = (e.left + e.right) / 2 - 150
        e.score_label.y = e.bottom + 100

        if custom_score_text_list:
            e.score_label.text = custom_score_text_list[0]

    env.add_render_callback(render_callback)

    #opp_driver_class = type(opp_driver).__name__
    opp_cache_filename = f"cache/opp_starts.pkl"
    opp_start_states: List[Tuple[RaceCar, Driver]] = get_opp_start_states(env, racetrack, poses, opp_driver,
                                                                          num_overtake_scenarios, opp_cache_filename)

    ego_cache_filename = f"cache/ego_starts.pkl"
    ego_start_states: List[Tuple[RaceCar, Driver]] = get_ego_start_states(env, racetrack, poses, ego_driver,
                                                                          opp_start_states, ego_cache_filename)

    # ok, now setup each scenario and evaluate it
    rv: Dict[str, List[int]] = defaultdict(list)

    start = time.perf_counter()
    for i, (ego_start, opp_start) in enumerate(zip(ego_start_states, opp_start_states)):
        result = eval_scenario(ego_start, opp_start, env, racetrack)

        assert not custom_score_text_list
        custom_score_text_list.append(result.capitalize())
        env.render(mode='human_fast')
        time.sleep(1)
        custom_score_text_list.pop()
        #time.sleep(4)
        #exit(1)

        if result == "overtake":
            rv['overtakes'].append(i)
        elif result == "crash":
            rv['crashes'].append(i)
        else:
            assert result == "overtake_timeout"
            rv['overtake_timeouts'].append(i)

    diff = time.perf_counter() - start
    print(f'overtake evaluation completed in {round(diff, 1)} sec')

    return rv

def main():
    """main entry point"""

    ego_driver = GapFollower(8.0)
    opp_driver = GapFollower(6.0)

    num_overtake_scenarios = 20

    scores = compute_scores(ego_driver, opp_driver, num_overtake_scenarios)

    for key, val in scores.items():
        print(f"{key} ({len(val)}): {val}")
    

if __name__ == "__main__":
    main()
