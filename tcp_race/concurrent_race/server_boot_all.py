"""
TCP race code for server
Stanley Bak
"""

from typing import Optional, Any, List

import socket
import os
import sys
import subprocess
import multiprocessing
import pickle
import time

import numpy as np
import gym
import pyglet

from networking import send_object, recv_object
from worker import init_worker, worker_func

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

def connect_docker(name_tarfiles):
    """start docker containers and connect to them with a tcp port"""

    num_agents = len(name_tarfiles)

    # wait for connections    
    sockets: List[Optional[Any]] = [None] * num_agents

    # odom = pack_odom(obs, i)
    docker_processes = []
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('localhost', 0))
        server_socket.listen()
        port = server_socket.getsockname()[1]
                
        print(f"Server listening on port {port}. Waiting for {len(sockets)} connections...")

        # spawn subprocesses for each docker container
        for i, (name, tarfile) in enumerate(name_tarfiles):
            p = start_docker(i, name, tarfile, port)
            docker_processes.append(p)

        print("Started all subprocesses")

        while True:
            client_socket, client_address = server_socket.accept()

            obj = recv_object(client_socket)
            assert obj is not None, "client socked disconnected before init"
            
            assert obj['type'] == 'init'

            driver_index = int(obj['driver_index'])
            assert sockets[driver_index] is None, "client socket connected twice?"
            sockets[driver_index] = client_socket

            driver_name = name_tarfiles[driver_index][0]
            print(f"got connection from {client_address} for driver #{driver_index} ({driver_name})")

            missing_names = []
            
            for name_tarfile, sock in zip(name_tarfiles, sockets):
                if sock is None:
                    missing_names.append(name_tarfile[0])

            if missing_names:
                print(f"Missing ({len(missing_names)}): {', '.join(missing_names)}")
            else:
                print("All drivers connected.")
                break

    return docker_processes, sockets

def start_server(name_tarfiles):
    """start server"""
    
    racetrack = 'SOCHI'
    start_poses = [[0.8007017, -0.2753365, 4.1421595], [0.8162458, 1.1614572, 4.1446321]]
    num_agents = len(name_tarfiles)
    pickled_filename = 'cached_race.pkl'

    try:
        with open(pickled_filename, 'rb') as f:
            result_list = pickle.load(f)

        print(f"Loaded race results from {pickled_filename}")
    except FileNotFoundError:
        result_list = None

    if result_list is None:
        docker_processes, sockets = connect_docker(name_tarfiles)

        ###############
        print("Starting parallel simulators...")
        pool = multiprocessing.Pool(num_agents, initializer=init_worker, initargs=(worker_func, racetrack, start_poses))

        pool_params = []

        for i, sock in enumerate(sockets):
            driver_name = name_tarfiles[i][0]
            tup = (sock, driver_name)
            pool_params.append(tup)

        result_list = pool.map(worker_func, pool_params)

        print("Sending exit command to all docker subprocesses...")

        for sock in sockets:
            send_object(sock, {"type": "exit"})

        # save result_list to pickled file
        print(f"Saving pickled results to {pickled_filename}")
        
        with open(pickled_filename, 'wb') as f:
            pickle.dump(result_list, f)

    print("Rendering result of race...")
    ##########

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    start_poses_array = np.array([start_poses[0]] * (num_agents + 1))
    
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=num_agents+1, num_beams=10, should_check_collisions=False)

    agent_names = ['GapFollower(7)'] + [a[0] for a in name_tarfiles]
    renderer = Renderer(agent_names)
    env.add_render_callback(renderer.callback)

    obs, step_reward, done, info = env.reset(poses=start_poses_array)
    env.render(mode='human_fast')
    all_done = False
    frame = 0

    renderer.positions = [(start_poses[0][0], start_poses[0][1])] * (num_agents + 1)
    
    while True:
        all_done = True
        frame += 1

        positions = []

        for i, result in enumerate(result_list):
            pose_history = result['pose_history']
            
            if frame < len(pose_history):
                all_done = False
                if i == 0: # use gap follower as ego
                    pose = pose_history[frame][1]
                                    
                    env.render_obs['poses_x'][0] = pose[0]
                    env.render_obs['poses_y'][0] = pose[1]
                    env.render_obs['poses_theta'][0] = pose[2]
                    renderer.positions[i] = (pose[0], pose[1])
                    
                pose = pose_history[frame][0]

                env.render_obs['poses_x'][i+1] = pose[0]
                env.render_obs['poses_y'][i+1] = pose[1]
                env.render_obs['poses_theta'][i+1] = pose[2]
                renderer.positions[i+1] = (pose[0], pose[1])

        if all_done:
            break

        env.render(mode='human_fast')
        time.sleep(0.005)

    print("Waiting for docker subprocesses to exit...")
                      
    for p in docker_processes:
        p.wait()

def start_docker(index, name, tarfile, port):
    """start docker for a car"""

    # sanitize name so docker accepts it as a container name
    name = name.lower().replace(" ", "_")

    params = ["bash", "client_in_docker.sh", str(port), str(index), tarfile, name]

    #print(subprocess.run(params))
    p = subprocess.Popen(params, stdout=subprocess.DEVNULL, stdin=None, stderr=subprocess.STDOUT)
    #p = subprocess.Popen(params)
    print(f"started subprocess: {' '.join(params)}")

    return p

def main():
    """main entry point"""

    suffix = "-snapshot.tar.gz"

    current_dir = os.path.abspath(os.path.dirname(__file__))
    snapshots_dir = os.path.join(current_dir, "snapshots")

    name_tarfiles = []

    for filename in os.listdir(snapshots_dir):
        if filename.endswith(suffix):
            if "QC" in filename: # QC Pass doesn't use standard interface
                continue
            
            folder = filename[:-len(suffix)]

            tarfile = os.path.join("snapshots", filename)
            tup = (folder, tarfile)
            name_tarfiles.append(tup)

    #name_tarfiles[0], name_tarfiles[6] = name_tarfiles[6], name_tarfiles[0]

    #name_tarfiles = name_tarfiles[:2]
    #name_tarfiles = name_tarfiles[1:]

    start_server(name_tarfiles)

if __name__ == "__main__":

    print("TODO: debug why does ICE go inside of the opponent gap follower(7)?!?")
    time.sleep(2)
    main()
