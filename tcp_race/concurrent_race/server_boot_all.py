"""
TCP race code for server
Stanley Bak
"""

from typing import Optional, Any, List

import os
import sys
import socket
import subprocess
import multiprocessing
import time

import numpy as np
import gym
import pyglet

from networking import send_object, recv_object
from my_laser_models import MyScanSimulator2D

def pack_odom(obs, i):
    """create single-car odometry from multi-car odometry"""

    keys = {
        'poses_x': 'pose_x',
        'poses_y': 'pose_y',
        'poses_theta': 'pose_theta',
        'linear_vels_x': 'linear_vel_x',
        'linear_vels_y': 'linear_vel_y',
        'ang_vels_z': 'angular_vel_z',
    }
    return {single: obs[multi][i] for multi, single in keys.items()}

class Renderer:

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

def init_scanner(func, map_path, yaml_path):
    """init func for multiprocessing pool"""

    func.scanner = MyScanSimulator2D(map_path, yaml_path)

def scanner_func(param):
    """get scan func for multiprocessing pool"""

    if param is None:
        return 'param-was-None-crashed?'

    sock, odom = param

    pose = np.array([odom['pose_x'], odom['pose_y'], odom['pose_theta']], dtype=float)

    scan = scanner_func.scanner.scan(pose)

    msg = {'type': 'obs', 'odom': odom, 'scan': list(scan)}
    send_object(sock, msg)

    obj = recv_object(sock)

    return obj

def start_server(name_tarfiles):
    """start server"""
    
    racetrack = 'SOCHI'

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    num_agents = len(name_tarfiles)

    start_poses = np.array([[0.8007017, -0.2753365, 4.1421595]] * num_agents)

    parent = os.path.dirname(os.path.realpath(__file__))
    map_path = os.path.join(parent, 'maps', f'{racetrack}.png')
    yaml_path = os.path.join(parent, 'maps', f'{racetrack}.yaml')

    # wait for connections    
    sockets: List[Optional[Any]] = [None] * num_agents

    # odom = pack_odom(obs, i)
    num_connections = 0
    processes = []

    pool = multiprocessing.Pool(num_agents, initializer=init_scanner, initargs=(scanner_func, map_path, yaml_path))
    
    #lidar_processes = []
    #lidar_process = multiprocessing.Process(target=lidar_func, args=(index, shared))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('localhost', 0))
        server_socket.listen()
        port = server_socket.getsockname()[1]
                
        print(f"Server listening on port {port}. Waiting for {len(sockets)} connections...")

        # spawn subprocesses for each docker container
        for i, (name, tarfile) in enumerate(name_tarfiles):
            p = start_docker(i, name, tarfile, port)
            processes.append(p)

        print("Started all subprocesses")

        while num_connections < len(sockets):
            client_socket, client_address = server_socket.accept()

            obj = recv_object(client_socket)
            assert obj is not None, "client socked disconnected before init"
            
            assert obj['type'] == 'init'

            driver_index = int(obj['driver_index'])
            assert sockets[driver_index] is None, "client socket connected twice?"
            sockets[driver_index] = client_socket
            num_connections += 1

            driver_name = name_tarfiles[driver_index][0]
            print(f"got connection from {client_address} for driver #{driver_index} ({driver_name})")
            print("Missing:", end='')

            for name_tarfile, sock in zip(name_tarfiles, sockets):
                if sock is None:
                    print(f" {name_tarfile[0]}", end='')

            print()

    for s in sockets:
        assert s is not None

    crashed = [False] * num_agents
    frames = 0

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, racetrack),
                   map_ext=".png", num_agents=num_agents, num_beams=10, should_check_collisions=False)

    agent_names = [a[0] for a in name_tarfiles]
    renderer = Renderer(agent_names)
    env.add_render_callback(renderer.callback)

    obs, step_reward, done, info = env.reset(poses=start_poses)
    env.render(mode='human_fast')
    
    while True:
        actions = np.zeros((num_agents, 2))

        loop_start = start = time.perf_counter()
        ### do scans in parallel
        params = []

        for i in range(num_agents):
            if obs['collisions'][i] and not crashed[i]:
                crashed[i] = True
                print(f"Car {i} crashed!")

            if crashed[i]:
                params.append(None)
            else:
                odom = pack_odom(obs, i)            
                params.append((sockets[i], odom))
        
        recv_obj_list = pool.map(scanner_func, params)

        diff = time.perf_counter() - start
        print(f"\npar-run time: {round(diff * 1000, 1)}ms")
        start = time.perf_counter()

        max_computation_time = [-1, -1]

        for i, obj in enumerate(recv_obj_list):
            if crashed[i]:
                continue
            
            if obj is None:
                print(f"driver #{i} disconnected")
                break
            
            assert obj['type'] == 'actions'
            
            actions[i, 0] = obj['steer']
            actions[i, 1] = obj['speed']

            computation_time = obj['computation_time']

            if computation_time > max_computation_time[0]:
                max_computation_time[0] = computation_time
                max_computation_time[1] = name_tarfiles[i][0]

        diff = time.perf_counter() - start
        print(f"stats time: {round(diff * 1000, 1)}ms")

        t, name = max_computation_time
        print(f"max computation time: {round(t * 1000, 1)} ms by {name}")
        start = time.perf_counter()

        obs, step_reward, done, info = env.step(actions)
        frames += 1

        # update positions for renderer
        positions = []
        
        for i in range(num_agents):
            pos = (obs['poses_x'][i], obs['poses_y'][i])
            positions.append(pos)

        renderer.positions = positions

        diff = time.perf_counter() - start
        print(f"env.step() time: {round(diff * 1000, 1)}ms")
        start = time.perf_counter()
        
        env.render(mode='human_fast')

        diff = time.perf_counter() - start
        print(f"render time: {round(diff * 1000, 1)}ms")

        diff = time.perf_counter() - loop_start
        print(f"full loop time: {round(diff * 1000, 1)}ms")

        toggle_list = info['checkpoint_done']

        if True in toggle_list:
            winner = list(toggle_list).index(True)
            winner_name = name_tarfiles[winner][0]
            
            print(f"Race completed in {round(frames / 100, 2)} sec. Winner: {winner_name} ({winner})")
            break

        if all(crashed):
            print("All cars crashed!")
            break

    for sock in sockets:
        send_object(sock, {"type": "exit"})

    for p in processes:
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

    name_tarfiles[0], name_tarfiles[6] = name_tarfiles[6], name_tarfiles[0]

    #name_tarfiles = name_tarfiles[:2]
    #name_tarfiles = name_tarfiles[1:]

    start_server(name_tarfiles)

if __name__ == "__main__":
    main()
