"""
TCP race code for server
Stanley Bak
"""

from typing import Optional, Any, List

import os
import sys
import socket

import gym
import numpy as np

from networking import send_object, recv_object

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

def start_server():
    """start server"""
    
    racetrack = 'SOCHI'

    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    num_agents = 2
    
    poses = np.array([
                [0.8007017, -0.2753365, 4.1421595],
                [0.8162458, 1.1614572, 4.1446321],
            ])

    assert poses.shape[0] <= num_agents
    
    env = gym.make('f110_gym:f110-v0',
                       map="{}/maps/{}".format(current_dir, racetrack),
                       map_ext=".png", num_agents=num_agents)

    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(poses=poses)
    env.render(mode='human_fast')

    # wait for connections
    
    sockets: List[Optional[Any]] = [None] * num_agents

    # odom = pack_odom(obs, i)
    num_connections = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(('localhost', 0))

        port = server_socket.getsockname()[1]
        print(f"Server listening on port {port}. Waiting for {len(sockets)} connections...")

        server_socket.listen()

        while num_connections < len(sockets):
            client_socket, client_address = server_socket.accept()

            obj = recv_object(client_socket)
            assert obj is not None, "client socked disconnected before init"
            
            assert obj['type'] == 'init'

            driver_index = int(obj['driver_index'])
            assert sockets[driver_index] is None, "client socket connected twice?"
            sockets[driver_index] = client_socket
            num_connections += 1

            print(f"got connection from {client_address} for driver #{driver_index}")

    assert sockets[0] is not None and sockets[1] is not None

    crashed = [False] * num_agents
    frames = 0
    
    while True:
        actions = np.zeros((2, 2))

        for i, sock in enumerate(sockets):

            if obs['collisions'][i] and not crashed[i]:
                crashed[i] = True
                print(f"Car {i} crashed!")

            if crashed[i]:
                continue
            
            odom = pack_odom(obs, i)
            scan = list(obs['scans'][i])

            msg = {'type': 'obs', 'odom': odom, 'scan': scan}
            send_object(sock, msg)

        for i, sock in enumerate(sockets):
            if crashed[i]:
                continue
            
            obj = recv_object(sock)

            if obj is None:
                print("driver #{i} disconnected")
                break
            
            assert obj['type'] == 'actions'
            
            actions[i, 0] = obj['steer']
            actions[i, 1] = obj['speed']

        obs, step_reward, done, info = env.step(actions)
        frames += 1
        env.render(mode='human_fast')

        toggle_list = info['checkpoint_done']

        if True in toggle_list:
            winner = list(toggle_list).index(True)
            print(f"Race completed in {round(frames / 100, 2)} sec. Winner: {winner}")
            break

        if all(crashed):
            print("All cars crashed!")
            break

    for sock in sockets:
        send_object(sock, {"type": "exit"})

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

def main():
    """main entry point"""

    start_server()

if __name__ == "__main__":
    main()
