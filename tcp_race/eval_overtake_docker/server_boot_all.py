"""
TCP race code for server
Stanley Bak
"""

from typing import Optional, Any, List, Tuple

import socket
import os
import subprocess
import multiprocessing

import pyglet

from f110_gym.envs.base_classes import RaceCar
from gap_driver import Driver

from networking import send_object, recv_object
from worker import worker_func
from start_states import get_opp_start_states

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

    #### make / load opponent start states

    #opp_start_states7: List[Tuple[RaceCar, Driver]] = get_opp_start_states(racetrack, start_poses,
    #                                                    gain=7, num_overtake_scenarios=10)
    opp_start_states: List[Tuple[RaceCar, Driver]] = get_opp_start_states(racetrack, start_poses,
                                                        gain=8, num_overtake_scenarios=10)

    print("!! development: only using first two opp positions")
    opp_start_states = opp_start_states[:1]

    docker_processes, sockets = connect_docker(name_tarfiles)

    ###############
    print("Starting parallel simulators...")
    #num_agents = len(name_tarfiles)
    #pool = multiprocessing.Pool(num_agents)
    pool = multiprocessing.Pool(os.cpu_count() // 2)

    pool_params = []

    for i, sock in enumerate(sockets):
        driver_name = name_tarfiles[i][0]
        tup = (racetrack, start_poses, sock, driver_name, opp_start_states)
        pool_params.append(tup)

    race_results = pool.map(worker_func, pool_params)

    print("Sending exit command to all docker subprocesses...")

    for sock in sockets:
        send_object(sock, {"type": "exit"})

    print_results(race_results)

    ############
    print("Waiting for docker subprocesses to exit...")
                      
    for p in docker_processes:
        p.wait()

def print_results(race_results):
    """print out results"""

    sorted_race_results = []

    for rr in race_results:
        num_overtakes = 0
        overtake_frames = 0
        num_crashes = 0
        num_timeouts = 0

        for r in rr['results']:
            if r == "timeout":
                num_timeouts += 1
            elif r == "crash":
                num_crashes += 1
            else:
                assert isinstance(r, int)
                num_overtakes += 1
                overtake_frames += r

        if num_overtakes == 0:
            avg_overtake_time = '-'
        else:
            avg_overtake_time = f"{round(overtake_frames / num_overtakes / 100, 2)}s"

        runtime = f"{round(rr['runtime'], 1)}s"

        tup = num_overtakes, avg_overtake_time, num_timeouts, num_crashes, runtime, rr['name']
        sorted_race_results.append(tup)

    sorted_race_results.sort(reverse=True)

    for num_overtakes, avg_overtake_time, num_timeouts, num_crashes, runtime, name in sorted_race_results:
        print(f"Driver: {name}, C:{num_crashes}, TO:{num_timeouts}, OV:{num_overtakes}, " + \
              f"OV_SEC:{avg_overtake_time}, Runtime: {runtime}")

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

    #name_tarfiles = [name_tarfiles[1]] # SBU 
    #name_tarfiles = [name_tarfiles[7]] # DS Play
    #print(name_tarfiles)

    #name_tarfiles = name_tarfiles[:2]
    #name_tarfiles = name_tarfiles[1:]    

    start_server(name_tarfiles)

if __name__ == "__main__":
    main()
