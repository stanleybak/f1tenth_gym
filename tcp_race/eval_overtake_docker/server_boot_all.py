"""
TCP race code for server
Stanley Bak
"""

from typing import Optional, Any, List, Tuple

import socket
import os
import subprocess
import multiprocessing
import pickle

import pyglet

from f110_gym.envs.base_classes import RaceCar
from gap_driver import Driver

from networking import send_object, recv_object
from worker import worker_func
from start_states import get_opp_start_states
from replay_results import replay_results

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
    gap_gain = 8
    pickled_filename = f"gain{gap_gain}_results.pkl"

    #### make / load opponent start states

    #opp_start_states7: List[Tuple[RaceCar, Driver]] = get_opp_start_states(racetrack, start_poses,
    #                                                    gain=7, num_overtake_scenarios=10)
    opp_start_states: List[Tuple[RaceCar, Driver]] = get_opp_start_states(racetrack, start_poses,
                                                        gain=gap_gain, num_overtake_scenarios=10)

    #print("!! development: only using first few opp positions")
    #opp_start_states = opp_start_states[:1]
    
    try:
        with open(pickled_filename, 'rb') as f:
            race_results = pickle.load(f)

        print(f"Loaded race results from {pickled_filename}")
    except FileNotFoundError:
        race_results = None

    if race_results is None:
        docker_processes, sockets = connect_docker(name_tarfiles)

        ###############
        print("Starting parallel simulators...")
        pool = multiprocessing.Pool(len(name_tarfiles)) # one process per agent
        #pool = multiprocessing.Pool(os.cpu_count() // 2)

        pool_params = []

        for i, sock in enumerate(sockets):
            driver_name = name_tarfiles[i][0]
            tup = (racetrack, start_poses, sock, driver_name, opp_start_states)
            pool_params.append(tup)

        race_results = pool.map(worker_func, pool_params)

        print("Sending exit command to all docker subprocesses...")

        for sock in sockets:
            send_object(sock, {"type": "exit"})

        print(f"Saving pickled results to {pickled_filename}")
        
        with open(pickled_filename, 'wb') as f:
            pickle.dump(race_results, f)

        ############
        print("Waiting for docker subprocesses to exit...")

        for p in docker_processes:
            p.wait()

    print_results(race_results)

    print("replaying results..")
    replay_results(race_results, racetrack, start_poses, opp_start_states)

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

        secs = rr['runtime']

        if secs < 60:
            runtime = f"{round(secs, 1)}s"
        else:
            runtime = f"{round(secs / 60, 2)}m"

        # use neg avg_overtake time for better sorted results (descending)
        tup = num_overtakes, -overtake_frames, avg_overtake_time, num_timeouts, num_crashes, runtime, rr['name']
        sorted_race_results.append(tup)

    sorted_race_results.sort(reverse=True)

    for num_overtakes, _neg_frames, avg_overtake_time, num_timeouts, num_crashes, runtime, name in sorted_race_results:
        
        print(f"Driver: {name}, OV:{num_overtakes}, " + \
              f"OV_SEC:{avg_overtake_time}, C:{num_crashes}, TO:{num_timeouts}, Runtime: {runtime}")

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
