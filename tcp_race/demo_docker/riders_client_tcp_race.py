"""
riders client code in docker
"""

import sys
import socket
import os

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

def start_client(server_port, driver_index, name):
    """start a client with gap follower with the given gain"""

    ####### import driver by looking in f1tenth_ros_agent.py
    current_dir = os.path.abspath(os.path.dirname(__file__))
        
    src_path = os.path.join(current_dir, "files", "pkg", "src")
    sys.path.append(src_path)

    path = os.path.join(current_dir, "files", "pkg", "nodes", "f1tenth_ros_agent.py")

    with open(path, "r") as f:
        lines = f.readlines()

    import_line = None
        
    for line in lines:
        line = line.strip()
        
        if line.startswith('from pkg.') and line.endswith('as Driver'):
            assert import_line is None, f"multiple import lines: '{import_line}' and '{line}'"
            import_line = line

    assert import_line is not None

    print(f"running Driver import line: {import_line}")
    exec(import_line, globals())

    #####################

    driver = Driver()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        server_address = ('localhost', server_port)

        print(f"Connecting to port {server_port}...", flush=True)
        sock.connect(server_address)

        init_obj = {'type': 'init', 'driver_index': driver_index, 'name': name}
        
        send_object(sock, init_obj)

        while True:
            # receive state

            obj = recv_object(sock)

            if obj is None or obj['type'] == 'exit':
                break

            assert obj['type'] == 'obs'
            odom = obj['odom']

            scan_list = obj['scan']
            
            scan = np.array(scan_list, dtype=float)
            
            if hasattr(driver, 'process_observation'):
                speed, steer = driver.process_observation(ranges=scan, ego_odom=odom)
            else:
                assert hasattr(driver, 'process_lidar')
                speed, steer = driver.process_lidar(scan)

            actions_obj = {'type': 'actions', 'speed': speed, 'steer': steer}
            send_object(sock, actions_obj)

def main():
    """main entry point"""

    assert len(sys.argv) == 4, 'Usage: <server_port> <driver_index> <gain>'
    
    port = int(sys.argv[1])
    driver_index = int(sys.argv[2])
    name = sys.argv[3]

    start_client(port, driver_index, name)

if __name__ == "__main__":
    main()
