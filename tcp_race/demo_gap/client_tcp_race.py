"""
eval a specific scenario for overtaking
"""

import sys
import socket

import numpy as np

from networking import send_object, recv_object
from gap_driver import GapFollower

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

def start_client(server_port, driver_index, gain):
    """start a client with gap follower with the given gain"""

    driver = GapFollower(gain)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        server_address = ('localhost', server_port)

        print(f"Connecting to port {server_port}...", flush=True)
        sock.connect(server_address)

        init_obj = {'type': 'init', 'driver_index': driver_index}
        
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
    gain = float(sys.argv[3])

    start_client(port, driver_index, gain)

if __name__ == "__main__":
    main()
