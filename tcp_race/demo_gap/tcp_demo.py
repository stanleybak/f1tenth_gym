"""
Stanley Bak
TCP Ports client / server demo on Python

run with a single command line argument, "server" for server or two arguments, <server_port> <send_string> for client
"""

from typing import Optional

import socket
import sys
import struct
import json

def main():
    """main entry point"""

    assert len(sys.argv) in [2, 3], "expected single argument, server or post:<send-string>"

    if sys.argv[1] == "server":
        run_server()
    else:
        run_client(int(sys.argv[1]), sys.argv[2])

def run_server():
    """run server, print listen port"""

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        sock.bind(('', 0))

        port = sock.getsockname()[1]
        print(f"Server started, listening on port: {port}")

        sock.listen()

        while True:
            connection, client_address = sock.accept()

            print(f"got connection from {client_address}")
            #recv_bytes = connection.recv(8192)
            obj = recv_object(connection)

            if obj:
                print(f"server got: {obj}")
                reply = {'type': 'ack', 'msg': obj['msg']}
                
                send_object(connection, reply)

            connection.close()

            if obj['msg'] == "exit":
                print("server is exiting")
                break

def run_client(server_port, send_string):
    """run client, conntect to server on given port"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        server_address = ('localhost', server_port)

        print(f"Connecting to port {server_port}")

        sock.connect(server_address)

        send_obj = {'type': 'orig', 'msg': send_string}
        
        send_object(sock, send_obj)
        print(f"sent: {send_obj}")

        recv_obj = recv_object(sock)

        if recv_obj:
            print(f"client got: {recv_obj}")

    print("client is exiting")

def send_object(sock, obj):
    """send python object over tcp"""

    msg = json.dumps(obj).encode('utf-8')

    # add length in network byte order
    len_msg = struct.pack('>I', len(msg)) + msg
    
    sock.sendall(len_msg)

def recv_object(sock):
    """receive python object over tcp"""
    
    raw_bytes = recv_num_bytes(sock, 4)

    if not raw_bytes:
        rv = None
    else:
        num_bytes = struct.unpack('>I', raw_bytes)[0]
        rv_bytes = recv_num_bytes(sock, num_bytes)

        if rv_bytes is not None:
            rv = json.loads(rv_bytes.decode('utf-8'))

    return rv

def recv_num_bytes(sock, n) -> Optional[bytearray]:
    """recieve a specific number of bytes on a tcp socket"""
    
    rv: Optional[bytearray] = bytearray()
    assert rv is not None
    
    while len(rv) < n:
        packet = sock.recv(n - len(rv))
        
        if not packet:
            rv = None
            break

        rv.extend(packet)
        
    return rv

if __name__ == "__main__":
    main()
