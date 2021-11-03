"""
Stanley Bak
networking utilities
"""

from typing import Optional, Any

import struct
import json
import time
import pickle

def send_object(sock, obj):
    """send python object over tcp"""

    #start = time.perf_counter()
    msg = json.dumps(obj).encode('utf-8')
    #diff = time.perf_counter() - start
    #print(f"json dumps: {round(diff * 1000,2)}ms")

    #start = time.perf_counter()
    #msg = pickle.dumps(obj).encode()
    #diff = time.perf_counter() - start
    #print("pickle: {round(diff * 1000,2)},s")

    # add length in network byte order
    len_msg = struct.pack('>I', len(msg)) + msg
    
    sock.sendall(len_msg)

def recv_object(sock) -> Optional[Any]:
    """receive python object over tcp"""
    
    raw_bytes = recv_num_bytes(sock, 4)

    if not raw_bytes:
        rv = None
    else:
        num_bytes = struct.unpack('>I', raw_bytes)[0]
        rv_bytes = recv_num_bytes(sock, num_bytes)

        if rv_bytes is not None:
            #start = time.perf_counter()
            rv = json.loads(rv_bytes.decode('utf-8'))
            #diff = time.perf_counter() - start
            #print(f"json loads: {round(diff * 1000,2)}ms")

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
