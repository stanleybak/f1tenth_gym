"""
Stanley Bak
networking utilities
"""

from typing import Optional, Any

import json
import struct

def send_object(sock, obj):
    """send python object over tcp"""

    msg = json.dumps(obj).encode('utf-8')

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
