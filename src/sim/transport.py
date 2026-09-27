"""Length-prefixed pickle messages between trusted localhost processes."""

from __future__ import annotations

import pickle
import socket
import struct

__all__ = ["recv_msg", "send_msg"]


def recv_msg(conn) -> dict | None:
    """-> the next message, or None when the peer has gone away."""
    hdr = conn.recv(8, socket.MSG_WAITALL)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack("<Q", hdr)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def send_msg(conn, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(data)) + data)
