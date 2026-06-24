"""Length-prefixed pickle protocol for the Day 6 ACT bridge.

The CALVIN side runs in the eval environment while the LeRobot policy worker
runs in the training environment.  The protocol is intentionally small: every
message is a pickle payload framed by a four-byte big-endian length.
"""

from __future__ import annotations

import pickle
import struct
from typing import Any, BinaryIO


HEADER_SIZE = 4
PICKLE_PROTOCOL = 4
MAX_FRAME_BYTES = 64 * 1024 * 1024


class BridgeProtocolError(RuntimeError):
    """Raised when bridge framing or message content is invalid."""


def write_frame(stream: BinaryIO, payload: dict[str, Any]) -> None:
    """Write one framed protocol message and flush the stream."""

    raw = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
    if len(raw) > MAX_FRAME_BYTES:
        raise BridgeProtocolError(f"bridge frame is too large: {len(raw)} bytes")
    stream.write(struct.pack(">I", len(raw)))
    stream.write(raw)
    stream.flush()


def read_frame(stream: BinaryIO) -> dict[str, Any]:
    """Read one framed protocol message."""

    header = _read_exact(stream, HEADER_SIZE)
    size = struct.unpack(">I", header)[0]
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise BridgeProtocolError(f"invalid bridge frame size: {size}")
    raw = _read_exact(stream, size)
    payload = pickle.loads(raw)
    if not isinstance(payload, dict):
        raise BridgeProtocolError(f"bridge payload must be a dict, got {type(payload).__name__}")
    return payload


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("bridge stream closed while reading a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
