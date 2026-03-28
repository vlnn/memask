import os
import time

ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    timestamp_ms = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10))

    chars = []
    for _ in range(16):
        chars.append(ENCODING[randomness & 31])
        randomness >>= 5

    for _ in range(10):
        chars.append(ENCODING[timestamp_ms & 31])
        timestamp_ms >>= 5

    return "".join(reversed(chars))
