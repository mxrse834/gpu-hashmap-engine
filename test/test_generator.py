#!/usr/bin/env python3

import random
import struct
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM = random.Random(0xDEADBEEF)


def build_reference_payload() -> bytes:
    payload = bytearray()
    payload.extend(struct.pack("<I", 0xCAFEBABE) * 64)
    payload.extend(bytes(range(256)))
    payload.extend(
        b"".join(struct.pack("<I", RANDOM.randint(0, 0xFFFFFFFF)) for _ in range(128))
    )
    payload.extend(b"\x00" * 256)
    payload.extend(b"\xFF" * 256)
    payload.extend(b"\xAB" * 256)
    payload.extend(b"STRESS_TEST_ODD_TAIL_XYZ")
    payload.extend(b"\x00" * 127 + b"\x01")
    payload.extend(bytes((0xAA if i % 2 == 0 else 0x55) for i in range(256)))
    payload.extend(b"P" * 17)
    return bytes(payload)


def write(name: str, data: bytes) -> None:
    (OUTPUT_DIR / name).write_bytes(data)


reference = build_reference_payload()
write("file_a.bin", reference)
write("file_b.bin", reference)

one_byte_changed = bytearray(reference)
one_byte_changed[512] ^= 0xFF
write("file_c.bin", one_byte_changed)

write("file_d.bin", reference[:-17])

reordered = bytearray(reference)
first = bytes(reordered[0:16])
later = bytes(reordered[320:336])
reordered[0:16] = later
reordered[320:336] = first
write("file_reordered.bin", reordered)

write("empty_a.bin", b"")
write("empty_b.bin", b"")

print(f"generated deterministic fixtures in {OUTPUT_DIR}")
