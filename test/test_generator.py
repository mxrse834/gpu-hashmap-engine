import random
import struct
import shutil

random.seed(0xDEADBEEF)

# generate file_a
with open("file_a.bin", "wb") as f:
    for i in range(64):
        f.write(struct.pack("<I", 0xCAFEBABE))
    for i in range(256):
        f.write(struct.pack("B", i % 256))
    for i in range(128):
        f.write(struct.pack("<I", random.randint(0, 0xFFFFFFFF)))
    f.write(b'\x00' * 256)
    f.write(b'\xFF' * 256)
    f.write(b'\xAB' * 256)
    f.write(b'STRESS_TEST_ODD_TAIL_XYZ')
    f.write(b'\x00' * 127 + b'\x01')
    for i in range(256):
        f.write(b'\xAA' if i % 2 == 0 else b'\x55')
    f.write(b'P' * 17)

# identical copy
shutil.copy("file_a.bin", "file_b.bin")

# one byte flipped
shutil.copy("file_a.bin", "file_c.bin")
with open("file_c.bin", "r+b") as f:
    f.seek(512)
    f.write(b'\x00')

# same start different end
with open("file_d.bin", "wb") as f:
    for i in range(64):
        f.write(struct.pack("<I", 0xCAFEBABE))
    f.write(b'\xDE\xAD\xBE\xEF' * 64)
    f.write(b'\x00' * 256)
    f.write(b'\xFF' * 256)
    f.write(b'\xAB' * 256)
    f.write(b'STRESS_TEST_ODD_TAIL_XYZ')
    f.write(b'\x00' * 127 + b'\x01')
    for i in range(256):
        f.write(b'\xAA' if i % 2 == 0 else b'\x55')
    f.write(b'P' * 17)

print("generated file_a.bin, file_b.bin, file_c.bin, file_d.bin")
