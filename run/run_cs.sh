#!/usr/bin/env bash
 
compute-sanitizer --tool memcheck --leak-check full ./gpu-hashmap <<EOF
/home/pnglinkpc/cuda/gpu-hashmap-engine/test/testfile1.bin
/home/pnglinkpc/cuda/gpu-hashmap-engine/test/testfile2.bin
EOF
