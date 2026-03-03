#!/usr/bin/env bash

	compute-sanitizer --tool memcheck ./gpu-hashmap <<EOF
/home/pnglinkpc/cuda/gpu-hashmap-engine/testfile1.bin
/home/pnglinkpc/cuda/gpu-hashmap-engine/testfile2.bin
EOF


