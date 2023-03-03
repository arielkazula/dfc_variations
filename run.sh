#!/bin/bash
# Create a Makefile with CMake, compile and run DFC

make clean ; rm -rf CMakeCache.txt CMakeFiles/ Makefile dfc /dev/shm/dfc_shared /opt/ext4/HAGIT/dfc_shared /mnt/dfcpmem/dfc_shared ; cmake . ; make ; ./dfc
