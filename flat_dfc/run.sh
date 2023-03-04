#!/bin/bash
# Create a Makefile with CMake, compile and run DFC

#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake  . ; make ; >> /dev/null 
#./dfc
#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DSINGLE_NUMA"  . ; make ; >> /dev/null 
#./dfc
#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DYIELD_WAIT"  . ; make ;  >> /dev/null 
#./dfc
#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DYIELD_COMBINER_DONE"  . ; make ; >> /dev/null 
#./dfc
make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DTHREAD_PIN"  . ; make ;  >> /dev/null 
./dfc
#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DYIELD_COMBINER_CPU"  . ; make ; >> /dev/null 
#./dfc
#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DSINGLE_NUMA -DYIELD_WAIT"  . ; make ; >> /dev/null 
#./dfc
#make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DSINGLE_NUMA -DYIELD_WAIT -DYIELD_COMBINER_DONE"  . ; make ; >> /dev/null 
#./dfc
make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DTHREAD_PIN -DSINGLE_NUMA -DYIELD_WAIT -DYIELD_COMBINER_DONE"  . ; make ; >> /dev/null 
./dfc
