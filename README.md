# dfc_variations

In this repository you can find code to run different implementations of the DFC stack algorithm described in the 
paper Flat-Combining-Based Persistent Data Structures for Non-Volatile Memory: https://arxiv.org/abs/2012.12868.
All the code here is based on Matan Rusanovsky's implementation of the DFC Stack which can be found here:
https://github.com/matanr/detectable_flat_combining

We added support for controlling the configuration of the system, such as the number of NUMA nodes, enabling CPU pinning and more.
The bench mark which is running and the theoretical background is a as described in our report:

For full details about the theoretical model and the algorithms implemented in this repository please read the project report.
In the report you will find all the needed prerequisites, along with a full breakdown of each configuration, design, experiments results and more.

# Installation and pre-run: 
```bash
git clone https://github.com/arielkazula/dfc_variations.git  
cd dfc_variants/  
module purge  
module load pmdk1.9  
module load libpmemobj++/1.10  
export PMEM_IS_PMEM_FORCE=1  
module load cmake/3.15.3
module load gnu/9.1.0-no-cuda-offloading
```

# Run benchmark on baseline DFC stack code with different system configuration:
Follow these steps:
 ```bash
cd ./baseline
make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="<enter here compile flags>" ; make ; >> /dev/null
./dfc
```

Compilation Flags:
These are the parameters you must set for each different run:

  * SINGLE_NUMA - to run the benchmark on a single NUMA node.
  * YIELD_WAIT - to run the benchmark where threads avoid busy waiting for the combiner.
  * YIELD_COMBINER_DONE - to run the benchmark where the combiner executes the yield() system call at the end of the combine phase.
  * THREAD_PIN - to run the benchmark when the threads are pinned to a CPU in a round-robin manner.
  
**Examples**
  
 1. Running the benchmark on the baseline DFC stack default configuration (no yield, 2 NUMA nodes, no CPU pinning):
```bash
make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake . ; make ; >> /dev/null
./dfc
```

 The output received in the file ./data/dfc.txt:
 |Threads| OP/S|
 |-  |-        |
 |  1|   626605|
 | 16|  2155737|
 | 24|  2522955|
 | 36|  2977331|
 | 48|  2912480|
 | 60|  2671619|
 | 72|  2287944|
 | 84|  2036364|
 | 96|  1708641|
  
  The results represent the median of the number of operations per second.
  
  2. Running the benchmark on the baseline DFC stack without busy waiting and on a single NUMA node:
```bash
make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="-DYIELD_WAIT -DSINGLE_NUMA" ; make ; >> /dev/null
./dfc
```
  
  The output received in the file ./data/dfc_noBusyWait_1numa.txt:
  |  Threads| OP/S|
|-|-|
|  1    |   1018814|
|  16   |   3682424|
|  24   |   3712921|
|  36   |   3902502|
|  48   |   4022289|
|  60   |   4083412|
|  72   |   4051604|
|  84   |   4107869|
|  96   |   4182981|
  
  The results represent the median of the number of operations per second.
  
  To run various configuration automatically simply run:
```bash
./run.sh
```
  # Run the benchmark on Flat-Obj DFC stack code with different system configuration:
  
  Follow these steps:
  ```bash
cd ./baseline
make clean ; rm -rf CMakeCache.txt /dev/shm/* CMakeFiles/ Makefile dfc  ; cmake -DCMAKE_CXX_FLAGS="<enter here compile flags>" ; make ; >> /dev/null
./dfc
    ```
  
  The compilation flags are the same compilation flags as for the baseline case, as does the run.sh script.
  
Contributing
Pull requests are welcomed.

License
All rights are reserved to Ariel Kazula, Saar Bar-Oz and Menucha Benisti. The stack implementation is basted on the work of Matan Rusanovsky.
