#!/usr/bin/bash

module load CMake/3.16.4-GCCcore-9.3.0
module load tbb/2020.1-GCCcore-9.3.0
module load OpenMPI/4.0.3-GCC-9.3.0

mkdir ../build
cd ../build
cmake ..
make

