#!/usr/bin/bash

module load CMake/3.18.4-GCCcore-10.2.0
module load intel-tbb-oss/intel64/2020.3
module load apps
module load site/infer/easybuild/setup
module load OpenMPI/4.0.5-GCC-10.2.0

mkdir ../build
cd ../build
cmake ..
make

