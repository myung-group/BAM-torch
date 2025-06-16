## This is a version under development. 


## Load modules
```
$ module load intel-22.3.1/icc-22.3.1 intel-22.3.1/fftw-3.3.10 cuda/cuda-11.8
    
    or simply (If you intend to use and install it on the Frodo server)
    $ module load intel-22.3.1/lammps-2Aug2023
```


## Installation 
Install libtorch
```
$ wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.2.0%2Bcu121.zip
$ unzip libtorch-shared-with-deps-2.2.0+cu121.zip
$ rm libtorch-shared-with-deps-2.2.0+cu121.zip
$ mv libtorch libtorch-gpu
```

Install LAMMPS modified for BAM Package
```
$ git clone https://github.com/myung-group/lammps.git
$ cd lammps
$ mkdir build
$ cd build
$ cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$(pwd) \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D BUILD_MPI=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_AMDAVX=ON \
    -D Kokkos_ARCH_AMPERE100=ON \
    -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch-gpu \
    -D PKG_ML-BAM=ON \
    ../cmake
$ make -j N     (N: integer)
    ex) $ make -j 8
$ make install
```

Then, you will see a sentence like the one below.
```
[ 85%] Building CXX object CMakeFiles/lammps.dir/home/gbsim/prog/lammps_bam/lammps/src/ML-BAM/pair_bam.cpp.o
```

You can check like below,
```
$ lmp -h | grep bam

>>> bam             born            buck            buck/coul/cut   buck/coul/cut/kk 
```
