#!/bin/bash

set -x

#  NOTE: The following is required when running from Gitlab CI via slurm job
source /etc/profile.d/modules.sh

export MY_CLUSTER=`uname -n | sed -e 's/[0-9]//g' -e 's/\..*//'`
export PROJ_DIR=/qfs/projects/exasgd
export APPS_DIR=/share/apps

module purge
if [ "$MY_CLUSTER" == "newell" ]; then
    export MY_GCC_VERSION=7.4.0
    export MY_CUDA_VERSION=10.2
    export MY_OPENMPI_VERSION=3.1.5
    export MY_CMAKE_VERSION=3.16.4
    export MY_MAGMA_VERSION=2.5.2_cuda10.2
    export MY_METIS_VERSION=5.1.0
    export MY_NVCC_ARCH="sm_70"
else
    #  NOTE: The following is required when running from Gitlab CI via slurm job
    export MY_CLUSTER="marianas"
    export MY_GCC_VERSION=7.3.0
    export MY_CUDA_VERSION=10.2.89
    export MY_OPENMPI_VERSION=3.1.3
    export MY_CMAKE_VERSION=3.15.3
    export MY_MAGMA_VERSION=2.5.2_cuda10.2
    export MY_METIS_VERSION=5.1.0
    export MY_NVCC_ARCH="sm_60"
fi

export NVBLAS_CONFIG_FILE=$PROJ_DIR/$MY_CLUSTER/nvblas.conf
module load gcc/$MY_GCC_VERSION
module load cuda/$MY_CUDA_VERSION
module load openmpi/$MY_OPENMPI_VERSION
module load cmake/$MY_CMAKE_VERSION
module load magma/$MY_MAGMA_VERSION

export MY_RAJA_DIR=$PROJ_DIR/$MY_CLUSTER/raja
export MY_UMPIRE_DIR=$PROJ_DIR/$MY_CLUSTER/umpire
export MY_UMFPACK_DIR=$PROJ_DIR/$MY_CLUSTER/suitesparse

# 5.1.0 is currently the only version available on both newell and marianas,
# but leave this line alone in case future versions are made available on one
# mahcine but not the other
export MY_METIS_DIR=$APPS_DIR/metis/$MY_METIS_VERSION

base_path=`dirname $0`
#  NOTE: The following is required when running from Gitlab CI via slurm job
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    cd $base_path          || exit 1
fi

export CMAKE_OPTIONS="\
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTS=ON \
    -DHIOP_USE_MPI=On \
    -DHIOP_DEEPCHECKS=ON \
    -DRAJA_DIR=$MY_RAJA_DIR \
    -DHIOP_USE_RAJA=On \
    -Dumpire_DIR=$MY_UMPIRE_DIR \
    -DHIOP_USE_UMPIRE=On \
    -DHIOP_WITH_KRON_REDUCTION=ON \
    -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR \
    -DHIOP_METIS_DIR=$MY_METIS_DIR \
    -DHIOP_USE_GPU=ON \
    -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR \
    -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH"

BUILDDIR="build"
rm -rf $BUILDDIR                            || exit 1
mkdir -p $BUILDDIR                          || exit 1
cd $BUILDDIR                                || exit 1
cmake $CMAKE_OPTIONS ..                     || exit 1
cmake --build .                             || exit 1
ctest                                       || cat Testing/Temporary/LastTest.log
exit 0
