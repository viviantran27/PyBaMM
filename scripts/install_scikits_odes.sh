#!/bin/bash

SUNDIALS_URL=https://github.com/LLNL/sundials/archive/v5.0.0.tar.gz
SUNDIALS_NAME=sundials-5.0.0.tar.gz
CURRENT_DIR=`pwd`
TMP_DIR=$CURRENT_DIR/tmp
mkdir $TMP_DIR
INSTALL_DIR=$CURRENT_DIR/sundials

cd $TMP_DIR
wget $SUNDIALS_URL -O $SUNDIALS_NAME
tar -xvf $SUNDIALS_NAME
mkdir build-sundials-5.0.0
cd build-sundials-5.0.0/
cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_TYPE=int32_t -DBUILD_ARKODE:BOOL=OFF -DEXAMPLES_ENABLE:BOOL=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-5.0.0/
make install
cd $CURRENT_DIR
rm -rf $TMP_DIR
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export SUNDIALS_INST=$INSTALL_DIR
pip install scikits.odes

