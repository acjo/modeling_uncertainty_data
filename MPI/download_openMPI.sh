#!/bin/bash
#download openMPI
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
# extract the files
tar -zxf openmpi-4.0.2.tar.gz
cd openmpi-4.0.2
# configure the files
./configure --prefix=/usr/local/openmpi
# compile openMPI
make all
# install openMPI
sudo make install
