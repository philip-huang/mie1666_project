#!/bin/bash

mkdir data
cd data
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
tar -xzf ALL_tsp.tar.gz

for f in *.gz ; do gunzip -c "$f" > "${f%.*}" ; done
rm -rf *.gz
