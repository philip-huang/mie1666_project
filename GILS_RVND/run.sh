#!/bin/bash

echo "--MLP Benchmark--"

make

k=1
for instance in ../data/test-optimal/S1/*.npz; do
	echo "Running $instance"
	echo "Instance $k"

	./mlp ${instance}

	k=$(($k + 1))
done


k=1
for instance in ../data/test-optimal/S2/*.npz; do
	echo "Running $instance"
	echo "Instance $k"

	./mlp ${instance}

	k=$(($k + 1))
done
