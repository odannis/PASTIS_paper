#!/bin/bash

# Loop from 0 to 100
for i in {0..1000}; do
    echo "Running: python3.11 large_deviation.py $i"
    python3.11 large_deviation.py "$i"
done
