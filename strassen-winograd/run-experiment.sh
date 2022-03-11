#!/bin/bash

for p in $(seq 1 8); do  
  ./prog OMP_NUM_THREADS=$p >> results_strassen.txt 
done

