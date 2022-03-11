#!/bin/bash

for p in $(seq 1 8); do
  for n in $(seq 22 1 22); do
    for run in $(seq 1 5); do
      ./fft $((2**$n)) $p &>> data.txt
    done
  done
done

