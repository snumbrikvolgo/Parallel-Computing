#!/bin/bash

OUT="lab"

mpic++ -o $OUT $OUT.cpp

while [ ! -e $OUT ]; do
    sleep 1;
done

mpiexec -n 4 ./$OUT

rm $OUT
