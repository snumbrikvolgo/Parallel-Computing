#!/bin/bash

for ((i = 50; i <= 500; i += 10))
do
./matrix $i > out.txt
cat out.txt >> matrix.csv
done

for ((i = 550; i <= 1000; i += 50))
do
./matrix $i > out.txt
cat out.txt >> matrix.csv
done