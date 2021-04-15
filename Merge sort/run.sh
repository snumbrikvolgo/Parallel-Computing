set -e
echo $'N Proc Time\n' > result.txt
for run in $(seq 1 1 10); do
for proc in $(seq 2 1 4); do
for num in $(seq 10000000 10000000 40000000); do
    mpiexec -n $proc ./a.out $num out_$num.txt
done
done
done


