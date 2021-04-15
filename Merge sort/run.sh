set -e
echo $'N Proc Time\n' > result.txt
for run in $(seq 1 1 10); do
for proc in $(seq 1 1 4); do
for num in $(seq 1000 1000 100000); do
    mpiexec -n $proc ./a.out $num in_$num$proc.txt out_$num$proc.txt
done
done
done


