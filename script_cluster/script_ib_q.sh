#$ -cwd
#$ -j y
#$ -M andonis.gerardos@univ-amu.fr

# module load oneapi_2021.2.0
# module load openmpi-x86_64
nump_cpu_use=${3}

num_cpu_allowed=$((2*$nump_cpu_use))
export NUMBA_NUM_THREADS=$num_cpu_allowed
export OPENBLAS_NUM_THREADS=$num_cpu_allowed
export MKL_NUM_THREADS=$num_cpu_allowed
num_cpu=$(grep -c ^processor /proc/cpuinfo)
num_cpu=$(($num_cpu/$nump_cpu_use))
hostname_=$(hostname)

echo hostname $hostname_
echo nump_cpu_use $nump_cpu_use
echo num_cpu_allowed $num_cpu_allowed
echo $num_cpu
echo ${2}
# Accessing the job number
echo "Current directory: $(pwd)"
#conda activate myclone
conda activate myenv
ulimit -c 0

for ((i = 0 ; i < $num_cpu ; i++)); do
    b=$(($num_cpu * ${1} + $i))
    python3.10 ${2} ${b} &
done

(
  maxMem=0
  maxCPU=0
  while pgrep python3.10 > /dev/null; do
    mem=$(free -m | awk 'FNR == 2 {printf("%.2f", $3/$2*100)}')
    cpu=$(top -bn1 | grep "Cpu(s)" | sed 's/.*, *\([0-9.]*\)%* id.*/\1/' | awk '{print 100 - $1}')
    #echo "RAM Usage: ${mem}%"
    if (( $(echo "$mem > $maxMem" | bc -l) )); then
      maxMem="$mem"
    fi
    if (( $(echo "$cpu > $maxCPU" | bc -l) )); then
      maxCPU="$cpu"
    fi
    echo "$maxMem" > max_mem_usage_${JOB_ID}.txt
    echo "$maxCPU" > max_cpu_usage_${JOB_ID}.txt
    sleep 5
  done
) &


wait

## Start command : qsub -N job_name script.sh name_file.ipynb
