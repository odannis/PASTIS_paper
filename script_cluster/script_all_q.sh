#$ -cwd
#$ -j y
#$ -M andonis.gerardos@univ-amu.fr

module load oneapi_2021.2.0
module load openmpi-x86_64
export NUMBA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

num_cpu=$(grep -c ^processor /proc/cpuinfo)

echo nump_cpu $num_cpu
# Accessing the job number
echo "The job number is: ${JOB_ID}"
echo ${2}
conda activate myclone
echo "Current directory: $(pwd)"

python3.10 ${2} ${1}

## Start command : qsub -N job_name script.sh name_file.ipynb
