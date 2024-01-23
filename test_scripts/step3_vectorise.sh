#!/bin/bash
#PBS -P c07
#PBS -q normal
#PBS -l walltime=00:30:00
#PBS -l mem=16GB
#PBS -l jobfs=4GB
#PBS -l ncpus=2
#PBS -l wd

module load papi
module load gcc/12.2.0
module load cmake/3.18.2
module load python3/3.8.5
module load python3-as-python
module load intel-compiler
module load intel-advisor

python3 auto_test.py
advisor -collect=survey --project-dir ./task3_vect_omp -- ./auto_build/kernel_vect_omp -n 1000 -i 50 -d 8 -s 1.0 -m 1.0 -f 10.0 -g 0.981 -b 3.0 -o 0.0 -t 0.05
advisor -collect=survey --project-dir ./task3_sse -- ./auto_build/kernel_sse -n 1000 -i 50 -d 8 -s 1.0 -m 1.0 -f 10.0 -g 0.981 -b 3.0 -o 0.0 -t 0.05
