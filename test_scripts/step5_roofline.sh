#!/bin/bash
#PBS -P c07
#PBS -q normal
#PBS -l walltime=00:15:00
#PBS -l mem=190GB
#PBS -l jobfs=4GB
#PBS -l ncpus=48
#PBS -l wd

module load papi
module load gcc/12.2.0
module load cmake/3.18.2
module load python3/3.8.5
module load python3-as-python
module load intel-compiler
module load intel-advisor

echo "last test: save in roofline"
python3 auto_test.py omp
advisor --collect survey --project-dir ./task5_roofline -- ./auto_build/kernel_omp -n 2000 -i 50 -d 8 -s 1.0 -m 1.0 -f 10.0 -g 0.981 -b 3.0 -o 0.0 -t 0.05 -p 48
advisor --collect tripcounts -flop -enable-cache-simulation --project-dir ./task5_roofline -- ./auto_build/kernel_omp -n 2000 -i 50 -d 8 -s 1.0 -m 1.0 -f 10.0 -g 0.981 -b 3.0 -o 0.0 -t 0.05 -p 48
