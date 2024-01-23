#!/bin/bash
#PBS -P c07
#PBS -q normal
#PBS -l walltime=02:30:00
#PBS -l mem=16GB
#PBS -l jobfs=4GB
#PBS -l ncpus=1
#PBS -l wd

module load papi
module load gcc/12.2.0
module load cmake/3.18.2
module load python3/3.8.5
module load python3-as-python
module load intel-compiler

python3 auto_test.py main
python3 auto_profile.py -p main -s 2 -n 1 -f gadi
