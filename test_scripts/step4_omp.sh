#!/bin/bash
#PBS -P c07
#PBS -q normal
#PBS -l walltime=02:00:00
#PBS -l mem=16GB
#PBS -l jobfs=8GB
#PBS -l ncpus=24
#PBS -l wd

module load papi
module load gcc/12.2.0
module load intel-compiler
module load cmake/3.18.2
module load python3/3.8.5
module load python3-as-python

python3 auto_test.py omp

echo "-------------------------"
echo "dynamic"
echo "-------------------------"
sed -i '23 c\#define SCHEDULE_TYPE dynamic' cloth_code_omp.cpp
python3 auto_profile.py -p omp -s 4 -n 1 -f gadi_dynamic

echo "-------------------------"
echo "static, 1"
echo "-------------------------"
sed -i '23 c\#define SCHEDULE_TYPE static, 1' cloth_code_omp.cpp
python3 auto_profile.py -p omp -s 4 -n 1 -f gadi_static_1

echo "-------------------------"
echo "static, 4"
echo "-------------------------"
sed -i '23 c\#define SCHEDULE_TYPE static, 4' cloth_code_omp.cpp
python3 auto_profile.py -p omp -s 4 -n 1 -f gadi_static_4

echo "-------------------------"
echo "static, 16"
echo "-------------------------"
sed -i '23 c\#define SCHEDULE_TYPE static, 16' cloth_code_omp.cpp
python3 auto_profile.py -p omp -s 4 -n 1 -f gadi_static_16

echo "-------------------------"
echo "static, 64"
echo "-------------------------"
sed -i '23 c\#define SCHEDULE_TYPE static, 64' cloth_code_omp.cpp
python3 auto_profile.py -p omp -s 4 -n 1 -f gadi_static_64
sed -i '23 c\#define SCHEDULE_TYPE dynamic' cloth_code_omp.cpp

echo "-------------------------"
echo "static, n / p"
echo "-------------------------"
sed -i '23 c\#define SCHEDULE_TYPE static, n / omp_get_max_threads()' cloth_code_omp.cpp
python3 auto_profile.py -p omp -s 4 -n 1 -f gadi_static_np

echo "PROFILE FINISHED"
