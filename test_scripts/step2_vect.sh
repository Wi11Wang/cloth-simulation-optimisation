python3 auto_test.py

advisor -collect=survey --project-dir ./step3_vect_omp -- ./auto_build/kernel_vect_omp -n 1000 -i 100 -d 4
advisor -collect=survey --project-dir ./step3_sse -- ./auto_build/kernel_sse -n 1000 -i 100 -d 4
