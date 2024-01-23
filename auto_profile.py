import subprocess
import os
import time
import re
import numpy as np
import csv
from datetime import datetime
import argparse

# Test configurations
node_list = [20, 400, 1000, 2000]
delta_list = [2, 4, 8]
thread_list = [1, 4, 8, 24, 48]
output_type = np.longlong
cur_formatter = "{:,.0f}"
sci_formatter = "{:.7e}"


def run_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.stdout, result.stderr


def get_param(n, d, p=0):
    s = 1.0
    m = 1.0
    f = 10.0
    g = 0.981
    b = 3.0
    o = 0.0
    t = 0.05
    i = 100
    if p == 0:
        return f"-n {int(n)} -s {s} -m {m} -f {f} -d {int(d)} -g {g} -b {b} -o {o} -t {t} -i {i}"
    else:
        return f"-n {int(n)} -s {s} -m {m} -f {f} -d {int(d)} -g {g} -b {b} -o {o} -t {t} -i {i} -p {p}"


def parse_result(program, result):
    split_lines = '\n'.join(line for line in result.split('\n') if line.lstrip().startswith(program))
    result_dict = {}
    for line in split_lines.split('\n'):
        match = re.search(rf'{program}: (.+?):\s+(.+)', line)
        if match:
            info_name = match.group(1)
            info_value = match.group(2)
            result_dict[info_name] = np.longlong(info_value)
    return result_dict


def run_n_tests(program, n_tests=3):
    result_list = {}
    for n in node_list:
        for d in delta_list:
            if program == 'omp':
                for p in thread_list:
                    parameters = get_param(n, d, p)
                    command = f"./auto_build/kernel_{program} {parameters}"
                    for _ in range(n_tests):
                        stdout, _ = run_command(command)
                        result = parse_result(program, stdout.decode())
                        print(f"n={n}, d={d}, p={p}: {result}")
                        if (n, d, p) not in result_list:
                            result_list[(n, d, p)] = []
                        result_list[(n, d, p)].append(result)
            else:
                parameters = get_param(n, d)
                command = f"./auto_build/kernel_{program} {parameters}"
                for _ in range(n_tests):
                    stdout, _ = run_command(command)
                    result = parse_result(program, stdout.decode())
                    print(f"n={n}, d={d}: {result}")
                    if (n, d) not in result_list:
                        result_list[(n, d)] = []
                    result_list[(n, d)].append(result)
    return result_list


def get_avg_result(result_list):
    avg_result = {}
    for setting, values in result_list.items():
        avg_wall_time = np.mean([value['wall time (us)'] for value in values]).astype(output_type)
        avg_DP_OPS = np.mean([value.get('PAPI_DP_OPS', 0) for value in values]).astype(output_type)
        avg_MFLOPS = np.mean([value.get('MFLOPS', 0) for value in values]).astype(output_type)
        avg_L1_DCM = np.mean([value.get('PAPI_L1_DCM', 0) for value in values]).astype(output_type)
        avg_L2_DCM = np.mean([value.get('PAPI_L2_DCM', 0) for value in values]).astype(output_type)
        avg_TOT_INS = np.mean([value.get('PAPI_TOT_INS', 0) for value in values]).astype(output_type)
        avg_BR_MSP = np.mean([value.get('PAPI_BR_MSP', 0) for value in values]).astype(output_type)
        avg_VEC_DP = np.mean([value.get('PAPI_VEC_DP', 0) for value in values]).astype(output_type)
        avg_result[setting] = [avg_wall_time, avg_DP_OPS, avg_MFLOPS, avg_L1_DCM, avg_L2_DCM, avg_TOT_INS, avg_BR_MSP, avg_VEC_DP]
    return avg_result


def align_two_files(file1_path, file2_path):
    def get_max_widths_from_file(file_obj):
        """Compute maximum widths of columns in a given file"""
        max_widths = []
        reader = csv.reader(file_obj, delimiter=';')
        for row in reader:
            if not max_widths:
                max_widths = [0] * len(row)
            for i, item in enumerate(row):
                max_widths[i] = max(max_widths[i], len(item))
        return max_widths

    def format_file_with_widths(file_obj, max_widths):
        """Format the contents of a file based on specified column widths"""
        reader = csv.reader(file_obj, delimiter=';')
        formatted_rows = [' | '.join(item.ljust(max_widths[i]) for i, item in enumerate(row)) for row in reader]
        return '\n'.join(formatted_rows)

    # First, compute the maximum widths for both files
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        max_widths_1 = get_max_widths_from_file(file1)
        max_widths_2 = get_max_widths_from_file(file2)
        max_widths_2 += [1] * (len(max_widths_1) - len(max_widths_2))
        overall_max_widths = [max(w1, w2) for w1, w2 in zip(max_widths_1, max_widths_2)]

    # Now, format both files using the overall maximum widths
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        formatted_file1 = format_file_with_widths(file1, overall_max_widths)
        formatted_file2 = format_file_with_widths(file2, overall_max_widths)

    # Combine the formatted contents of both files, separated by 20 dots
    return ((formatted_file1 + '\n'
             + '.' * (sum(overall_max_widths) + len(overall_max_widths) * 3 - 2)
             + '\n' + formatted_file2)
            + '\n' + '~' * (sum(overall_max_widths) + len(overall_max_widths) * 3 - 2))


def write_to_csv(program, step, avg_result, time):
    csv_formatter = cur_formatter
    save_path0 = f"./report/step{step}/{program}_{time}_0.csv"
    save_path1 = f"./report/step{step}/{program}_{time}_1.csv"
    # Open CSV file for writing
    with open(save_path0, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=';')

        # Iterate through dictionary and write each item
        if program == 'omp':
            for (n, delta, nthreads), values in avg_result.items():
                writer.writerow([int(n), cur_formatter.format(int(delta)), nthreads] + [csv_formatter.format(v) for v in values[:3]])
        else:
            for (n, delta), values in avg_result.items():
                writer.writerow([int(n), cur_formatter.format(int(delta))] + [csv_formatter.format(v) for v in values[:3]])

    with open(save_path1, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=';')

        # Iterate through dictionary and write each item
        for _, values in avg_result.items():
            writer.writerow([csv_formatter.format(v) for v in values[3:]])

    formatted_str = align_two_files(save_path0, save_path1)
    print(f'\n--- Generated CSV for {program} ---')
    print(formatted_str)
    print("." * 30)
    if program == 'omp':
        print('n | d | p | us | dps | flops')
    else:
        print('n  | d  | us  | dps | flops')
    print("." * 30)
    print("l1 | l2  | ins | msp | vec")
    print("-" * 40)
    return [save_path0, save_path1]


def cleanup(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
        else:
            print(f'{path} not exist')
    print('~' * 30)
    print('Cleanup Finished!')
    print('~' * 30)


def compile_kernels():
    print("=== Compiling ===")
    if os.path.exists("auto_build/CMakeCache.txt"):
        make_out, make_err = run_command(
            "mkdir -p auto_build && cd auto_build && make -j 24"
        )
    else:
        make_out, make_err = run_command(
            "mkdir -p auto_build && cd auto_build && rm -r * && cmake .. && make -j 24"
        )

    err_string = f"STDOUT:\n{make_out.decode()}\nSTDERR:\n{make_err.decode()}"
    if "error" in str(make_err) or "CMake Error" in str(make_err):
        print("Error while running 'make':")
        print(err_string)
    else:
        print("Compile without error")


def main(args):
    start_time = time.time()

    programs = ["main", "opt", "sse", "vect_omp", "omp"]

    compile_kernels()

    cleanup_kernels = args.s == 1
    if args.p in programs:
        programs = [args.p]
    step = args.s
    n_tests = args.n
    flag = args.f
    print("=== Profile Settings ===")
    print(f"Nodes: {node_list}")
    print(f"Deltas: {delta_list}")
    print(f"Threads: {thread_list}")

    if cleanup_kernels:
        print(f"Result will NOT be saved")
    else:
        print(f"Save result at ./report/step{step}", end='')
        if len(flag) > 0:
            print(f" with {flag}", end='')
    print("\n" + "=" * 21)

    for program in programs:
        print("\n" + "-" * 40)
        print(f"Profiling kernel_{program}")
        print("-" * 40)

        filename_time = datetime.now().strftime("%d_%H_%M_%S") + f"_{flag}"
        result_list = run_n_tests(program, n_tests)
        avg_result = get_avg_result(result_list)
        paths = write_to_csv(program, step, avg_result, filename_time)
        if cleanup_kernels:
            cleanup(paths)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(np.ceil(elapsed_time)), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"EST GADI: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, default="none", help='program')
    parser.add_argument("-s", type=int, default=1, help='step')
    parser.add_argument("-n", type=int, default=1, help='n tests')
    parser.add_argument("-f", type=str, default="", help='flag')
    main(parser.parse_args())
