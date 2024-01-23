#import "./report/template.typ": *

#show: project.with(
  title: "Cloth Simulation Optimisation Report",
  author: "Will (Bogong) Wang",
)

#show figure: set block(breakable: true)
#set heading(numbering: "1.1.1.")

// -----------------------------------------------------------------------------
= Initial Performance Assessment 

#question("Motivation: gather performance data for different problem sizes as a baseline for future comparison.")

After conducting an analysis of the source file `cloth_code_main.cpp`, we can deduce that the computational complexity of the function loop_code is given by $cal(O)(n^2 (2d + 1)^2 + 6 n^2) = cal(O)(n^2 d^2)$, where $n$ represents the number of nodes per dimension and $d$ signifies the level of node interaction. 
Subsequently, we evaluate the performance of the function kernel_main under varying conditions of $n$ and $d$.

The overall complexity of the cloth simulation algorithm can be expressed as $cal(O)(n^2 d^2 i)$, taking into account that the simulation must be executed for $i$ iterations.

For the experimental evaluation, the default settings are adopted as follows: $s = 1.0$, $m = 1.0$, $f = 10.0$, $g = 0.981$, $b = 3.0$, $o = 0.0$, $t = 0.05$, and $i = 100$. 
Performance metrics are accumulated under varied $n$ and $d$, and are tabulated in reference @tab:main. The nomenclature for column headers is elucidated in @tab:name_table.

@fig:complexity reveals that the observed program complexity, quantified in terms of wall time, aligns closely with the anticipated complexity.
#figure(
  image("./report/step2/complexity.png"),
  caption: "Expected performance vs. actual performance for different problem sizes"
)<fig:complexity>
#result_table("./step2/main", "Performance data of " + `kernel_main` + " for different problem sizes")<tab:main>

// -----------------------------------------------------------------------------
#pagebreak()
= Serial Code Optimisation

#question("Motivation: optimise the serial code before parallelisation.")

== Memory Access Optimisation

=== Optimise memory access pattern and cache locality


Optimising memory access patterns enhances the spatial locality within the cache, thereby contributing to improved memory performance. 
An example is illustrated below to substantiate this claim. A comparative analysis reveals that prior to optimisation, memory is accessed with a stride of $n$. 
After optimisation, the stride is reduced to 1, thereby making memory access more efficient.
#comparev(
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    x[j * n + i] += dt * (vx[j * n + i] + dt * fx[j * n + i] * 0.5 / mass);
    oldfx[j * n + i] = fx[j * n + i];
  }
}
...
```,
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    x[i * n + j] += dt * (vx[i * n + j] + dt * fx[i * n + j] * 0.5 / mass);
    oldfx[i * n + j] = fx[i * n + j];
  }
}
...
```,
placement: none
)<comp:mem_acc_pat>

Another optimisation is focusing on temporal locality, which aims to keep frequently or recently accessed data in cache memory for quicker retrieval.
An example is shown below.
#comparev(
placement: none,
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    x[i * n + j] += dt * (vx[i * n + j] + dt * fx[i * n + j] * 0.5 / mass);
    ...
  }
}
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    y[i * n + j] += dt * (vy[i * n + j] + dt * fy[i * n + j] * 0.5 / mass);
    ...
  }
}
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    z[i * n + j] += dt * (vz[i * n + j] + dt * fz[i * n + j] * 0.5 / mass);
    ...
  }
}
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    xdiff = x[i * n + j] - xball;
    ydiff = y[i * n + j] - yball;
    zdiff = z[i * n + j] - zball;
    ...
  }
}
```,
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    x[i * n + j] += dt * (vx[i * n + j] + dt * fx[i * n + j] * 0.5 / mass);
    ...
    y[i * n + j] += dt * (vy[i * n + j] + dt * fy[i * n + j] * 0.5 / mass);
    ...
    z[i * n + j] += dt * (vz[i * n + j] + dt * fz[i * n + j] * 0.5 / mass);
    ...
    xdiff = x[i * n + j] - xball;
    ydiff = y[i * n + j] - yball;
    zdiff = z[i * n + j] - zball;
    ...
  }
}
...
```
)

=== Reduce memory access

Memory access is inherently time-intensive; therefore, the second optimisation aims to minimise the frequency of both read and write operations. 
This can be achieved through the frequent use of registers to obviate redundant memory accesses. 
Accessing data from registers is typically the fastest method, often requiring only a single cycle. 
Temporary variables can be created and employed for this purpose, serving as in-register storage. 
A comparative example showing this optimisation is presented below.
#compare(
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    xdiff = x[i * n + j] - ...;
    ...
    if (vmag < rball) {
      ...
      x[i * n + j] = xball + ...;
      ...
    }
  }
} 
...
 
```,
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    double x_tmp = x[i * n + j];
    xdiff = x_tmp - ...;
    ...
    if (vmag < rball) {
      ...
      x_tmp = xball + ...;
      ...
    }
  }
} 
...
```
)
=== Results
Subsequent to the implementation of memory access optimisation, performance metrics were re-acquired and are delineated in @fig:opt_mem and @tab:opt_mem. 
The data indicates a notable enhancement in performance, resulting a 14% reduction in wall time. 
Moreover, a significant cache performance also shows. 
The optimised code has decreases in L1 and L2 cache misses was observed, registering reductions of 64% and 51%, respectively.
#image("./report/step2/mem_wt.png")
#v(-0.4cm)
#image("./report/step2/mem_l1.png")
#v(-0.4cm)
#figure(
  image("./report/step2/mem_l2.png"),
  caption: "Performance comparison between " + `kernel_main` + " and "+ `kernel_opt` + " after memory access optimisation"
)<fig:opt_mem>

#result_table("./step2/opt_mem", "Full performance data of " + `kernel_opt` + " after memory access optimisation", placement: none)<tab:opt_mem>

== Computational Optimisation

=== Avoid redundant computations

Reduce redundant computations can directly reduce the number of instructions for calculations, 
hence increases the speed of the program.
An example of such optimisation is shown below.
#comparev(
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    x[i * n + j] += dt * (vx[i * n + j] + dt * fx[i * n + j] * 0.5 / mass);
    ...
    y[i * n + j] += dt * (vy[i * n + j] + dt * fy[i * n + j] * 0.5 / mass);
    ...
    z[i * n + j] += dt * (vz[i * n + j] + dt * fz[i * n + j] * 0.5 / mass);
    ...
  }
}
...
```,
```c
double half_dt_div_mass = dt * 0.5 / mass;
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    x[i * n + j] += dt * (vx[i * n + j] + fx[i * n + j] * half_dt_dv_mass);
    ...
    y[i * n + j] += dt * (vy[i * n + j] + fy[i * n + j] * half_dt_dv_mass);
    ...
    z[i * n + j] += dt * (vz[i * n + j] + fz[i * n + j] * half_dt_dv_mass);
    ...
  }
}
...
```
)

=== Remove branches
Reducing branches improves CPU pipelining by making instruction flow more predictable. It also enables better vectorisation by allowing for efficient use of single instruction multiple data (SIMD) operations.
There are generally two ways of removing branches in a loop, reorganising the loop structure and masking.

Below example shows how reorganising work. In our code, we avoid the branches by separating the loop body into 4 parts,
top stride, middle-left stride, middle-right stride and bottom stride.
#comparev(
  placement: none,
```c
for (ny = 0; ny < n; ny++) {
  for (nx = 0; nx < n; nx++) {
    ...
    // loop over displacements
    for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++) {
      for (dx = MAX(nx - delta, 0); dx < MIN(nx + delta + 1, n); dx++) {
        // exclude self interaction
        if (nx != dx || ny != dy) {
          ...
        }
      }
    }
  }
}
```,
```c
for (ny = 0; ny < n; ny++) {
  for (nx = 0; nx < n; nx++) {
    ...
    dx_start = MAX(nx - delta, 0), dx_end = MIN(nx + delta + 1, n);
    dy_start = MAX(ny - delta, 0), dy_end = MIN(ny + delta + 1, n);
    // Top stride
    for (dx = dx_start; dx < nx; dx++) {
      for (dy = dy_start; dy < dy_end; dy++) {
        ...
      }
    }

    // Middle strides
    dx = nx;
    for (dy = dy_start; dy < ny; dy++) {
      ...
    }
    for (dy = ny + 1; dy < dy_end; dy++) {
      ...
    }

    // Bottom stride
    for (dx = nx + 1; dx < dx_end; dx++) {
      for (dy = dy_start; dy < dy_end; dy++) {
        ...
      }
    }
  }
}
```
)
Masking optimisation was initially explored, however, it was ultimately omitted due to its computational inefficiency. 
As evidenced by the subsequent example, the removal of branches paradoxically increased the number of floating-point operations, negating the intended benefits. 
Consequently, this optimisation was not incorporated into the final optimisation.
#comparev(
placement: none,
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    ...
    vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
    if (vmag < rball) {
      inv_vmag = 1 / (vmag);
      ...
      x_tmp = xball + xdiff_unit * rball;
      y_tmp = yball + ydiff_unit * rball;
      z_tmp = zball + zdiff_unit * rball;
      ...
      *x_vel = 0.1 * (*x_vel - xdiff_unit * proj_scalar);
      *y_vel = 0.1 * (*y_vel - ydiff_unit * proj_scalar);
      *z_vel = 0.1 * (*z_vel - zdiff_unit * proj_scalar);
    }
    ...
  }
}
```,
```c
for (i = 0; i < n; i++) {
  for (j = 0; j < n; j++) {
    ...
    vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
    double mask = vmag < rball;
    double inv_mask = vmag >= rball;
    inv_vmag = mask / (vmag + inv_mask);
    ...
    x_tmp = xball + xdiff_unit * rball * mask + x_tmp * inv_mask;
    y_tmp = yball + ydiff_unit * rball * mask + y_tmp * inv_mask;
    z_tmp = zball + zdiff_unit * rball * mask + z_tmp * inv_mask;
    ...
    proj_scalar = (*x_vel * xdiff_unit + *y_vel * ydiff_unit + *z_vel * zdiff_unit);
    *x_vel = 0.1 * (*x_vel - xdiff_unit * proj_scalar) * mask + *x_vel * inv_mask;
    *y_vel = 0.1 * (*y_vel - ydiff_unit * proj_scalar) * mask + *y_vel * inv_mask;
    *z_vel = 0.1 * (*z_vel - zdiff_unit * proj_scalar) * mask + *z_vel * inv_mask;
    ...
  }
}
```
)

=== Use lookup table

Using a lookup table achieves rapid data retrieval and enhances computational efficiency by eliminating the need for redundant calculations.
In our case, we can observe that the reference distance between two nodes can be reused.
The optimisation is as follows.
#comparev(
placement: none,
```c
for (ny = 0; ny < n; ny++) {
  for (nx = 0; nx < n; nx++) {
    ...
    // Top stride
    for (dx = dx_start; dx < nx; dx++) {
      for (dy = dy_start; dy < dy_end; dy++) {
        rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
        ...
      }
    }

    // Middle strides
    dx = nx;
    for (dy = dy_start; dy < ny; dy++) {
      rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
      ...
    }
    for (dy = ny + 1; dy < dy_end; dy++) {
      rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
      ...
    }

    // Bottom stride
    for (dx = nx + 1; dx < dx_end; dx++) {
      for (dy = dy_start; dy < dy_end; dy++) {
        rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
        ...
      }
    }
  }
}
```,
```c
// Pre-compute reference distance
int size = (2 * delta + 1);
for (dx = -delta; dx <= delta; dx++) {
  for (dy = -delta; dy <= delta; dy++) {
    loop_idx = (dx + delta) * size + (dy + delta);
    rlen_table[loop_idx] = sqrt((double) (dx * dx + dy * dy)) * sep;
  }
}

for (ny = 0; ny < n; ny++) {
  for (nx = 0; nx < n; nx++) {
    ...
    // Top stride
    for (dx = dx_start; dx < nx; dx++) {
      for (dy = dy_start; dy < dy_end; dy++) {
        rlen = rlen_table[(dx - nx + delta) * size + (dy -ny + delta)];
        ...
      }
    }

    // Middle strides
    dx = nx;
    for (dy = dy_start; dy < ny; dy++) {
      rlen = rlen_table[(dx - nx + delta) * size + (dy -ny + delta)];
      ...
    }
    for (dy = ny + 1; dy < dy_end; dy++) {
      rlen = rlen_table[(dx - nx + delta) * size + (dy -ny + delta)];
      ...
    }

    // Bottom stride
    for (dx = nx + 1; dx < dx_end; dx++) {
      for (dy = dy_start; dy < dy_end; dy++) {
        rlen = rlen_table[(dx - nx + delta) * size + (dy -ny + delta)];
        ...
      }
    }
  }
}
```
)

=== Results
After conducting computational optimisation, performance metrics were re-evaluated and are presented in @tab:opt_comp and @fig:opt_comp. 
The data reveal a significant enhancement in performance, with a 52% increase following memory access optimisation and a 58% improvement relative to the baseline metrics. 
Additionally, a 9% reduction in total floating-point operations was observed, which correlates with a 37% decline in the overall instruction count. 
The branch misprediction rate also exhibited a notable reduction, averaging a 20% decrease.

#image("./report/step2/comp_wt.png")
#v(-0.4cm)
#image("./report/step2/comp_fp.png")
#v(-0.4cm)
#figure(
  image("./report/step2/comp_br.png"),
  caption: "Performance comparison between " + `kernel_main` + " and "+ `kernel_opt` + " after computational optimisation"
)<fig:opt_comp>
#result_table("./step2/opt_comp", "Full performance data of " + `kernel_opt` + " after computational optimisation")<tab:opt_comp>

//------------------------------------------------------------------------------
#pagebreak()
= Vectorisation with AVX2 and OpenMP

#question("Motivation: compare the performance difference between manual and compiler vectorisation.")
== Manual Vectorisation
To manually vectorise the code, the AVX2 intrinsic functions are used to vectorise the code. 
The manually vectorised code can be found in `code_cloth_sse.cpp`
To compile the code, we use compiler instruction `-march=core-avx2 -O3` to enable AVX2 intrinsic functions support.

To optimise the code with AVX2 intrinsics, we manually vectorised most of the loops by unrolling loops to handle four items at a time.
The memory alignment for all the arrays are also employed to maximise the memory efficiency.

== Compiler Vectorisation
The openmp vectorised code can be found in `code_cloth_vect_omp.cpp`.
```c #pragma omp parallel``` is used to vectorise different loops.
There are seven loops intended to be vectorised with openmp's vectorisation directive, 
the vectorisation can be checked via intel advisor. 
The profile setting is: `n = 1000, d = 8, s = 1.0`, `m = 1.0`, `f = 10.0`, `g = 0.981`, `b = 3.0`, `o = 0.0`, `t = 0.05`, `i = 100`.

#linebreak()
There are seven loops in the loop code (we use Intel Advisor to help to analyse the compiler-vectorisation):
+ The loop updates position of the node and update velocity if the node collide with the ball was successfully vectorised.
  #image("./report/step3/loop1.png")
+ The loop pre-computes the reference distance was successfully vectorised.
  #image("./report/step3/loop2.png")
+ The loop updates energy and force from top stride was successfully vectorised.
  #image("./report/step3/loop3.png")
+ The loop updates energy and force from middle-left stride was successfully vectorised.
  #image("./report/step3/loop4.png")
+ The loop updates energy and force from middle-right stride was successfully vectorised.
  #image("./report/step3/loop5.png")
+ The loop updates energy and force from bottom stride was not vectorised.
  #image("./report/step3/loop6.png")
  From the below compiler message, since there is no data dependency exist between iterations, 
  the reason for failure vectorisation might be the loop index is too complex for compiler to analyse. 
  #linebreak()
  #block(
    fill: luma(240),
    width: 100%,
    inset: 8pt,
    radius: 4pt,
  [
    #align(
      center,
      `Source Code`
    )
    ```c
    int size = (2 * delta + 1);
    for (dx = -delta; dx <= delta; dx++) {
      #pragma omp simd
      for (dy = -delta; dy <= delta; dy++) {
        loop_idx = (dx + delta) * size + (dy + delta);
        rlen_table[loop_idx] = sqrt((double) (dx * dx + dy * dy)) * sep;
      }
    }
    ```
  ]
  )
  #block(
    fill: luma(240),
    width: 100%,
    inset: 8pt,
    radius: 4pt,
  [
    #align(
      center,
      `Error Message`
    )
  ```
  cloth_code_vect_omp.cpp:151:5: warning: loop not vectorised: the optimizer was unable to perform the requested transformation; the transformation might be disabled or specified as part of an unsupported transformation ordering [-Wpass-failed=transform-warning] 
  #pragma omp simd
  ```
  ]
  )
+ The loop adds a damping factor to set velocity to zero and calculate kinetic energy was successfully vectorised.
  #image("./report/step3/loop7.png")


== Performance Comparison
The performance data was collected for manual vectorised (`kernel_sse`) and openmp (`kernel_vect_omp`) vectorised code and shown in @tab:sse, @tab:vect_omp and plotted in @fig:vect.
After some inspecting tables and charts, we can observe that the code uses AVX2 intrinsic functions (`kernel_sse`) provides an average of 62% speed up,
especially for $d$ is large, because when $d$ is small the vectorisation will not apply.

Similarly, the openmp vectorised code provides an approximately the same speed up, the performance difference to `kernel_sse` is marginal (less than 1%),
the 1% difference may result from not vectorising the loop for calculating reference length.

We can also observe that as the level of node interaction level increases, there is a corresponding increase in the MFLOPs. 
This can be attributed to the vectorisation, which yields computational benefits for the programme.

#image("./report/step3/wt.png")
#v(-0.4cm)
#image("./report/step3/ins.png")
#v(-0.4cm)
#figure(
  image("./report/step3/vec.png"),
  caption: "Performance comparison between " + `kernel_main` + " and "+ `kernel_opt` + " after computational optimisation"
)<fig:vect>
#result_table("./step3/sse", "Full performance data of " + `kernel_sse`)<tab:sse>
#result_table("./step3/vect_omp", "Full performance data of " + `kernel_vect_omp`)<tab:vect_omp>

// -----------------------------------------------------------------------------
#pagebreak()
= Parallelisation using OpenMP
#question("Motivation: analyse the performance increase after combining parallelisation and vectorisation.")
To apply the parallelisation, the vectorisation code was modified accordingly.
Since openmp is not supporting reduction for custom operators (AVX add intrinsic functions),
we change the code accordingly.
After changed the code, even though a some overhead was introduced on executing `HSUM_PD`, we avoid the race condition of `pe_vec` variable and maximises the parallel performance (no need to use lock).

#comparev(
```c
DOUBLE_VEC pe_vec = SET_PD(0.0);
for (ny = 0; ny < n; ny++) {
  for (nx = 0; nx < n; nx++) {
    ...
    // Top stride
    for (dx = dx_start; dx < nx; dx++) {
      simd_bound = dy_end - ((dy_end - dy_start) % 4);
      for (dy = dy_start; dy < simd_bound; dy += 4) {
        calc_interaction_avx(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp_vec, y_tmp_vec, z_tmp_vec, &fx_tmp_vec, &fy_tmp_vec, &fz_tmp_vec, &pe_vec, rlen_table);0
      }
      ...
    }
    ...
  }
}
return 0.5 * (pe + HSUM_PD(pe_vec));
```,
```c
#pragma omp parallel for ...
for (ny = 0; ny < n; ny++) {
  for (nx = 0; nx < n; nx++) {
    DOUBLE_VEC pe_vec = SET_PD(0.0);
    ...
    // Top stride
    for (dx = dx_start; dx < nx; dx++) {
      simd_bound = dy_end - ((dy_end - dy_start) % 4);
      for (dy = dy_start; dy < simd_bound; dy += 4) {
        calc_interaction_avx(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp_vec, y_tmp_vec, z_tmp_vec, &fx_tmp_vec, &fy_tmp_vec, &fz_tmp_vec, &pe_vec, rlen_table);0
      }
      ...
    }
    ...
  }
  pe += HSUM_PD(pe_vec);
}
return 0.5 * pe
```
)

The parallelisation directive is placed in the following loops:
  + The loop updates position of the node and update velocity if the node collide.
  + The loop pre-computes the reference distance.
  + The loop updates energy and force for all the particles.
  + The loop adds a damping factor to set velocity to zero and calculates kinetic energy.

== Experiment settings
Since in multi-thread environment, the openmp's result may be inaccurate, for this section, we focus on program wall time relating to different number of threads, problem size and chunk size.
For this performance testing, we are using 24 cpus, 96 GB of memory.
#linebreak()
In this section, we measured the performance data for 
  + different problem size ($n = [20, 400, 1000, 2000]$, $d = [2, 4, 8]$)
    - different $n$ and $d$ represents different problem size
    - for $n$: 20: small problem, 400: medium problem, 1000: large problem, 2000: super large problem
    - for $d$: 2: small problem, 4: medium problem, 8: large problem
  + different number of threads ($p = [1, 4, 8, 24, 48]$)
    - since we will be test on a 24 cores environment, we will test the performance of the program on a multiple of 24
    - for different $p$, we have below test purposes:
      - 1: test the performance for different scheduling strategy 
      - 4 & 8: test the performance of correlation between different problem size
      - 24: mainly test the performance if we use all the available threads and the thread overhead
      - 24: mainly test the performance if we use extra number of threads thread overhead
  + different scheduling strategies (dynamic, static with chunk size = $[1, 4, 16, 64, n / p]$)
    - the purpose of different scheduling strategies are listed below.
      - `dynamic`: let openmp to decide scheduling
      - `static, [1|4]`: small chunk size
      - `static, [16|64]`: round robin scheduling with medium and large chunk size
      - `static, n/p`: evenly distribute the tasks to each threads
After testing with above designed test parameter, we get below result and we can conclude following findings.

== Experiment Results
=== Relation to Amdahl's Law
Amdahl's Law describes the speedup in performance that can be gained from improving a particular part of a system. It's often used to predict the maximum speedup from parallelising a program. The formula is:
$
"Speedup" = frac(1, (1 + P) + P/S),
$
where
- $P$ is the proportion of the program that can be parallelised.
- $S$ is the speedup of the parallelised part.

=== Analysis of threads utilisation
Our performance data reveals a correlation between the number of threads and program performance, as visualised in @fig:nthreads. 
Several factors contribute to this performance variation:

- Concurrency: Leveraging multi-core CPUs allows for simultaneous task execution, optimising computational resources.
- Load Balancing: Distributing computational tasks across multiple threads mitigates the risk of bottlenecking.
- Resource Utilisation: Effective allocation of CPU and memory resources minimises idle times.

Incorporating Amdahl's Law, these elements contribute to the $P$ variable, which represents the proportion of the program that benefits from parallelisation. This allows us to theoretically estimate the upper limit of achievable speedup.
#figure(
  image("./report/step4/nthreads.png"),
  caption: "Performance comparison between " + `kernel_main` + " and "+ `kernel_opt` + " after computational optimisation"
)<fig:nthreads>

However, an excess of threads introduces new performance issue:
- Thread Creation and Termination Overhead: As evidenced by the performance drop when $n=20,d=2$ 
  and $p$ transitions from 4 to 48 (@fig:init_threads, left graph), the overhead of thread management depresses the 
  $S$ variable in Amdahl's Law, impeding performance gains.
- Context Switching and Resource Contention: When $n=2000,d=8$, a performance decline is observed between 24 and 48 threads (fig:init_threads, right graph). 
  This aligns with Amdahl's Law by further diminishing the $S$ value, corroborating the limitations in attainable speedup.

#figure(
image(width: 100%, "./report/step4/init_threads.png"),
caption: "Performance comparison between different scheduling strategy and different number of threads"
)<fig:init_threads>


=== Analysis of Scheduling Strategies
Our findings, substantiated by @fig:schedule_nthreads_problem_size, can be summarised thus:
- Dynamic Scheduling: 
  #linebreak()
  Generally, this strategy yields robust performance across an array of problem sizes and thread counts. 
  However, the intrinsic overhead of dynamic task allocation impacts the $S$ variable in Amdahl's Law, elucidating the performance dip as the thread count escalates.

- Static Scheduling with Fixed Chunk Sizes: 
  #linebreak()
  For smaller chunk sizes (1 and 4) yields the best performance. 
  In contrast, larger chunk sizes (16 and 64) engender workload imbalances, thereby reducing the $P$ variable in Amdahl's Law and limiting speedup potential.

- Static Scheduling with Dynamic Chunk Size (`static,n/p`): 
  #linebreak()
  This strategy yields suboptimal results. The inefficiency in chunk size allocation leads to unbalanced workload (especially in the loop for updating velocity), which affects both 
  $P$ and $S$ in Amdahl's Law.
#image("./report/step4/chunksize_0.png")
#v(-0.4cm)
#image("./report/step4/chunksize_1.png")
#v(-0.4cm)
#image("./report/step4/chunksize_2.png")
#v(-0.4cm)
#image("./report/step4/chunksize_3.png")
#v(-0.4cm)
#figure(
image(width: 100%, "./report/step4/chunksize_4.png"),
caption: "Performance comparison between different scheduling strategy, number of threads and problem size"
)<fig:schedule_nthreads_problem_size>


#grid(
  columns: 2,
  column-gutter: (1em),
  figure(
    table(
      columns: (1fr, 1fr, 1fr, 2fr),
      align: horizon+center,
      row-gutter: (4pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt),
      [*$n$*], [*$d$*], [*$p$*], [*wall time ($mu s$)*],
      ..csv("./report/step4/omp_dynamic.csv", delimiter: ";").flatten(),
    ),
    caption: "Full performance data of " + `kernel_omp` + " with dynamic scheduling" 
  ),
  figure(
    table(
      columns: (1fr, 1fr, 1fr, 2fr),
      align: horizon+center,
      row-gutter: (4pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt),
      [*$n$*], [*$d$*], [*$p$*], [*wall time ($mu s$)*],
      ..csv("./report/step4/omp_static_1.csv", delimiter: ";").flatten(),
    ),
    caption: "Full performance data of " + `kernel_omp` + " with static scheduling, chunk size 1" 
  )
)<fig:>

#grid(
  columns: 2,
  column-gutter: (1em),
  figure(
    table(
      columns: (1fr, 1fr, 1fr, 2fr),
      align: horizon+center,
      row-gutter: (4pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt),
      [*$n$*], [*$d$*], [*$p$*], [*wall time ($mu s$)*],
      ..csv("./report/step4/omp_static_4.csv", delimiter: ";").flatten(),
    ),
    caption: "Full performance data of " + `kernel_omp` + " with static scheduling, chunk size 4" 
  ),
  figure(
    table(
      columns: (1fr, 1fr, 1fr, 2fr),
      align: horizon+center,
      row-gutter: (4pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt),
      [*$n$*], [*$d$*], [*$p$*], [*wall time ($mu s$)*],
      ..csv("./report/step4/omp_static_16.csv", delimiter: ";").flatten(),
    ),
    caption: "Full performance data of " + `kernel_omp` + " with static scheduling, chunk size 16" 
  )
)

#grid(
  columns: 2,
  column-gutter: (1em),
  figure(
    table(
      columns: (1fr, 1fr, 1fr, 2fr),
      align: horizon+center,
      row-gutter: (4pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt),
      [*$n$*], [*$d$*], [*$p$*], [*wall time ($mu s$)*],
      ..csv("./report/step4/omp_static_64.csv", delimiter: ";").flatten(),
    ),
    caption: "Full performance data of " + `kernel_omp` + " with static scheduling, chunk size 64" 
  ),
  figure(
    table(
      columns: (1fr, 1fr, 1fr, 2fr),
      align: horizon+center,
      row-gutter: (4pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 1pt, 0pt, 0pt, 0pt, 0pt, 2pt, 0pt),
      [*$n$*], [*$d$*], [*$p$*], [*wall time ($mu s$)*],
      ..csv("./report/step4/omp_static_np.csv", delimiter: ";").flatten(),
    ),
    caption: "Full performance data of " + `kernel_omp` + " with static scheduling, chunk size " + $n / p$ 
  )
)

//------------------------------------------------------------------------------
#pagebreak()
= Roofline analysis 
#question("Motivation: find the performance bottleneck of the program (whether it's compute-bounded or IO-bounded).")
The intel advisor is used to generate the roofline plot shown in roofline.
The profile settings is as follows:
$n = 2000$ $d = 8$ $p = 48$ $s = 1.0$, $m = 1.0$, $f = 10.0$, $g = 0.981$, $b = 3.0$, $o = 0.0$, $t = 0.05$, and $i = 100$.

From the plot, we can see that the yellow and red dots (corresponds to `calc_interaction_avx` function) are closer to the peak computational performance line (the upper horizontal line), hence `calc_interaction_avx` is computational bounded.
In addition, since the function `calc_interaction_avx` is nested in the most computationally intensive ($cal(O)(n^2d^2)$ complexity) loop,
hence we conclude that the program is compute bounded.

To even further improve the performance of the current program, we can use AVX-512 intrinsics function, as it will maximise the power of SIMD, 
however, it may introduce other performance degrade problem caused by overheat. 

#figure(
  image("./report/step5/roofline.svg"),
  caption: "Roofline plot for " + `kernel_omp`
)<fig:roofline>


#pagebreak()
= Appendix
#v(12pt)
#figure(
  table(
    columns: (1fr, 3fr),
    align: center + horizon,
    row-gutter: (3pt, 0pt),
    [*Name*],             [*Meaning*],
    [$n$],                [Nodes per dimenstion],
    [$d$],                [Node interaction level],
    [$p$],                [Maximum number of OpenMP threads],
    [wall time ($mu s$)], [Program wall time after $i$ iterations],
    [`PAPI_DP_OPS`],      [Floating point operations; optimized to count scaled double precision vector operations],
    [`PAPI_L1_DCM`],      [Level 1 data cache misses],
    [`PAPI_L2_DCM`],      [Level 2 data cache misses],
    [`PAPI_TOT_INS`],     [Instructions completed],
    [`PAPI_BR_MSP`],      [Conditional branch instructions mispredicted],
    [`PAPI_VEC_DP`],      [Double precision vector/SIMD instructions],
  ),
  caption: "Performance metrices names"
)<tab:name_table>