# Optimise Cloth Simulation Program

Technical report for this optimisation is available [here](https://github.com/Wi11Wang/cloth-simulation-optimisation/blob/main/technical_report.pdf).

## Project Background

This project aims to optimise a serial cloth simulation program: a piece of cloth falling under gravity onto a stationary spherical ball. The cloth is modelled as a 2-D rectangular network, where node $(x,y)$ ($x$ and $y$ are integers) in an ($N*N$) square network interacts with all other nodes $(x',y')$ such that $(x-\delta \le x' \le x+\delta,y-\delta \le y' \le y+\delta)$ where $\delta$ is some (low value) integer. Thus if $\delta=1$ a typical node in the center of the cloth will interact with 8 neighboring nodes, while if $\delta=2$ there are 24 interactions to consider. When a node is located near the cloth edge there will be fewer interactions. Thus in contrast to the MD simulation, each node in the network interacts with a finite number of other nodes, making the evaluation of the total interaction potential scale as $O(N)$, where N is the number of nodes.

Between each pair of nodes $i$ and $j$, we define an interaction given by [Hooke's law](http://en.wikipedia.org/wiki/Hooke's_law). The potential energy is

```math
PE_{ij} = K * (R_{ij}-E_{ij})
```

Where $K$ is the force constant determining how stiff the spring (or cloth) is, $R_{ij}$ is the Euclidean distance between the two nodes and $E_{ij}$ is the equilibrium distance between these two nodes.
For example if node (1,1) and node (1,2) have equilibrium distance $E_{(1,1)(1,2)}=d$ then node (1,1) and node (2,2) have equilibrium distance $E_{(1,1)(2,2)}=d\sqrt{2}$. Exactly like the first assignment, the force on each node can be calculated using the first derivative of the potential energy such that the force contribution on node $i$ from node $j$ assuming they are within the required distance is

```math
F_{x_{ij}} = K * \frac{(R_{ij}-E_{ij}) * (x_i - x_j)}{R_{ij}}
```

Each node is given a constant mass $m$, noting that the force and acceleration are related by $F=ma$. The cloth is initially positioned in the xz plane and subjected to a gravitational force g in the y direction. As a consequence, the cloth will slowly fall under gravity.

Positioned below the cloth is a ball of radius, r, such that the centre of the cloth is located 1+r units above the centre of the ball. The cloth is allowed to fall under gravity until it collides with the ball. The motions of the nodes using the same velocity verlet algorithm that are commonly used for the molecular dynamics program. The difference arises when a node in the cloth hits the ball. You detect this by noticing that the updated position of the node is within the radius of the ball. At this point, you move the node to the nearest point on the surface of the ball.

## Project Structure

- `cloth_code_[main|opt|vect_omp|sse|omp].cpp`: source code for different implementations `main`, `opt`, `vect_omp`, `sse` and `omp`
- `cloth_code_[main|opt|vect_omp|sse|omp].h`: headers for different implementations
- `cloth_param.h`: modified parameters file, added support for accepting "-p" (number of threads)
- `kernel_*.cpp`: source code kernels for different implementations
- `myprofiler.cpp` and `myprofiler.h`: the custom profiler for different implementations, it uses `papi`
- `report/`: directory relates to writeup.pdf
- `CMakeLists.txt`: modified cmake file, added support for different implementations
- `auto_profile.py`: automatically profiles given implementation, to learn how to run it, type `python3 auto_profile.py -h`
- `auto_test.py`: modified auto test, capable of automatically test given implementation, to run it, type `python3 auto_test.py [main|opt|vect_omp|sse|omp]`
- `step[2|3|4]_[main|opt|vect_omp|sse|omp].sh`: driver file for running on supercomputer

## Compiling and building the code

This code uses a CMake build system.
In order to build the code you will need to have CMake version >= 3.12 installed.
If CMake is not installed on your system, please follow the installation instructions on the CMake website [https://cmake.org/install/](https://cmake.org/install/).
In order to build the code on your machine execute the following code from within your project directory (if you are running on supercomputer)

```sh
module load papi
module load gcc/12.2.0
module load intel-compiler
module load cmake/3.18.2
module load python3/3.8.5
module load python3-as-python

mkdir build
cd build
cmake ..
make
```

## Running the code

You can run the code simply with the command `./opengl_main` or `./kernel_main` from within the build folder. There are a number of additional parameters that you can modify when calling the program:

```text
Nodes_per_dimension:             -n int 
Grid_separation:                 -s float 
Mass_of_node:                    -m float 
Force_constant:                  -f float 
Node_interaction_level:          -d int 
Gravity:                         -g float 
Radius_of_ball:                  -b float 
offset_of_falling_cloth:         -o float 
timestep:                        -t float 
num iterations:                  -i int
Timesteps_per_display_update:    -u int 
Perform X timesteps without GUI: -x int
Rendermode (1 for face shade, 2 for vertex shade, 3 for no shade):
                                 -r (1,2,3)
```

So for example, the command ```./opengl_main -n 20 -i 1000``` will run the cloth code for 1000 timesteps with 20 nodes per dimension in the cloth using the visualization.


## Note

- The project will build with errors, I deliberately leave it here, it's from `kernel_vect_omp`, the error tells us which loop is unable to vectorise with openmp's directive.
- The project have to be built with intel compiler, if we build it with GNU compiler, there will be some memory alignment issue that causes segmentation fault.
