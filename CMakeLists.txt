# Force use intel compiler, make sure the compiler is correctly loaded
#set(CMAKE_C_COMPILER "icx")
#set(CMAKE_CXX_COMPILER "icpx")

# CMake build system for cloth simulation
cmake_minimum_required(VERSION 3.12)
project(cloth VERSION 1.0.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# ##############################################################################
# enable C++  ##
# ##############################################################################
enable_language(CXX)

# ##############################################################################
# setup general C++ compilation flags#
# ##############################################################################

# add explictly debug symbols 
set(CMAKE_CXX_FLAGS " -g ")
# add g++ specific flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS}  -mavx2 -fpic -Wall -Wextra -Wno-deprecated-declarations -Wpedantic"
  )
endif()

# add icc specific flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=vec -qopt-report-file=stdout"
  )
endif()

# OpenGL
find_package(OpenGL REQUIRED)
message(STATUS "OpenGL included at ${OPENGL_INCLUDE_DIR}")

# GLUT
find_package(GLUT REQUIRED)
message(STATUS "GLUT included at ${GLUT_INCLUDE_DIRS}")
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework GLUT")
endif()

# OpenMP
find_package(OpenMP)
message(STATUS "OpenMP included at ${OPENMP_INCLUDE_DIR}")

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

# Check if the papi_avail executable was found
find_program(PAPI_FOUND NAMES papi_avail)
if(PAPI_FOUND)
  message(STATUS "PAPI found at ${PAPI_FOUND_DIR}")
else()
  message(STATUS "PAPI NOT found")
endif()

add_executable(
        opengl_main
        opengl_main.cpp
        cloth_code.cpp)


add_executable(
        kernel_main
        kernel_main.cpp
        cloth_code.cpp
        myprofiler.cpp
)

add_executable(
        kernel_opt
        kernel_opt.cpp
        cloth_code_opt.cpp
        myprofiler.cpp
)

add_executable(
        kernel_sse
        kernel_sse.cpp
        cloth_code_sse.cpp
        myprofiler.cpp
)

add_executable(
        kernel_vect_omp
        kernel_vect_omp.cpp
        cloth_code_vect_omp.cpp
        myprofiler.cpp
)

add_executable(
        kernel_omp
        kernel_omp.cpp
        cloth_code_omp.cpp
        myprofiler.cpp
)

target_compile_options(kernel_main PRIVATE -no-vec)
target_compile_options(kernel_opt PRIVATE -no-vec)
target_compile_options(kernel_sse PRIVATE -march=core-avx2 -O3)
target_compile_options(kernel_vect_omp PRIVATE -qopenmp-simd -march=core-avx2 -O3)
target_compile_options(kernel_omp PRIVATE -march=core-avx2 -O3)

if(PAPI_FOUND)
  target_link_libraries(kernel_main m papi)
  target_link_libraries(kernel_opt m papi)
  target_link_libraries(kernel_sse m papi)
  target_link_libraries(kernel_vect_omp m papi)
  target_link_libraries(kernel_omp m papi)
  target_compile_definitions(kernel_main PRIVATE PAPI)
  target_compile_definitions(kernel_opt PRIVATE PAPI)
  target_compile_definitions(kernel_sse PRIVATE PAPI)
  target_compile_definitions(kernel_vect_omp PRIVATE PAPI)
  target_compile_definitions(kernel_omp PRIVATE PAPI)
endif()

if(OpenMP_FOUND)
  target_link_libraries(kernel_main m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_opt m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_sse m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_vect_omp m OpenMP::OpenMP_CXX)
  target_link_libraries(kernel_omp m OpenMP::OpenMP_CXX)
  target_link_libraries(opengl_main m ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}
          OpenMP::OpenMP_CXX)
else()
  target_link_libraries(kernel_main m)
  target_link_libraries(kernel_opt m)
  target_link_libraries(kernel_sse m)
  target_link_libraries(kernel_vect_omp m)
  target_link_libraries(kernel_omp m)
  target_link_libraries(opengl_main m ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
endif()
