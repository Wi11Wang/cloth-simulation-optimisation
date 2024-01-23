#include "./cloth_code_vect_omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

using namespace std;

void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int delta, double UNUSED(grav), double sep,
                double rball, double offset, double UNUSED(dt), double **x,
                double **y, double **z, double **cpx, double **cpy,
                double **cpz, double **fx, double **fy, double **fz,
                double **vx, double **vy, double **vz, double **oldfx,
                double **oldfy, double **oldfz, double **rlen_table) {
  int i, nx, ny;

  // Free existing arrays
  _mm_free(*x);
  _mm_free(*y);
  _mm_free(*z);
  _mm_free(*cpx);
  _mm_free(*cpy);
  _mm_free(*cpz);

  // Allocate aligned arrays with padding
  *x = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *y = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *z = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *cpx = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *cpy = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *cpz = (double *) _mm_malloc(n * n * sizeof(double), 32);

  // Initialize coordinates
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      int loop_idx = nx * n + ny;
      (*x)[loop_idx] = nx * sep - (n - 1) * sep * 0.5 + offset;
      (*z)[loop_idx] = rball + 1;
      (*y)[loop_idx] = ny * sep - (n - 1) * sep * 0.5 + offset;
      (*cpx)[loop_idx] = 0;
      (*cpz)[loop_idx] = 1;
      (*cpy)[loop_idx] = 0;
    }
  }

  // Free old force and velocity arrays
  _mm_free(*fx);
  _mm_free(*fy);
  _mm_free(*fz);
  _mm_free(*vx);
  _mm_free(*vy);
  _mm_free(*vz);
  _mm_free(*oldfx);
  _mm_free(*oldfy);
  _mm_free(*oldfz);

  // Allocate new aligned arrays with padding
  *fx = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *fy = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *fz = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *vx = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *vy = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *vz = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *oldfx = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *oldfy = (double *) _mm_malloc(n * n * sizeof(double), 32);
  *oldfz = (double *) _mm_malloc(n * n * sizeof(double), 32);

  // Initialize arrays
  for (i = 0; i < n * n; i++) {
    (*vx)[i] = 0.0;
    (*vy)[i] = 0.0;
    (*vz)[i] = 0.0;
    (*fx)[i] = 0.0;
    (*fy)[i] = 0.0;
    (*fz)[i] = 0.0;
  }

  // Allocate aligned rlen lookup table
  *rlen_table = (double *) _mm_malloc((2 * delta + 1) * (2 * delta + 1) * sizeof(double), 32);
}

void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball, double zball, double dt,
              double *__restrict__ x, double *__restrict__ y, double *__restrict__ z,
              double **__restrict__ fx, double **__restrict__ fy, double **__restrict__ fz,
              double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
              double **__restrict__ oldfx, double **__restrict__ oldfy, double **__restrict__ oldfz,
              double *__restrict__ rlen_table, double *pe, double *ke, double *te) {
  int i, j, loop_idx;
  double xdiff, ydiff, zdiff, vmag, inv_vmag, damp, proj_scalar;
  double xdiff_unit, ydiff_unit, zdiff_unit;
  double x_tmp, y_tmp, z_tmp;
  double *x_vel, *y_vel, *z_vel;
  double half_dt_div_mass = dt * 0.5 / mass;

  for (i = 0; i < n; i++) {
#pragma omp simd
    for (j = 0; j < n; j++) {
      loop_idx = i * n + j;
      x_tmp = x[loop_idx];
      y_tmp = y[loop_idx];
      z_tmp = z[loop_idx];

      x_tmp += dt * (vx[loop_idx] + (*fx)[loop_idx] * half_dt_div_mass);
      y_tmp += dt * (vy[loop_idx] + (*fy)[loop_idx] * half_dt_div_mass);
      z_tmp += dt * (vz[loop_idx] + (*fz)[loop_idx] * half_dt_div_mass);

      //	apply constraints - push cloth outside of ball
      xdiff = x_tmp - xball;
      ydiff = y_tmp - yball;
      zdiff = z_tmp - zball;
      vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
      if (vmag < rball) {
        inv_vmag = 1 / vmag;
        // Orthogonal vector to the surface
        xdiff_unit = xdiff * inv_vmag;
        ydiff_unit = ydiff * inv_vmag;
        zdiff_unit = zdiff * inv_vmag;

        x_tmp = xball + xdiff_unit * rball;
        y_tmp = yball + ydiff_unit * rball;
        z_tmp = zball + zdiff_unit * rball;

        x_vel = &vx[loop_idx];
        y_vel = &vy[loop_idx];
        z_vel = &vz[loop_idx];
        proj_scalar = (*x_vel * xdiff_unit + *y_vel * ydiff_unit + *z_vel * zdiff_unit);
        *x_vel = 0.1 * (*x_vel - xdiff_unit * proj_scalar);
        *y_vel = 0.1 * (*y_vel - ydiff_unit * proj_scalar);
        *z_vel = 0.1 * (*z_vel - zdiff_unit * proj_scalar);
      }
      x[loop_idx] = x_tmp;
      y[loop_idx] = y_tmp;
      z[loop_idx] = z_tmp;
    }
  }

  *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, *oldfx, *oldfy, *oldfz, rlen_table);

  // Swap pointers
  swap(*fx, *oldfx);
  swap(*fy, *oldfy);
  swap(*fz, *oldfz);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  for (i = 0; i < n; i++) {
#pragma omp simd
    for (j = 0; j < n; j++) {
      loop_idx = i * n + j;
      vx[loop_idx] = (vx[loop_idx] +
                      ((*fx)[loop_idx] + (*oldfx)[loop_idx]) * half_dt_div_mass) *
                     damp;
      vy[loop_idx] = (vy[loop_idx] +
                      ((*fy)[loop_idx] + (*oldfy)[loop_idx]) * half_dt_div_mass) *
                     damp;
      vz[loop_idx] = (vz[loop_idx] +
                      ((*fz)[loop_idx] + (*oldfz)[loop_idx]) * half_dt_div_mass) *
                     damp;
      *ke += vx[loop_idx] * vx[loop_idx] + vy[loop_idx] * vy[loop_idx] +
             vz[loop_idx] * vz[loop_idx];
    }
  }
  *ke *= 0.5;
  *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep, double fcon,
                double *__restrict__ x, double *__restrict__ y, double *__restrict__ z,
                double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
                double *__restrict__ rlen_table) {
  double pe = 0.0, fx_tmp, fy_tmp, fz_tmp;
  int nx, ny, dx, dy, loop_idx;
  int dx_start, dx_end, dy_start, dy_end;
  double x_tmp, y_tmp, z_tmp;
  double rlen, xdiff, ydiff, zdiff, vmag, inv_vmag, tmp;
  int inner_loop_idx, rlen_index_x, rlen_index_y, lookup_idx;


  int size = (2 * delta + 1);

  for (dx = -delta; dx <= delta; dx++) {
#pragma omp simd
    for (dy = -delta; dy <= delta; dy++) {
      loop_idx = (dx + delta) * size + (dy + delta);
      rlen_table[loop_idx] = sqrt((double) (dx * dx + dy * dy)) * sep;
    }
  }

  // Loop over particles
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      fx_tmp = 0.0;
      fy_tmp = 0.0;
      fz_tmp = -mass * grav;
      loop_idx = nx * n + ny;

      x_tmp = x[loop_idx];
      y_tmp = y[loop_idx];
      z_tmp = z[loop_idx];

      dx_start = MAX(nx - delta, 0), dx_end = MIN(nx + delta + 1, n);
      dy_start = MAX(ny - delta, 0), dy_end = MIN(ny + delta + 1, n);

      // Top stride
      for (dx = dx_start; dx < nx; dx++) {
#pragma omp simd
        for (dy = dy_start; dy < dy_end; dy++) {
          inner_loop_idx = dx * n + dy;

          // Compute reference distance
          rlen_index_x = dx - nx + delta;
          rlen_index_y = dy - ny + delta;
          lookup_idx = rlen_index_x * size + rlen_index_y;
          rlen = rlen_table[lookup_idx];

          // Compute actual distance
          xdiff = x[inner_loop_idx] - x_tmp;
          ydiff = y[inner_loop_idx] - y_tmp;
          zdiff = z[inner_loop_idx] - z_tmp;
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          inv_vmag = 1 / vmag;

          // Potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          tmp = fcon - fcon * rlen * inv_vmag;
          fx_tmp += tmp * xdiff;
          fy_tmp += tmp * ydiff;
          fz_tmp += tmp * zdiff;
        }
      }

      // Middle strides
      dx = nx;
#pragma omp simd
      for (dy = dy_start; dy < ny; dy++) {
        inner_loop_idx = dx * n + dy;

        // Compute reference distance
        rlen_index_x = dx - nx + delta;
        rlen_index_y = dy - ny + delta;
        lookup_idx = rlen_index_x * size + rlen_index_y;
        rlen = rlen_table[lookup_idx];

        // Compute actual distance
        xdiff = x[inner_loop_idx] - x_tmp;
        ydiff = y[inner_loop_idx] - y_tmp;
        zdiff = z[inner_loop_idx] - z_tmp;
        vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
        inv_vmag = 1 / vmag;

        // Potential energy and force
        pe += fcon * (vmag - rlen) * (vmag - rlen);
        tmp = fcon - fcon * rlen * inv_vmag;
        fx_tmp += tmp * xdiff;
        fy_tmp += tmp * ydiff;
        fz_tmp += tmp * zdiff;
      }
#pragma omp simd
      for (dy = ny + 1; dy < dy_end; dy++) {
        inner_loop_idx = dx * n + dy;

        // Compute reference distance
        rlen_index_x = dx - nx + delta;
        rlen_index_y = dy - ny + delta;
        lookup_idx = rlen_index_x * size + rlen_index_y;
        rlen = rlen_table[lookup_idx];

        // Compute actual distance
        xdiff = x[inner_loop_idx] - x_tmp;
        ydiff = y[inner_loop_idx] - y_tmp;
        zdiff = z[inner_loop_idx] - z_tmp;
        vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
        inv_vmag = 1 / vmag;

        // Potential energy and force
        pe += fcon * (vmag - rlen) * (vmag - rlen);
        tmp = fcon - fcon * rlen * inv_vmag;
        fx_tmp += tmp * xdiff;
        fy_tmp += tmp * ydiff;
        fz_tmp += tmp * zdiff;
      }

      // Bottom stride
      for (dx = nx + 1; dx < dx_end; dx++) {
#pragma omp simd
        for (dy = dy_start; dy < dy_end; dy++) {
          inner_loop_idx = dx * n + dy;

          // Compute reference distance
          rlen_index_x = dx - nx + delta;
          rlen_index_y = dy - ny + delta;
          lookup_idx = rlen_index_x * size + rlen_index_y;
          rlen = rlen_table[lookup_idx];

          // Compute actual distance
          xdiff = x[inner_loop_idx] - x_tmp;
          ydiff = y[inner_loop_idx] - y_tmp;
          zdiff = z[inner_loop_idx] - z_tmp;
          vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
          inv_vmag = 1 / vmag;

          // Potential energy and force
          pe += fcon * (vmag - rlen) * (vmag - rlen);
          tmp = fcon - fcon * rlen * inv_vmag;
          fx_tmp += tmp * xdiff;
          fy_tmp += tmp * ydiff;
          fz_tmp += tmp * zdiff;
        }
      }

      // Update force
      fx[loop_idx] = fx_tmp;
      fy[loop_idx] = fy_tmp;
      fz[loop_idx] = fz_tmp;
    }
  }

  return 0.5 * pe;
}