#include "./cloth_code_omp.h"
#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

using namespace std;

// Marcos for SIMD operations
#define DOUBLE_VEC __m256d
#define SET_PD(x) _mm256_set1_pd(x)
#define ADD_PD(a, b) _mm256_add_pd(a, b)
#define SUB_PD(a, b) _mm256_sub_pd(a, b)
#define MUL_PD(a, b) _mm256_mul_pd(a, b)
#define DIV_PD(a, b) _mm256_div_pd(a, b)
#define SQRT_PD(x) _mm256_sqrt_pd(x)
#define LOAD_PD(ptr) _mm256_load_pd(ptr)
#define STORE_PD(ptr, dst) _mm256_store_pd(ptr, dst)
#define HSUM_PD(vec) hsum_avx(vec)

#define SCHEDULE_TYPE static, 16

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

/**
 * Horizontally adds all the values contained in a 256-bit register of [4 x double].
 * e.g.
 * __m256d vector = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
 * hsum_avx(vector) returns 10.0
 *
 * @param vector a 256-bit register
 * @return the summed value in a register
 */
double hsum_avx(DOUBLE_VEC vector) {
  __m256d sum = _mm256_hadd_pd(vector, vector);
  __m128d high = _mm256_extractf128_pd(sum, 1);
  __m128d low = _mm256_castpd256_pd128(sum);
  __m128d finalSum = _mm_add_pd(low, high);
  return _mm_cvtsd_f64(finalSum);
}

void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball, double zball, double dt,
              double *__restrict__ x, double *__restrict__ y, double *__restrict__ z,
              double **__restrict__ fx, double **__restrict__ fy, double **__restrict__ fz,
              double *__restrict__ vx, double *__restrict__ vy, double *__restrict__ vz,
              double **__restrict__ oldfx, double **__restrict__ oldfy, double **__restrict__ oldfz,
              double *__restrict__ rlen_table, double *pe, double *ke, double *te, int nthreads) {
  int i, j, loop_idx;
  double xdiff, ydiff, zdiff, vmag, inv_vmag, damp, proj_scalar;
  double xdiff_unit, ydiff_unit, zdiff_unit;
  double x_tmp, y_tmp, z_tmp;
  double *x_vel, *y_vel, *z_vel;
  double half_dt_div_mass = dt * 0.5 / mass;

  omp_set_num_threads(nthreads);

#pragma omp parallel for schedule(SCHEDULE_TYPE) default(none) private(i, j, loop_idx, x_tmp, y_tmp, z_tmp, xdiff, ydiff, zdiff, vmag, inv_vmag, xdiff_unit, ydiff_unit, zdiff_unit, x_vel, y_vel, z_vel, proj_scalar) shared(n, x, y, z, vx, vy, vz, fx, fy, fz, dt, half_dt_div_mass, xball, yball, zball, rball)
  for (i = 0; i < n; i++) {
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
        inv_vmag = 1 / (vmag);
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
  std::swap(*fx, *oldfx);
  std::swap(*fy, *oldfy);
  std::swap(*fz, *oldfz);

  // Add a damping factor to eventually set velocity to zero
  damp = 0.995;
  *ke = 0.0;
  DOUBLE_VEC damp_vec = SET_PD(damp);
  DOUBLE_VEC ke_vec = SET_PD(0.0);
  DOUBLE_VEC half_dt_div_mass_vec = SET_PD(dt * 0.5 / mass);

  double local_ke = 0.0;
#pragma omp parallel for schedule(SCHEDULE_TYPE) reduction(+:local_ke) default(none) private(j, loop_idx, ke_vec) shared(vx, vy, vz, fx, fy, fz, oldfx, oldfy, oldfz, n, half_dt_div_mass, half_dt_div_mass_vec, damp, damp_vec)
  for (i = 0; i < n; i++) {
    int simd_bound = n - (n % 4);
    for (j = 0; j < simd_bound; j += 4) {
      loop_idx = i * n + j;

      DOUBLE_VEC vx_vec = LOAD_PD(&vx[loop_idx]);
      DOUBLE_VEC vy_vec = LOAD_PD(&vy[loop_idx]);
      DOUBLE_VEC vz_vec = LOAD_PD(&vz[loop_idx]);
      DOUBLE_VEC fx_vec = LOAD_PD(&(*fx)[loop_idx]);
      DOUBLE_VEC fy_vec = LOAD_PD(&(*fy)[loop_idx]);
      DOUBLE_VEC fz_vec = LOAD_PD(&(*fz)[loop_idx]);
      DOUBLE_VEC oldfx_vec = LOAD_PD(&(*oldfx)[loop_idx]);
      DOUBLE_VEC oldfy_vec = LOAD_PD(&(*oldfy)[loop_idx]);
      DOUBLE_VEC oldfz_vec = LOAD_PD(&(*oldfz)[loop_idx]);

      DOUBLE_VEC tmp_x = ADD_PD(fx_vec, oldfx_vec);
      DOUBLE_VEC tmp_y = ADD_PD(fy_vec, oldfy_vec);
      DOUBLE_VEC tmp_z = ADD_PD(fz_vec, oldfz_vec);

      tmp_x = MUL_PD(tmp_x, half_dt_div_mass_vec);
      tmp_y = MUL_PD(tmp_y, half_dt_div_mass_vec);
      tmp_z = MUL_PD(tmp_z, half_dt_div_mass_vec);

      vx_vec = MUL_PD(ADD_PD(vx_vec, tmp_x), damp_vec);
      vy_vec = MUL_PD(ADD_PD(vy_vec, tmp_y), damp_vec);
      vz_vec = MUL_PD(ADD_PD(vz_vec, tmp_z), damp_vec);

      // Update velocity
      STORE_PD(&vx[loop_idx], vx_vec);
      STORE_PD(&vy[loop_idx], vy_vec);
      STORE_PD(&vz[loop_idx], vz_vec);

      // Update kinetic energy
      ke_vec = ADD_PD(ADD_PD(MUL_PD(vx_vec, vx_vec), MUL_PD(vy_vec, vy_vec)), MUL_PD(vz_vec, vz_vec));
      local_ke += HSUM_PD(ke_vec);
    }
    for (j = simd_bound; j < n; j++) {
      loop_idx = i * n + j;
      // Update velocity
      vx[loop_idx] = (vx[loop_idx] +
                      ((*fx)[loop_idx] + (*oldfx)[loop_idx]) * half_dt_div_mass) *
                     damp;
      vy[loop_idx] = (vy[loop_idx] +
                      ((*fy)[loop_idx] + (*oldfy)[loop_idx]) * half_dt_div_mass) *
                     damp;
      vz[loop_idx] = (vz[loop_idx] +
                      ((*fz)[loop_idx] + (*oldfz)[loop_idx]) * half_dt_div_mass) *
                     damp;
      // Update kinetic energy
      local_ke += vx[loop_idx] * vx[loop_idx] + vy[loop_idx] * vy[loop_idx] +
                  vz[loop_idx] * vz[loop_idx];
    }
  }
  *ke = 0.5 * local_ke;
  *te = *pe + *ke;
}

inline __attribute__((always_inline)) void
calc_interaction_avx(const int dx, const int dy, const int nx, int ny, const int n,
                     const int size, const int delta, const double fcon,
                     const double *__restrict__ x, const double *__restrict__ y, const double *__restrict__ z,
                     const DOUBLE_VEC x_tmp_vec, const DOUBLE_VEC y_tmp_vec, const DOUBLE_VEC z_tmp_vec,
                     DOUBLE_VEC *__restrict__ fx_tmp_vec, DOUBLE_VEC *__restrict__ fy_tmp_vec,
                     DOUBLE_VEC *__restrict__ fz_tmp_vec, DOUBLE_VEC *__restrict__ pe_vec,
                     const double *__restrict__ rlen_table) {
  int inner_loop_idx = dx * n + dy;
  int rlen_index_x = dx - nx + delta;
  int rlen_index_y = dy - ny + delta;
  int lookup_idx = rlen_index_x * size + rlen_index_y;

  DOUBLE_VEC rlen_vec = LOAD_PD(&rlen_table[lookup_idx]);
  DOUBLE_VEC x_inner_vec = LOAD_PD(&x[inner_loop_idx]);
  DOUBLE_VEC y_inner_vec = LOAD_PD(&y[inner_loop_idx]);
  DOUBLE_VEC z_inner_vec = LOAD_PD(&z[inner_loop_idx]);

  DOUBLE_VEC xdiff_vec = SUB_PD(x_inner_vec, x_tmp_vec);
  DOUBLE_VEC ydiff_vec = SUB_PD(y_inner_vec, y_tmp_vec);
  DOUBLE_VEC zdiff_vec = SUB_PD(z_inner_vec, z_tmp_vec);

  DOUBLE_VEC vmag_vec = SQRT_PD(
      ADD_PD(ADD_PD(MUL_PD(xdiff_vec, xdiff_vec), MUL_PD(ydiff_vec, ydiff_vec)), MUL_PD(zdiff_vec, zdiff_vec)));
  DOUBLE_VEC inv_vmag_vec = DIV_PD(SET_PD(1.0), vmag_vec);

  DOUBLE_VEC vmag_sub_rlen_vec = SUB_PD(vmag_vec, rlen_vec);

  DOUBLE_VEC fcon_vec = SET_PD(fcon);
  DOUBLE_VEC tmp = SUB_PD(fcon_vec, MUL_PD(fcon_vec, MUL_PD(rlen_vec, inv_vmag_vec)));
  *fx_tmp_vec = ADD_PD(*fx_tmp_vec, MUL_PD(tmp, xdiff_vec));
  *fy_tmp_vec = ADD_PD(*fy_tmp_vec, MUL_PD(tmp, ydiff_vec));
  *fz_tmp_vec = ADD_PD(*fz_tmp_vec, MUL_PD(tmp, zdiff_vec));
  *pe_vec = ADD_PD(*pe_vec, (MUL_PD(MUL_PD(vmag_sub_rlen_vec, vmag_sub_rlen_vec), SET_PD(fcon))));
}

inline __attribute__((always_inline)) double
calc_interaction(const int dx, const int dy, const int nx, int ny, const int n,
                 const int size, const int delta, const double fcon,
                 const double *__restrict__ x, const double *__restrict__ y, const double *__restrict__ z,
                 const double x_tmp, const double y_tmp, const double z_tmp,
                 double *__restrict__ fx_tmp, double *__restrict__ fy_tmp, double *__restrict__ fz_tmp,
                 const double *__restrict__ rlen_table) {
  double pe = 0.0, rlen, xdiff, ydiff, zdiff, vmag, inv_vmag, tmp;
  int inner_loop_idx = dx * n + dy;

  // Compute reference distance
  int rlen_index_x = dx - nx + delta;
  int rlen_index_y = dy - ny + delta;
  int lookup_idx = rlen_index_x * size + rlen_index_y;
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
  *fx_tmp += tmp * xdiff;
  *fy_tmp += tmp * ydiff;
  *fz_tmp += tmp * zdiff;

  return pe;
}

double eval_pef(int n, int delta, double mass, double grav, double sep, double fcon,
                double *__restrict__ x, double *__restrict__ y, double *__restrict__ z,
                double *__restrict__ fx, double *__restrict__ fy, double *__restrict__ fz,
                double *__restrict__ rlen_table) {
  double pe = 0.0, fx_tmp, fy_tmp, fz_tmp;
  int nx, ny, dx, dy, loop_idx, simd_bound;
  int dx_start, dx_end, dy_start, dy_end;
  double x_tmp, y_tmp, z_tmp;

  int neighbour_size = (2 * delta + 1);
  DOUBLE_VEC sep_vec = SET_PD(sep);
#pragma omp parallel for schedule(SCHEDULE_TYPE) default(none) private(dx, dy, simd_bound) shared(delta, neighbour_size, sep, sep_vec, rlen_table, n)
  for (dx = -delta; dx <= delta; dx++) {
    DOUBLE_VEC dx_vec = SET_PD((double) dx * dx);
    simd_bound = delta + 1 - (delta * 2 + 1) % 4;
    for (dy = -delta; dy < simd_bound; dy += 4) {
      DOUBLE_VEC dy_vec = _mm256_set_pd((double) (dy + 3) * (dy + 3),
                                        (double) (dy + 2) * (dy + 2),
                                        (double) (dy + 1) * (dy + 1),
                                        (double) dy * dy);
      DOUBLE_VEC rlen_vec = MUL_PD(SQRT_PD(ADD_PD(dx_vec, dy_vec)), sep_vec);
      STORE_PD(&rlen_table[(dx + delta) * neighbour_size + (dy + delta)], rlen_vec);
    }
    for (dy = simd_bound; dy <= delta; dy++) {
      rlen_table[(dx + delta) * neighbour_size + (dy + delta)] = sqrt(dx * dx + dy * dy) * sep;
    }
  }

  // Loop over particles
#pragma omp parallel for schedule(SCHEDULE_TYPE) reduction(+:pe) default(none) private(nx, ny, fx_tmp, fy_tmp, fz_tmp, loop_idx, dx_start, dx_end, dy_start, dy_end, x_tmp, y_tmp, z_tmp, dx, dy, simd_bound) shared(neighbour_size, fcon, mass, n, grav, delta, x, y, z, fx, fy, fz, rlen_table)
  for (nx = 0; nx < n; nx++) {
    for (ny = 0; ny < n; ny++) {
      fx_tmp = 0.0;
      fy_tmp = 0.0;
      fz_tmp = -mass * grav;
      loop_idx = nx * n + ny;
      DOUBLE_VEC fx_tmp_vec = SET_PD(0.0);
      DOUBLE_VEC fy_tmp_vec = SET_PD(0.0);
      DOUBLE_VEC fz_tmp_vec = SET_PD(0.0);

      x_tmp = x[loop_idx];
      y_tmp = y[loop_idx];
      z_tmp = z[loop_idx];
      DOUBLE_VEC x_tmp_vec = SET_PD(x_tmp);
      DOUBLE_VEC y_tmp_vec = SET_PD(y_tmp);
      DOUBLE_VEC z_tmp_vec = SET_PD(z_tmp);

      DOUBLE_VEC pe_vec = SET_PD(0.0);

      dx_start = MAX(nx - delta, 0), dx_end = MIN(nx + delta + 1, n);
      dy_start = MAX(ny - delta, 0), dy_end = MIN(ny + delta + 1, n);
      // Top stride
      for (dx = dx_start; dx < nx; dx++) {
        simd_bound = dy_end - ((dy_end - dy_start) % 4);
        for (dy = dy_start; dy < simd_bound; dy += 4) {
          calc_interaction_avx(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp_vec, y_tmp_vec,
                               z_tmp_vec, &fx_tmp_vec, &fy_tmp_vec, &fz_tmp_vec, &pe_vec, rlen_table);
        }
        for (dy = simd_bound; dy < dy_end; dy++) {
          pe += calc_interaction(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp, y_tmp, z_tmp, &fx_tmp,
                                 &fy_tmp, &fz_tmp, rlen_table);
        }
      }

      dx = nx;
      // Left middle stride
      simd_bound = ny - ((ny - dy_start) % 4);
      for (dy = dy_start; dy < simd_bound; dy += 4) {
        calc_interaction_avx(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp_vec, y_tmp_vec,
                             z_tmp_vec, &fx_tmp_vec, &fy_tmp_vec, &fz_tmp_vec, &pe_vec, rlen_table);
      }
      for (dy = simd_bound; dy < ny; dy++) {
        pe += calc_interaction(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp, y_tmp, z_tmp, &fx_tmp,
                               &fy_tmp, &fz_tmp, rlen_table);
      }

      // Right middle stride
      simd_bound = dy_end - ((dy_end - ny - 1) % 4);
      for (dy = ny + 1; dy < simd_bound; dy += 4) {
        calc_interaction_avx(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp_vec, y_tmp_vec,
                             z_tmp_vec, &fx_tmp_vec, &fy_tmp_vec, &fz_tmp_vec, &pe_vec, rlen_table);
      }
      for (dy = simd_bound; dy < dy_end; dy++) {
        pe += calc_interaction(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp, y_tmp, z_tmp, &fx_tmp,
                               &fy_tmp, &fz_tmp, rlen_table);
      }

      // Bottom stride
      for (dx = nx + 1; dx < dx_end; dx++) {
        simd_bound = dy_end - ((dy_end - dy_start) % 4);
        for (dy = dy_start; dy < simd_bound; dy += 4) {
          calc_interaction_avx(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp_vec, y_tmp_vec,
                               z_tmp_vec, &fx_tmp_vec, &fy_tmp_vec, &fz_tmp_vec, &pe_vec, rlen_table);
        }
        for (dy = simd_bound; dy < dy_end; dy++) {
          pe += calc_interaction(dx, dy, nx, ny, n, neighbour_size, delta, fcon, x, y, z, x_tmp, y_tmp, z_tmp, &fx_tmp,
                                 &fy_tmp, &fz_tmp, rlen_table);
        }
      }

      // Update force
      pe += HSUM_PD(pe_vec);
      fx[loop_idx] = fx_tmp + HSUM_PD(fx_tmp_vec);
      fy[loop_idx] = fy_tmp + HSUM_PD(fy_tmp_vec);
      fz[loop_idx] = fz_tmp + HSUM_PD(fz_tmp_vec);
    }
  }

  return 0.5 * pe;
}
