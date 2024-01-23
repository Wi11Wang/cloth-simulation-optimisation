#ifndef CLOTH_CODE_H
#define CLOTH_CODE_H

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifdef __GNUC__
#  define UNUSED(x) UNUSED_ ## x __attribute__((__unused__))
#else
#  define UNUSED(x) UNUSED_ ## x
#endif

void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int delta, double UNUSED(grav), double sep,
                double rball, double offset, double UNUSED(dt), double **x,
                double **y, double **z, double **cpx, double **cpy,
                double **cpz, double **fx, double **fy, double **fz,
                double **vx, double **vy, double **vz, double **oldfx,
                double **oldfy, double **oldfz, double **rlen_table);

void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball, double zball, double dt,
              double * __restrict__ x, double * __restrict__ y, double * __restrict__ z,
              double ** __restrict__ fx, double ** __restrict__ fy, double ** __restrict__ fz,
              double * __restrict__ vx, double * __restrict__ vy, double * __restrict__ vz,
              double ** __restrict__ oldfx, double ** __restrict__ oldfy, double ** __restrict__ oldfz,
              double * __restrict__ rlen_table, double *pe, double *ke, double *te);

double eval_pef(int n, int delta, double mass, double grav, double sep, double fcon,
                double * __restrict__ x, double * __restrict__ y, double * __restrict__ z,
                double * __restrict__ fx, double * __restrict__ fy, double * __restrict__ fz,
                double * __restrict__ rlen_table);
#endif
