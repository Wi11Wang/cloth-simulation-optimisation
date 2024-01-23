#ifndef CLOTH_PARAM_H
#define CLOTH_PARAM_H
// Default values
int n = 20, delta = 2, maxiter = 400;
double sep = 1.0, mass = 1.0, fcon = 10;
double grav = 0.981, dt = 0.05;
double xball = 0.0, yball = 0.0, zball = 0.0, rball = 3.0, offset = 0.0;

// UPDATE SUPPORT FOR SPECIFYING NUMBER OF THREADS
int nthreads = 8;

// Pointers to cloth data structures
double *x, *y, *z, *fx, *fy, *fz, *vx, *vy, *vz, *oldfx, *oldfy, *oldfz, *rlen_table;

// OpenGL related stuff
double *cpx, *cpy, *cpz;
int update = 3, rendermode = 1, paused = 0, loop = 0;
double reset_time = 100;

#endif
