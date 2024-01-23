//
// Created by Will on 28/9/2023.
// This custom profiler code is used to collect performance data
// To use this, place start_profiling() and finish_profiling() in the code
//

#include "./myprofiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef PAPI
#include <papi.h>
#endif

static struct timeval wall_start, wall_end;
#ifdef PAPI
static const int NUM_EVENTS = 6;
static int EVENTS[NUM_EVENTS] = {
    PAPI_DP_OPS,  // 0. Floating point operations
    PAPI_L1_DCM,  // 1. L1 data cache miss
    PAPI_L2_DCM,  // 2. L2 data cache miss
    PAPI_TOT_INS, // 3. Total Instructions
    PAPI_BR_MSP,  // 4. Branch mispredictions
    PAPI_VEC_DP,  // 5. DP SIMD operations
};
static long long VALUES[NUM_EVENTS];
int EventSet = PAPI_NULL;
int retval = 0;
#endif

void start_profiling() {
  gettimeofday(&wall_start, nullptr);
#ifdef PAPI
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
  fprintf(stderr,"PAPI Library initialization error! %d\n",  __LINE__);
  exit(1);
  }

  if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
    fprintf(stderr, "PAPI failed to create event set\n");
    exit(1);
  }

  PAPI_add_events(EventSet, EVENTS, NUM_EVENTS);

  /* Start counting events */
  if ((retval = PAPI_start(EventSet)) != PAPI_OK) {
    fprintf(stderr,"PAPI Start counter error! %d, %d\n",  retval, __LINE__);
    exit(1);
  }
#endif
}

void finish_profiling(const char *program) {
  long seconds, useconds;
  long elapsed_time;
  gettimeofday(&wall_end, nullptr);

  seconds = wall_end.tv_sec - wall_start.tv_sec;
  useconds = wall_end.tv_usec - wall_start.tv_usec;
  elapsed_time = seconds * 1000000L + useconds;

  printf("\n--------------------\n");
  printf("%s: wall time (us): %ld\n", program, elapsed_time);

#ifdef PAPI
  /* Stop counting events */
  if ((retval = PAPI_stop(EventSet, VALUES)) != PAPI_OK) {
    fprintf(stderr,"PAPI stop counters error! %d, %d\n", retval, __LINE__);
    exit(1);
  }

  printf("%s: PAPI_DP_OPS:    %lld\n", program, VALUES[0]);
  printf("%s: MFLOPS:         %lld\n", program, VALUES[0] / elapsed_time);
  printf("%s: PAPI_L1_DCM:    %lld\n", program, VALUES[1]);
  printf("%s: PAPI_L2_DCM:    %lld\n", program, VALUES[2]);
  printf("%s: PAPI_TOT_INS:   %lld\n", program, VALUES[3]);
  printf("%s: PAPI_BR_MSP:    %lld\n", program, VALUES[4]);
  printf("%s: PAPI_VEC_DP:    %lld\n", program, VALUES[5]);
#endif
  printf("--------------------\n");
}
