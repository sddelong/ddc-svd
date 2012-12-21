#ifdef __APPLE__

#include <sys/time.h>

typedef struct timeval timestamp_type;

static void get_timestamp(timestamp_type *t)
{
  gettimeofday(t, NULL);
}

double timestamp_diff_in_seconds(timestamp_type x, timestamp_type y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x.tv_usec < y.tv_usec) {
    int nsec = (y.tv_usec - x.tv_usec) / 1000000 + 1;
    y.tv_usec -= 1000000 * nsec;
    y.tv_sec += nsec;
  }
  if (x.tv_usec - y.tv_usec > 1000000) {
    int nsec = (x.tv_usec - y.tv_usec) / 1000000;
    y.tv_usec += 1000000 * nsec;
    y.tv_sec -= nsec;
  }

  return x.tv_sec + (x.tv_usec - y.tv_usec)*1e-6;
}

#else

#include <time.h>

typedef struct timespec timestamp_type;

static void get_timestamp(timestamp_type *t)
{
  clock_gettime(CLOCK_REALTIME, t);
}

static double timestamp_diff_in_seconds(timestamp_type start, timestamp_type end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp.tv_sec + 1e-9*temp.tv_nsec;
}

#endif
