fractal_hyb2.c

Type
C
Size
5 KB (5,293 bytes)
Storage used
5 KB (5,293 bytes)
Location
Final.2
Owner
me
Modified
May 6, 2015 by me
Opened
7:01 PM by me
Created
May 6, 2015 with Google Chrome
Add a description
Viewers can download
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

unsigned char *GPU_Init(int size);
void GPU_Exec(int width, int from, int to, int maxdepth, double dx, double dy, unsigned char *cnt_d);
void GPU_Fini(unsigned char *cnt, unsigned char *cnt_d, int size);

static void WriteBMP(int x, int y, unsigned char *bmp, char * name)
{
  const unsigned char bmphdr[54] = {66, 77, 255, 255, 255, 255, 0, 0, 0, 0, 54, 4, 0, 0, 40, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 8, 0, 0, 0, 0, 0, 255, 255, 255, 255, 196, 14, 0, 0, 196, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  unsigned char hdr[1078];
  int i, j, c, xcorr, diff;
  FILE *f;

  xcorr = (x+3) >> 2 << 2;  // BMPs have to be a multiple of 4 pixels wide.
  diff = xcorr - x;

  for (i = 0; i < 54; i++) hdr[i] = bmphdr[i];
  *((int*)(&hdr[18])) = xcorr;
  *((int*)(&hdr[22])) = y;
  *((int*)(&hdr[34])) = xcorr*y;
  *((int*)(&hdr[2])) = xcorr*y + 1078;
  for (i = 0; i < 256; i++) {
    j = i*4 + 54;
    hdr[j+0] = i;  // blue
    hdr[j+1] = i;  // green
    hdr[j+2] = i;  // red
    hdr[j+3] = 0;  // dummy
  }

  f = fopen(name, "wb");
  assert(f != NULL);
  c = fwrite(hdr, 1, 1078, f);
  assert(c == 1078);
  if (diff == 0) {
    c = fwrite(bmp, 1, x*y, f);
    assert(c == x*y);
  } else {
    *((int*)(&hdr[0])) = 0;  // need up to three zero bytes
    for (j = 0; j < y; j++) {
      c = fwrite(&bmp[j * x], 1, x, f);
      assert(c == x);
      c = fwrite(hdr, 1, diff, f);
      assert(c == diff);
    }
  }
  fclose(f);
}

int main(int argc, char *argv[])
{
  int width, /*height,*/ maxdepth, percent;
  double dx, dy;
  unsigned char *cnt, *cnt_d, *cntLocal;
  struct timeval start, end;
  int idx, row, col, depth;
  double cx, cy, x, y, x2, y2;

  /* initialize MPI threads */
  int comm_sz;
  int my_rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  printf("Fractal v1.2 [Hybrid]\n");

  /* check command line */
  if (argc != 4) {fprintf(stderr, "usage: %s edge_length max_depth GPU_percentage\n", argv[0]); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}
  maxdepth = atoi(argv[2]);
  if (maxdepth < 10) {fprintf(stderr, "max_depth must be at least 10\n"); exit(-1);}
  percent = atoi(argv[3]);
  if ((percent < 0) || (100 < percent)) {fprintf(stderr, "GPU_percentage out of range\n"); exit(-1);}
  // height = percent * width / 100;

  if(my_rank == 0)
  {
    printf("computing %d by %d fractal with a maximum depth of %d and %d%% on the GPU\n", width, width, maxdepth, percent);
  }

  /* calculate from, to, and cut */
  assert((width % comm_sz) == 0); // check for integer multiple
  int from = my_rank * width / comm_sz;
  int to = (my_rank + 1) * width / comm_sz;
  int cut = from + percent * (to - from) / 100;

  /* allocate arrays */
  if(my_rank == 0)
  {
    cnt = (unsigned char *)malloc(width * width * sizeof(unsigned char));
    if (cnt == NULL) {fprintf(stderr, "could not allocate memory\n"); exit(-1);}
  }
  /* allocate local arrays */
  cntLocal = (unsigned char *)malloc(width * (to - from) * sizeof(unsigned char));
  if (cntLocal == NULL) {fprintf(stderr, "could not allocate memory\n"); exit(-1);}

  //cnt_d = GPU_Init(width * (to - from) * sizeof(unsigned char));

  /* start time */
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);

  /* compute dx and dy */
  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  // the following call should compute rows from through cut - 1
  //GPU_Exec(width, from, cut, maxdepth, dx, dy, cnt_d);

  // the following code should compute rows cut through width - 1
  // insert OpenMP parallelized for loop with 16 threads, default(none), and a dynamic schedule with a chunk size of 1
# pragma omp parallel for num_threads(16) default(none) \
    private(col, row, depth, x, y, cx, cy, x2, y2) \
    shared(cnt, dy, dx, maxdepth, width, /*height,*/ cut, to, cntLocal) \
    schedule(dynamic, 1)
  for (row = cut; row < to; row++) {
    cy = yMin + row * dy;
    for (col = 0; col < width; col++) {
      cx = xMin + col * dx;
      x = -cx;
      y = -cy;
      depth = maxdepth;
      do {
        x2 = x * x;
        y2 = y * y;
        y = 2 * x * y - cy;
        x = x2 - y2 - cx;
        depth--;
      } while ((depth > 0) && ((x2 + y2) <= 5.0));
      cntLocal[(row - cut) * width + col] = depth & 255;
    }
  }

  // the following call should copy the GPU's result into the beginning of the CPU's cnt array
  //GPU_Fini(cntLocal, cnt_d, width * (cut - from) * sizeof(unsigned char));

  /* reduce */
  MPI_Gather(cntLocal, width/comm_sz, MPI_UNSIGNED_CHAR, cnt, width/comm_sz, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  /* end time */
  gettimeofday(&end, NULL);
  if(my_rank == 0)
  {
    printf("compute time: %.4f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
  }

  /* verify result by writing it to a file */
  if (my_rank && width <= 1024) {
    WriteBMP(width, width, cnt, "fractal.bmp");
  }

  if(my_rank == 0)
  {
    free(cnt);
  }
  MPI_Finalize();
  return 0;
}
