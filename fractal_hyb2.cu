#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#define THREADSPERBLOCK 256

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

static __global__ void FractalKernel(int width, int from, int to, int maxdepth, double dx, double dy, unsigned char cnt[])
{
  // kernel code goes in here

  /* compute thread index */
  int index = threadIdx.x + blockIdx.x * blockDim.x + (width * from);
  double cx, cy, x, y, x2, y2;
  int row, col, depth;

  /* compute fractal */
  if(index < (width * to))
  {
    //calculate row and col
    col = index % width;
    row = index / width;

    cy = yMin + row * dy;
    cx = xMin + col * dx;
    x = -cx;
    y = -cy;
    depth = maxdepth;
    do
    {
      x2 = x * x;
      y2 = y * y;
      y = 2 * x * y - cy;
      x = x2 - y2 - cx;
      depth--;
    } while ((depth > 0) && ((x2 + y2) <= 5.0));
    cnt[row * width + col] = depth & 255;
  }
}

extern "C" unsigned char *GPU_Init(int size)
{
  /* device copies */
  unsigned char *d_cnt;

  // allocate array on GPU and return pointer to it
  cudaMalloc((void **) &d_cnt, size);

  return d_cnt;
}

extern "C" void GPU_Exec(int width, int from, int to, int maxdepth, double dx, double dy, unsigned char *cnt_d)
{
  // call the kernel (and do nothing else)
  FractalKernel <<< (width * (to - from) + THREADSPERBLOCK - 1) / THREADSPERBLOCK , THREADSPERBLOCK >>> (width, from, to, maxdepth, dx, dy, cnt_d);
}

extern "C" void GPU_Fini(unsigned char *cnt, unsigned char *cnt_d, int size)
{
  // copy the pixel data to the CPU and deallocate the GPU array
  cudaMemcpy(cnt, cnt_d, size, cudaMemcpyDeviceToHost);
  cudaFree(cnt_d);
}
