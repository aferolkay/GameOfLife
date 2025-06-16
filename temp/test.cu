#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr,                                                        \
              "CUDA error %s:%d: '%s' (%d)\n",                              \
              __FILE__, __LINE__, cudaGetErrorString(err), err);            \
      exit(err);                                                             \
    }                                                                        \
  } while (0)

__global__ void dummyKernel(uint8_t *grid, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y*width + x;
    grid[idx] = (idx % 5 < 3) ? 1 : 0;
  }
}

int main(){
  int width = 128, height = 64;
  size_t sz = width*height*sizeof(uint8_t);
  uint8_t *d_grid;
  CUDA_CHECK(cudaMalloc(&d_grid, sz));

  dim3 blk(16,16), grd((width+15)/16,(height+15)/16);
  dummyKernel<<<grd,blk>>>(d_grid,width,height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_grid));

  puts("Success!");
  return 0;
}
