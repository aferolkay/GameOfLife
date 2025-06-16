#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

extern "C" __device__ int countNeighbors(uint8_t *grid, int x, int y, int width, int height);
extern "C" __global__ void updateKernel(uint8_t *currentGrid, uint8_t *nextGrid, int width, int height);
extern "C" __global__ void dummyKernel(uint8_t *grid, int width, int height);

extern "C" void launchDummyKernel(
    uint8_t* grid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY);