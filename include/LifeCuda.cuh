#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

extern "C" void launchDummyKernel(
    uint8_t* grid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY);

extern "C" void launchUpdateKernelBasic(
    uint8_t* currentGrid,
    uint8_t* nextGrid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY,
    float *timeTook);

extern "C" void launchUpdateKernelShared(
    uint8_t* currentGrid,
    uint8_t* nextGrid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY,
    float *timeTook);