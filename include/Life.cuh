#pragma once
#include <iostream>

using std::malloc;
using std::cout;
using std::endl;

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cstdint>

__device__ int countNeighbors(uint8_t *grid, int x, int y, int width, int height);
__global__ void updateKernel(uint8_t *currentGrid, uint8_t *nextGrid, int width, int height);

class LifeCuda {
private:
    int width, height;
    uint8_t *d_currentGrid, *d_nextGrid;
    std::vector<uint8_t> _grid;

public:
    LifeCuda(int w, int h);
    ~LifeCuda();

    void setInitialState(std::vector<uint8_t> &initialState);

    void update();

    uint8_t getLifeform(int x, int y);

};
