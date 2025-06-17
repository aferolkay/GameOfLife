#pragma once

#include <vector>
#include <cstdint>

class LifeCuda {

public:
    enum CudaKernelType
    {
        KERNEL_BASIC = 0,
        KERNEL_SHARED_MEMORY = 1,

        KERNEL_COUNT = 2
    };

private:
    CudaKernelType kernelType = KERNEL_BASIC;
    int width, height;
    uint8_t *d_currentGrid, *d_nextGrid;
    std::vector<uint8_t> _grid;

public:
    LifeCuda(int w, int h);
    ~LifeCuda();

    void setInitialState(std::vector<uint8_t> &initialState);

    void update();

    uint8_t getLifeform(int x, int y);

    void alterKernelType ();
};
