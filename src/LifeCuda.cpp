#include <iostream>
#include <cstdint>
#include <algorithm>
#include <unistd.h>

#include "LifeCuda.h"
#include "LifeCuda.cuh"

LifeCuda::LifeCuda(int w, int h) : width(w), height(h), _grid(w * h, false)
{
    size_t mapSize = (width + 2) * (height+ 2) * sizeof(uint8_t); // +2 for the border cells
    cudaError_t err;

    err = cudaMalloc(&d_currentGrid, mapSize);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error allocating d_currentGrid: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_nextGrid, mapSize);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error allocating d_nextGrid: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_currentGrid);
        exit(EXIT_FAILURE);
    }

    cudaMemset(d_currentGrid, 0, mapSize);
    cudaMemset(d_nextGrid, 0, mapSize);

    _grid.resize((width + 2) * (height + 2), 0);
}

LifeCuda::~LifeCuda()
{
    if (d_currentGrid)
    {
        cudaFree(d_currentGrid);
    }
    if (d_nextGrid)
    {
        cudaFree(d_nextGrid);
    }
}

static int numberOfOnes(const std::vector<uint8_t> &grid) // TODO: delete later
{
    return std::count(grid.begin(), grid.end(), 1);
}

// Set initial state
void LifeCuda::setInitialState(std::vector<uint8_t> &initialState)
{
    size_t mapSize = (width + 2) * (height + 2) * sizeof(uint8_t);

    for (int i = 0; i < width + 2; i++)
    {
        for (int j = 0; j < height + 2; j++)
        {
            if (i == 0 || i == width + 1 || j == 0 || j == height + 1)
            {
                _grid[j * (width + 2) + i] = 0; // Set border cells to 0
            }
            else
            {
                _grid[j * (width + 2) + i] = initialState[(j - 1) * width + (i - 1)];
            }
        }
    }

    cudaMemcpy(d_currentGrid, _grid.data() , mapSize, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "Initial state set. Total active cells: " << numberOfOnes(initialState) << std::endl;
}


// Update method
void LifeCuda::update()
{
    int gridWidth = width + 2;
    int gridHeight = height + 2;
    dim3 blockSize(16, 16);
    dim3 gridSize((gridWidth + blockSize.x - 1) / blockSize.x, (gridHeight + blockSize.y - 1) / blockSize.y);

    if (kernelType == KERNEL_BASIC)
    {
        launchUpdateKernelBasic(d_currentGrid, d_nextGrid, gridWidth, gridHeight, gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    }
    else if (kernelType == KERNEL_SHARED_MEMORY)
    {
        launchUpdateKernelShared(d_currentGrid, d_nextGrid, gridWidth, gridHeight, gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    }
    else
    {
        std::cerr << "Unknown kernel type!" << std::endl;
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(_grid.data() , d_currentGrid, gridWidth * gridHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::swap(d_currentGrid, d_nextGrid);

    return;
}

uint8_t LifeCuda::getLifeform(int x, int y)
{
    return _grid[(y + 1) * (width + 2) + x + 1];
}

void LifeCuda::alterKernelType()
{
    if (kernelType == KERNEL_SHARED_MEMORY)
    {
        kernelType = KERNEL_BASIC;
        std::cout << "Kernel type changed to BASIC." << std::endl;
    }
    else
    {
        kernelType = KERNEL_SHARED_MEMORY;
        std::cout << "Kernel type changed to SHARED MEMORY." << std::endl;
    }
}