#include <iostream>
#include <cstdint>
#include <algorithm>
#include <unistd.h>

#include "LifeCuda.h"
#include "LifeCuda.cuh"

LifeCuda::LifeCuda(int w, int h) : width(w), height(h), _grid(w * h, false)
{
    cudaSetDevice(0);
    cudaDeviceProp prop;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Available CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name
                  << " | CC " << deviceProp.major << "." << deviceProp.minor
                  << " | Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB"
                  << std::endl;
    }

    size_t mapSize = (width + 2) * (height+ 2) * sizeof(uint8_t);
    cudaError_t err;

    err = cudaMalloc(&d_currentGrid, mapSize);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error allocating d_currentGrid: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        std::cout << "Allocated d_currentGrid of size: " << mapSize << " bytes." << std::endl;
    }

    err = cudaMalloc(&d_nextGrid, mapSize);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error allocating d_nextGrid: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_currentGrid);
        exit(EXIT_FAILURE);
    }
    else
    {
        std::cout << "Allocated d_nextGrid of size: " << mapSize << " bytes." << std::endl;
    }

    cudaMemset(d_currentGrid, 0, mapSize);
    cudaMemset(d_nextGrid, 0, mapSize);

    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max grid dimensions: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;

    _grid.resize((width + 2) * (height + 2), 0);
}

LifeCuda::~LifeCuda()
{
    if (d_currentGrid)
        cudaFree(d_currentGrid);
    if (d_nextGrid)
        cudaFree(d_nextGrid);
}

static int numberOfOnes(const std::vector<uint8_t> &grid)
{
    return std::count(grid.begin(), grid.end(), 1);
}

// Set initial state
void LifeCuda::setInitialState(std::vector<uint8_t> &initialState)
{
    size_t mapSize = (width + 2) * (height + 2) * sizeof(uint8_t);

    cudaMemcpy(d_currentGrid, initialState.data() , mapSize, cudaMemcpyHostToDevice);
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
    uint8_t *tempGrid = nullptr;

    sleep (1);

    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;

    tempGrid = (uint8_t *)malloc(gridWidth * gridHeight * sizeof(uint8_t));
    if (!tempGrid)
    {
        std::cerr << "Failed to allocate memory for tempGrid." << std::endl;
        return;
    }

    memset(tempGrid, 0, gridWidth * gridHeight * sizeof(uint8_t));

    //updateKernel<<<gridSize, blockSize>>>(d_currentGrid, d_nextGrid, gridWidth, gridHeight);
    if (d_currentGrid == nullptr || d_nextGrid == nullptr)
    {
        std::cerr << "Device grids are not allocated." << std::endl;
        free(tempGrid);
        return;
    }

    std::cout << "Launching kernel with grid size: " << gridSize.x << "x" << gridSize.y << " and block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    std::cout << "Grid width: " << gridWidth << ", Grid height: " << gridHeight << std::endl;

    // dummyKernel<<<gridSize, blockSize>>>(d_currentGrid, gridWidth, gridHeight);
    launchDummyKernel(
        d_currentGrid, gridWidth, gridHeight,
        gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        free(tempGrid);
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaMemcpy(tempGrid, d_currentGrid, gridWidth * gridHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    std::copy(tempGrid, tempGrid + (gridWidth * gridHeight), _grid.begin());

    free(tempGrid);

    return;

    /********************************************************************* */
    /********************************************************************* */

    // Copy the updated grid back to the host
    cudaMemcpy(_grid.data(), d_nextGrid, gridWidth * gridHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    std::cout << "Number of live cells: " << numberOfOnes(_grid) << std::endl;

    // Swap the grids
    std::swap(d_currentGrid, d_nextGrid);
    std::cout << "Grid updated." << std::endl;

    sleep(1); // Sleep for 1 second to simulate time delay
}

uint8_t LifeCuda::getLifeform(int x, int y)
{
    return _grid[y * (width + 2) + x];
}