#include "Life.cuh"

// __device__ function definition
__device__ int countNeighbors(uint8_t *grid, int x, int y, int width, int height)
{
    int count = 0;
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            count += grid[ny * width + nx];
        }
    }
    return count;
}

// __global__ function definition
__global__ void updateKernel(uint8_t *currentGrid, uint8_t *nextGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        int neighbors = countNeighbors(currentGrid, x, y, width, height);

        if (currentGrid[idx])
        {
            nextGrid[idx] = (neighbors == 2 || neighbors == 3);
        }
        else
        {
            nextGrid[idx] = (neighbors == 3);
        }
    }
}

// Constructor definition
LifeCuda::LifeCuda(int w, int h) : width(w), height(h), _grid(w * h, false)
{
    size_t gridSize = width * height * sizeof(uint8_t);
    cudaMalloc(&d_currentGrid, gridSize);
    cudaMalloc(&d_nextGrid, gridSize);
    cudaMemset(d_currentGrid, 0, gridSize);
    cudaMemset(d_nextGrid, 0, gridSize);
}

// Destructor definition
LifeCuda::~LifeCuda()
{
    cudaFree(d_currentGrid);
    cudaFree(d_nextGrid);
}

// Set initial state
void LifeCuda::setInitialState(std::vector<uint8_t> &initialState)
{
    cudaMemcpy(d_currentGrid, initialState.data() , width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

// Update method
void LifeCuda::update()
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    updateKernel<<<gridSize, blockSize>>>(d_currentGrid, d_nextGrid, width, height);
    cudaDeviceSynchronize();

    // Copy the updated grid back to the host
    cudaMemcpy(_grid.data(), d_nextGrid, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Swap the grids
    std::swap(d_currentGrid, d_nextGrid);
}

// Get lifeform state
uint8_t LifeCuda::getLifeform(int x, int y)
{
    return _grid[y * width + x];
}