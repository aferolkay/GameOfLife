#include "LifeCuda.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

extern "C" __device__ int countNeighbors(uint8_t *grid, int x, int y, int width, int height)
{
    /*
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
    */
   int count = 0;

	count += grid[(x-1) * (width) + (y)];
	count += grid[(x) * (width) + (y-1)];
	count += grid[(x - 1) * (width) + (y-1)];
	count += grid[(x + 1) * (width) + (y)];

	count += grid[(x) * (width) + (y+1)];
	count += grid[(x+1) * (width) + (y+1)];
	count += grid[(x+1) * (width) + (y-1)];
	count += grid[(x - 1) * (width) + (y+1)];

	return count;
}

extern "C" __global__ void updateKernel(uint8_t *currentGrid, uint8_t *nextGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height && x > 0 && y > 0)
    {
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

extern "C" __global__ void dummyKernel(uint8_t *grid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        grid[idx] = (idx % 5 < 3) ? 1 : 0; // Repeated pattern: 3 ones followed by 2 zeros
    }
}

extern "C" void launchDummyKernel(
    uint8_t* grid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY)
{
  dim3 gridSize (gridX, gridY);
  dim3 blockSize(blockX, blockY);
  dummyKernel<<<gridSize, blockSize>>>(grid, width, height);
  cudaDeviceSynchronize();
}

extern "C" void launchUpdateKernel(
    uint8_t* currentGrid,
    uint8_t* nextGrid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY)
{
  dim3 gridSize (gridX, gridY);
  dim3 blockSize(blockX, blockY);
  updateKernel<<<gridSize, blockSize>>>(currentGrid, nextGrid, width, height);
  cudaDeviceSynchronize();
}