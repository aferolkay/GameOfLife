#include "LifeCuda.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* dummy kernel used for testing purposes */
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

/* most basic form of cuda kernel to used, no performance considerations were made */
extern "C" __device__ int countNeighbors(uint8_t *grid, int x, int y, int width, int height)
{
   int count = 0;

	count += grid[((y - 1) * width) + x]; // up
	count += grid[((y - 1) * width) + (x + 1)]; // up right
    count += grid[(y * width) + (x + 1)]; // right
    count += grid[((y + 1) * width) + (x + 1)]; // down right
    count += grid[((y + 1) * width) + x]; // down
    count += grid[((y + 1) * width) + (x - 1)]; // down left
    count += grid[(y * width) + (x - 1)]; // left
    count += grid[((y - 1) * width) + (x - 1)]; // up left

	return count;
}

extern "C" __global__ void updateKernelBasic(uint8_t *currentGrid, uint8_t *nextGrid, int width, int height)
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

extern "C" __global__ void updateKernelShared(uint8_t *currentGrid, uint8_t *nextGrid, int width, int height)
{
    extern __shared__ uint8_t sharedGrid[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    int localX = threadIdx.x + 1; // +1 for halo
    int localY = threadIdx.y + 1; // +1 for halo
    int sharedWidth = blockDim.x + 2; // +2 for halo

    // Load the current cell and its neighbors into shared memory
    if (x < width && y < height)
    {
        sharedGrid[localY * sharedWidth + localX] = currentGrid[idx];

        // Load halo cells
        if (threadIdx.x == 0 && x > 0)
            sharedGrid[localY * sharedWidth] = currentGrid[idx - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            sharedGrid[localY * sharedWidth + localX + 1] = currentGrid[idx + 1];
        if (threadIdx.y == 0 && y > 0)
            sharedGrid[(localY - 1) * sharedWidth + localX] = currentGrid[idx - width];
        if (threadIdx.y == blockDim.y - 1 && y < height - 1)
            sharedGrid[(localY + 1) * sharedWidth + localX] = currentGrid[idx + width];

        // Load corner halo cells
        if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0)
            sharedGrid[(localY - 1) * sharedWidth] = currentGrid[idx - width - 1];
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < width - 1 && y > 0)
            sharedGrid[(localY - 1) * sharedWidth + localX + 1] = currentGrid[idx - width + 1];
        if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < height - 1)
            sharedGrid[(localY + 1) * sharedWidth] = currentGrid[idx + width - 1];
        if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < width - 1 && y < height - 1)
            sharedGrid[(localY + 1) * sharedWidth + localX + 1] = currentGrid[idx + width + 1];
    }

    __syncthreads();

    if (x < width && y < height && x > 0 && y > 0)
    {
        int neighbors = 0;

        neighbors += sharedGrid[(localY - 1) * sharedWidth + localX];     // up
        neighbors += sharedGrid[(localY - 1) * sharedWidth + localX + 1]; // up right
        neighbors += sharedGrid[localY * sharedWidth + localX + 1];       // right
        neighbors += sharedGrid[(localY + 1) * sharedWidth + localX + 1]; // down right
        neighbors += sharedGrid[(localY + 1) * sharedWidth + localX];     // down
        neighbors += sharedGrid[(localY + 1) * sharedWidth + localX - 1]; // down left
        neighbors += sharedGrid[localY * sharedWidth + localX - 1];       // left
        neighbors += sharedGrid[(localY - 1) * sharedWidth + localX - 1]; // up left

        if (sharedGrid[localY * sharedWidth + localX])
        {
            nextGrid[idx] = (neighbors == 2 || neighbors == 3);
        }
        else
        {
            nextGrid[idx] = (neighbors == 3);
        }
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

extern "C" void launchUpdateKernelBasic(
    uint8_t* currentGrid,
    uint8_t* nextGrid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY)
{
  dim3 gridSize (gridX, gridY);
  dim3 blockSize(blockX, blockY);
  updateKernelBasic<<<gridSize, blockSize>>>(currentGrid, nextGrid, width, height);
  cudaDeviceSynchronize();
}

extern "C" void launchUpdateKernelShared(
    uint8_t* currentGrid,
    uint8_t* nextGrid,
    int width, int height,
    int gridX, int gridY,
    int blockX, int blockY)
{
  dim3 gridSize (gridX, gridY);
  dim3 blockSize(blockX, blockY);
  size_t sharedMemSize = (blockX + 2) * (blockY + 2) * sizeof(uint8_t); // +2 for halo
  updateKernelShared<<<gridSize, blockSize, sharedMemSize>>>(currentGrid, nextGrid, width, height);
  cudaDeviceSynchronize();
}