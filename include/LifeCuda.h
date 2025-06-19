#pragma once

#include <vector>
#include <cstdint>
#include <deque>

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
    std::deque<float> justCalculations;

private:
    void resetJustCalculations ()
    {
        justCalculations.clear();
    }

    void addJustCalculation (float value)
    {
        justCalculations.push_back(value);
        if (justCalculations.size() > 1000) // Keep the last 1000 calculations
        {
            justCalculations.pop_front();
        }
    }

    float getJustCalculations () const
    {
        if (justCalculations.empty())
            return 0.0f;

        float sum = 0.0f;
        for (const auto &value : justCalculations)
        {
            sum += value;
        }
        return sum / justCalculations.size();
    }

public:
    LifeCuda(int w, int h);
    ~LifeCuda();

    void setInitialState(std::vector<uint8_t> &initialState);

    void update();

    uint8_t getLifeform(int x, int y);

    void alterKernelType ();

    float getAverageCalculationTime() const
    {
        return getJustCalculations();
    }
};
