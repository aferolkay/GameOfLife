#pragma once

#include <vector>
#include <cstdint>

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
