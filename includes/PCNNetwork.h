#pragma once
#include "PCNLayer.h"

namespace Deep
{
    class PCNetwork
    {
    public:
        PCNetwork() = default;
        ~PCNetwork() = default;

    private:
        std::unique_ptr<PCLayer[]> layers;

        size_t inputSize;
        size_t hiddenSize;
        size_t outputSize;
    };
}