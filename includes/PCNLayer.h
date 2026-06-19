#pragma once
#include <cstddef>
#include <memory>
#include <Activations.h>

/**
 * @file PCNLayer.h
 * @brief Defines the single-layer implementation of a PC model.
 * 
 * This header includes implementations of inference steps, inference updates, learning, and prediction.
 * 
 * Usage:
 *  #include <PCNLayer.h>
 *  using namespace Deep;
 * 
 * Example:
 *  PCLayer layer(2, 3);
 *  layer.RunPrediction({ 0.1f, 0.75f });
 * 
 * @note All members are stored as unique pointers except for the input itself.
 * @version 1.0
 * @date 2026-06-18
 * @author Jack Rose
 */

using ActivationFn = void(*)(float*, size_t);

namespace Deep
{
    class PCLayer
    {
    public:
        PCLayer(size_t inSize, size_t outSize, float lr = 1e-6, float ir = 1e-6, int stepSize = 30, ActivationFn act = relu); // default
        PCLayer(const PCLayer&) = delete;
        PCLayer& operator=(const PCLayer&) = delete;
        ~PCLayer(); // destructor

        /// @brief Calculates prediction and returns it.
        /// @return prediction
        void CalcPrediction() noexcept;
        /// @brief Gets error; note, prediction must be ran before this.
        /// @attention x must be set
        void CalcStepError(const float *x) noexcept;
        /// @brief Updates z, one inference step
        void UpdateBeliefs() noexcept;
        /// @brief Runs an inference loop (learns z)
        /// @attention some inputs may be remaining; ensure you run `Flush()` after.
        /// @param x Input
        void RunPrediction(const float *x) noexcept;
        /// @brief Runs remaining predictions.
        void Flush() noexcept;
        /// @brief Updates weights using the learning rule `W += lr * e * x^T`
        /// @attention This assumes error has already been calculated
        /// @param x Inputs to learn
        void UpdateWeights(const float *x) noexcept;

        // @internal GETTERS
        float *GetPrediction() const noexcept { return arr + pBegin; }
        float *GetInferenceError() const noexcept { return arr + errBegin; }
        float *GetBeliefs() const noexcept { return arr + zBegin; }
        float *GetWeights() const noexcept { return arr; }
        float GetLR() const noexcept { return lr; }
        float GetIR() const noexcept { return ir; }
        size_t GetInputSize() const noexcept { return inputSize; }
        size_t GetOutputSize() const noexcept { return outputSize; }
        size_t GetBatchSize() const noexcept { return B; }

        #ifdef _DEBUG
        void DebugStats() const;
        #endif

    private:

        void RunBatchedPrediction(const float *x) noexcept;

        // @internal SIZES
        size_t inputSize = 0;
        size_t outputSize = 0;
        // Batch size - defaults to 64.
        size_t B = 64;

        // @internal PARAMS
        float lr = 0.0f; // learning rate
        float ir = 0.0f; // inference rate
        int stepSize = -1; // Inference steps (TODO: May become optimized!!)
        ActivationFn activation;

        // @internal MEMBERS
        float *arr; // Weights
        // @internal the following are used to reduce cache misses:
        size_t zBegin;
        size_t pBegin;
        size_t errBegin;
        size_t totalSize;

        std::unique_ptr<float[]> inputBuffer;  // [B * inSize] staging buffer
        size_t pendingCount = 0;
    };
}