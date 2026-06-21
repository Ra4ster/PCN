#include "PCNNetwork.h"
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>

// Generates a noisy one-hot vector of size `classes`
static std::vector<float> MakeTarget(int label, int classes, std::mt19937 &rng, float noise = 0.05f)
{
    std::uniform_real_distribution<float> jitter(-noise, noise);
    std::vector<float> t(classes, 0.0f);
    t[label] = 1.0f;
    for (auto &v : t) v += jitter(rng);
    return t;
}

// Generates a random "MNIST-like" input (784 values in [0, 1])
static std::vector<float> MakeInput(std::mt19937 &rng)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(784);
    for (auto &v : x) v = dist(rng);
    return x;
}

// Returns the index of the maximum value (argmax)
static int Argmax(const float *data, size_t size)
{
    return static_cast<int>(std::max_element(data, data + size) - data);
}

int main()
{
    std::cout << "[Deepity] Booting...\n";

    Deep::PCNetwork net;
    net.AddLayer(784, 256, 1e-4f, 1e-4f, 30);
    net.AddLayer(256, 64,  1e-4f, 1e-4f, 30);
    net.AddLayer(64,  10,  1e-4f, 1e-4f, 30);
    net.Compile();

    std::mt19937 rng(42);

    // --- INFERENCE PASS: feed inputs bottom-up, watch energy settle ---

    std::cout << "\n[Deepity] -- INFERENCE PASS --\n";

    const int NUM_SAMPLES    = 200;
    const int REPORT_EVERY   = 50;

    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        auto input = MakeInput(rng);
        net.InferenceStep(input.data());

        if ((i + 1) % REPORT_EVERY == 0)
            std::cout << "  [Step " << (i + 1) << "] Energy: " << net.GetTotalEnergy() << '\n';
    }

    net.FlushInference();
    std::cout << "  [Post-flush] Energy: " << net.GetTotalEnergy() << '\n';

    // --- GENERATION PASS: feed targets top-down for each class label ---

    std::cout << "\n[Deepity] -- GENERATIVE PASS (per class) --\n";

    const int NUM_CLASSES    = 10;
    const int GEN_SAMPLES    = 100;

    for (int label = 0; label < NUM_CLASSES; ++label)
    {
        // Reset to a fresh network state per label by just running the pass;
        // energy difference between labels is the signal we care about
        for (int i = 0; i < GEN_SAMPLES; ++i)
        {
            auto target = MakeTarget(label, NUM_CLASSES, rng);
            net.GenerationStep(target.data());
        }
        net.FlushGeneration();

        std::cout << "  [Label " << label << "] Energy after generative pass: "
                  << net.GetTotalEnergy() << '\n';
    }

    // --- BIDIRECTIONAL: interleave inference and generative steps ---
    std::cout << "\n[Deepity] -- BIDIRECTIONAL PASS --\n";

    const int BIDI_STEPS = 200;

    for (int i = 0; i < BIDI_STEPS; ++i)
    {
        int label  = rng() % NUM_CLASSES;
        auto input  = MakeInput(rng);
        auto target = MakeTarget(label, NUM_CLASSES, rng);

        net.InferenceStep(input.data());
        net.GenerationStep(target.data());

        if ((i + 1) % REPORT_EVERY == 0)
            std::cout << "  [Step " << (i + 1) << "] Energy: " << net.GetTotalEnergy() << '\n';
    }

    net.FlushInference();
    net.FlushGeneration();
    std::cout << "  [Post-flush] Energy: " << net.GetTotalEnergy() << '\n';

    // --- SAVE / LOAD round-trip sanity check ---
    std::cout << "\n[Deepity] -- SAVE/LOAD ROUND-TRIP --\n";

    const char *model_path = "/tmp/deepity_test.bin";
    float energy_before = net.GetTotalEnergy();
    net.SaveModel(model_path);
    net.LoadModel(model_path);
    float energy_after = net.GetTotalEnergy();

    std::cout << "  Energy before save : " << energy_before << '\n';
    std::cout << "  Energy after load  : " << energy_after  << '\n';
    std::cout << "  Round-trip " << (std::fabs(energy_before - energy_after) < 1e-4f ? "PASSED" : "FAILED") << '\n';

#ifdef _DEBUG
    net.DebugStats();
#endif

    std::cout << "\n[Deepity] Done.\n";
    return 0;
}