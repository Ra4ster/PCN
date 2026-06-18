#include <cstdint>

class SplitMix64 {
    public:
    SplitMix64(uint64_t initial_seed) : state(initial_seed) {}
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }
    float next_float() {
        return (next() >> 11) * 0x1.0p-53; // [0, 1)
    }
};