#include <iostream>
#include <chrono>
#include "PCNLayer.h"

int main(void)
{
    std::cout << "Hello world!" << std::endl;

    float *x = new float[10];
    for (int i=0; i < 10; i++)
        *(x+i) = i+1;

    PCN::PCLayer pc(10, 2);


    auto start = std::chrono::high_resolution_clock::now();
    pc.RunPrediction(x);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Time elapsed: " << elapsed.count() << " ms" << std::endl;

    std::cout << "Z after predicting: [" << pc.GetBeliefs()[0] << ", " << pc.GetBeliefs()[1] << "]" << std::endl;

    delete []x;
    return 0;
}