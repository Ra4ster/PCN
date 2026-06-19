#!/bin/bash

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export GOTO_NUM_THREADS=${OPENBLAS_NUM_THREADS}
echo "Running with " ${OMP_NUM_THREADS} " OMPTHREADS and " ${OPENBLAS_NUM_THREADS} " BLASTHREADS."

# Replace with Release || Debug:
cmake --build build/Release
./bin/Deepity