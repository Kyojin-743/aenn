#pragma once

namespace aenn {
    namespace cuda_matmul {
        int sgemm_tiled_coarsed(float *A, float *B, float *C, int numARows, int numACols, int numBRows, int numBCols);
        void launch_matmul(const float* A, const float* B, float* C, int numARows, int numACols, int numBRows, int numBCols);
    }
}