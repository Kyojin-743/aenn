#include <cuda_matmul.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace aenn {
    namespace cuda_matmul {
        #define BLOCK_SIZE          16
        #define COARSE_FACTOR       4
        #define COARSED_BLOCK_SIZE  (BLOCK_SIZE*COARSE_FACTOR)
        #define intceil(A,B) ((A+B-1)/B)

        //launched with tile_width as blocksize
        __global__ void sgemm_tiled_coarsed_kernel(float* M, float* N, float* P, int outRows, int inner, int outCols) {
            const int bx = blockIdx.x, by = blockIdx.y;
            const int tx = threadIdx.x, ty = threadIdx.y;
            const int blockOfsY = by * COARSED_BLOCK_SIZE;
            const int blockOfsX = bx * COARSED_BLOCK_SIZE;
            // //Size of the area that a thread is responsible for

            __shared__ float Mds[COARSED_BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Nds[BLOCK_SIZE][COARSED_BLOCK_SIZE];
            float results[COARSE_FACTOR][COARSE_FACTOR] = { 0.0f };

            //Offset to this thread's block's row in M
            M += blockOfsY * inner;
            //Offset to this thread's blocks's column in N
            N += blockOfsX;
            //Offset to this thread's block's row,col in P
            P += blockOfsY*outCols + blockOfsX;

            for(int tile = 0; tile < intceil(inner,BLOCK_SIZE); ++tile) {
                //Load sharedM ( this thread loads values down M to shared )
                for(int t = 0; t < COARSED_BLOCK_SIZE; t += BLOCK_SIZE) {//blockdimy
                    if (((blockOfsY + t + ty) < outRows) && ((tile*BLOCK_SIZE + tx) < inner))
                        Mds[t + ty][tx] = M[(t + ty) * inner + tx];
                    else 
                        Mds[t + ty][tx] = 0.0f;
                }
                //Load sharedN ( this thread loads the values across to the right of the N to shared )
                for(int t = 0; t < COARSED_BLOCK_SIZE; t += BLOCK_SIZE) {//blockdimx
                    if (((tile*BLOCK_SIZE + ty) < inner) && ((blockOfsX + t + tx) < outCols))
                        Nds[ty][t + tx] = N[ty * outCols + t + tx];
                    else 
                        Nds[ty][t + tx] = 0.0f;
                }
                __syncthreads();
                //For next Tile -> move M and N pointers
                M += BLOCK_SIZE;                //sweeps the columns
                N += BLOCK_SIZE * outCols;      //sweeps the rows
                //write
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    for (int x = 0; x < COARSE_FACTOR; ++x) {
                        for (int y = 0; y < COARSE_FACTOR; ++y) {
                            if(((ty * COARSE_FACTOR + x) < COARSED_BLOCK_SIZE) && ((tx * COARSE_FACTOR + y) < COARSED_BLOCK_SIZE))
                                results[x][y] += Mds[ty * COARSE_FACTOR + x][k] * Nds[k][tx * COARSE_FACTOR + y];
                        }
                    }
                }        
                __syncthreads();
            }
            for (int x = 0; x < COARSE_FACTOR; ++x) {
                for (int y = 0; y < COARSE_FACTOR; ++y) {
                    int rowInP = x + ty * COARSE_FACTOR;
                    int colInP = tx * COARSE_FACTOR + y;
                    if(rowInP + blockOfsY < outRows && colInP + blockOfsX < outCols)
                        P[(rowInP * outCols) + colInP] += results[x][y];
                }
            }
        }
            
        int sgemm_tiled_coarsed(float *A, float *B, float *C, int numARows, int numACols, int numBRows, int numBCols) {
            float *A_d, *B_d, *C_d;

            int sizeA = numARows * numACols;
            int sizeB = numBRows * numBCols;
            int sizeC = numARows * numBCols;


            cudaMalloc(&A_d, sizeof(float) * sizeA);
            cudaMalloc(&B_d, sizeof(float) * sizeB);
            cudaMalloc(&C_d, sizeof(float) * sizeC);

            cudaMemcpy(A_d, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(B_d, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
            dim3 gridSize(intceil(numBCols, COARSED_BLOCK_SIZE), intceil(numARows, COARSED_BLOCK_SIZE), 1);

            sgemm_tiled_coarsed_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, numARows, numACols, numBCols);
            cudaError err;
            if ((err = cudaGetLastError()) != cudaSuccess)
                printf("\nKernel error: %s\n", cudaGetErrorString(err));
            cudaDeviceSynchronize();

            cudaMemcpy(C, C_d, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(A_d);
            cudaFree(B_d);
            cudaFree(C_d);
            
            return 0;
        }

    }
}