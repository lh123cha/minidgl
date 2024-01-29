#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)*ldb+(j)]
#define C(i,j) C[(i)*ldc+(j)]

#define TILE_K 16
#define TILE_X_4 32


//A(M*K),B(K*N),OUT(M*N)
__global__ void navie_matrixMul(const float* A,const float* B,const float* out,int M,int N,int K){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if(ty<M && tx<N){
        float c = 0.0;
        //out[ty][tx]=A的ty行与B的tx列的向量积
        for (int i=0;i<K;++i){
            c += A[ty*K+i] * B[i*N+tx];
        }
        out[ty * N + tx] = c;
    }
}

//GEMM tiled global memory to shared memory
template<int BLOCK_SIZE>
__global__ void tiled_matrixMul(const float * __restrict__ A,const float * __restrict__ B,const float* __restrict__ C,int M,int K,int N){
    //shared mem A_tile(bm,bk),B_tile(bk,bn)
    //这里bm=bk=bn=BLOCK_SIZE
    
    //block index 一个block处理一个tile的数据
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //第一个Atile的index
    int aBegin = K*BLOCK_SIZE*by;



    __shared__ float A_tile[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float B_tile[BLOCK_SIZE*BLOCK_SIZE];

    float accu = 0;
    //K/dk次大循环
    for(int tileIdx = 0 ; tileIdx<K/blockDim.x ; tileIdx++){
        //A的i行
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        //A的j列
        int j = tileIdx * blockDim.x + threadIdx.x;
        //A(i,j) to shared mem
        A_tile(threadIdx.y,threadIdx.x) = A(i.j);
        B_tile()

    }
}
//GEMM:Block Tile + Thread Tile + K Tile + float4
//BM=BN=128,BK=8.一个block处理一个tile
//TM=TN=8
//blockDim(BN/TN,BM-TM)
//gridDim((N+BN-1)/BN,(M+BM-1)/BM)
template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    const int THREAD_SIZE_M,
    const int THREAD_SIZE_N>
__global__ void tiled_matrixMul_vec4(const __restrict__float* a,const float* __restrict__ b,const float* __restrict__ c,int M,int K,int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty  threadIdx.y;
    int tid = ty * blockDim.x + tx;
    //分配A,B的共享内存 s_a(128,8),s_a,s_b大小一共2*128*8*4=8K
    __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K],s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];

    //计算shared_memory中的索引
    int load_smem_a_m = tid/2;
    int load_smem_a_k = (tid%2==0) ? 0 : 4;

    int load_smem_b_k = tid/32;
    int load_smem_b_n = (tid%32)*4;

    int load_gmem_a_m = by*BLOCK_SIZE_M + load_smem_a_m;
    int load_gmem_b_n = bx*BLOCK_SIZE_N + load_smem_b_n;

    float r_c[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0};
    //对A,B矩阵按K维度分块，每块BK长度
    for(int bk=0;bk<(K+BLOCK_SIZE_K-1)/BLOCK_SIZE_K;++bk){
        
        int load_gmem_a_k = bk*BLOCK_SIZE_K+load_smem_a_k;//global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) =  
    }
}



//GEMM with double buffer and bandconflict
// template<>
// __global__ void tiled_matrixMul_vec4_bandconflict(){

// }
