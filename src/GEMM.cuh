#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)*ldb+(j)]
#define C(i,j) C[(i)*ldc+(j)]
#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define TILE_K 16
#define TILE_X_4 32

using namespace cute;


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
    //这里是 BN=BM=128,BK=8,TM=TN=8的情况
    int load_smem_a_m = tid/2;
    int load_smem_a_k = (tid%2==0) ? 0 : 4;

    int load_smem_b_k = tid/32;
    int load_smem_b_n = (tid%32)*4;

    int load_gmem_a_m = by*BLOCK_SIZE_M + load_smem_a_m;
    int load_gmem_b_n = bx*BLOCK_SIZE_N + load_smem_b_n;

    float r_c[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0};
    //对A,B矩阵按K维度分块，每块BK长度
    //大循环2048/8=256次
    for(int bk=0;bk<(K+BLOCK_SIZE_K-1)/BLOCK_SIZE_K;++bk){
        
        int load_gmem_a_k = bk*BLOCK_SIZE_K+load_smem_a_k;//global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) =  FLOAT4(a[load_gmem_a_addr]);

        int load_gmem_b_k = bk*BLOCK_SIZE_K+load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(s_b[load_gmem_b_addr]);
        __syncthreads();
        #pragma unroll
        //小循环8*8=64次
        for(int k=0;k<BLOCK_SIZE_K;++k){
            //每个线程负责计算一个blockBM*BN中TM*TN个元素
            #pragma unroll
            for(int m=0;m<THREAD_SIZE_M;m++){
                #pragma unroll
                for(int n=0;n<THREAD_SIZE_N;++n){
                    int comp_smem_a_m = ty*THREAD_SIZE_M + m;
                    int comp_smeme_b_n = tx*THREAD_SIZE_N + n;
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smeme_b_n];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<THREAD_SIZE_M;++m){
        int store_gmem_c_m = by * BLOCK_SIZE_M + ty * THREAD_SIZE_M + m;
        #pragma unroll
        for(int n=0;n<THREAD_SIZE_N;++n){
            int store_gmem_c_n = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_N + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n])
        }
    }
}

//GEMM:Block Tile + Thread Tile + K Tile + float4 + double buffer prefetch
//BM=BN=128,BK=8.一个block处理一个tile
//TM=TN=8
//blockDim(BN/TN,BM-TM)
//gridDim((N+BN-1)/BN,(M+BM-1)/BM)
template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    cont int THREAD_SIZE_M,
    const int THREAD_SIEZ_N,
    const bool DOUBLE_BUFFER>
__global__ void tiled_matrixMul_vec4_db(const __restrict__float* a,const float* __restrict__ b,const float* __restrict__ c,int M,int K,int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int THREAD_N_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIEZ_N;
    const int THREAD_M_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int THREAD_NUM_PER_BLOCK = THREAD_M_PER_BLOCK * THREAD_N_PER_BLOCK;
    //当前线程在256个一个block的256个线程中的id号
    const int tid = ty * THREAD_N_PER_BLOCK + tx;

    //分配A,B矩阵的共享内存，加上prefetch的共享内存
    //A_smem矩阵需要转置shape为(bk,bm),因为线程要取A得一列，转置之后A的一列就是一行，可以利用局部性
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    //为C矩阵分配寄存器
    float r_c[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0};
    //为A_smem，B_smem矩阵分配寄存器,大小为2*THREA_SIZE_M
    float reg_a[2][THREAD_SIZE_M];
    float reg_b[2][THREAD_SIZE_N];
    //使用寄存器将A,B矩阵中的元素从Global memory加载到shared memory中



    const int A_SMEM_THREAD_PER_ROW = BLOCK_SIZE_K / 4;//因为float4，所以除4
    const int B_SMEM_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    //该tid的线程在共享内存中加载A_smem,B_smem的row、col的位置
    const int A_SMEM_ROW_START = tid / A_SMEM_THREAD_PER_ROW;
    const int B_SMEM_ROW_START = tid / B_SMEM_THREAD_PER_ROW;
    
    const int A_SMEM_COL = tid % A_SMEM_THREAD_PER_ROW * 4;
    const int B_SMEM_COL = tid % B_SMEM_THREAD_PER_ROW * 4;
    //如果一个线程处理多行数据，row stride决定一个线程处理距离多远的row的数据
    const int A_SMEM_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_SMEM_THREAD_PER_ROW;//总线程数/每行线程数
    const int B_SMEM_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_SMEM_THREAD_PER_ROW; 
    
    //每个线程处理多少个数 ldg_num_a,ldg_num_b
    const int ldg_num_a = BLOCK_SIZE_M*BLOCK_SIZE_K/THREAD_NUM_PER_BLOCK;
    const int ldg_num_b = BLOCK_SIZE_N*BLOCK_SIZE_K/THREAD_NUM_PER_BLOCK;
    float ldg_a_reg[ldg_num_a];
    float ldg_b_reg[ldg_num_b];
    //load gloab memory to shared memory
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_M;i+=A_SMEM_ROW_STRIDE){
        int ldg_index = i/A_SMEM_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index])=FETCH_FLOAT4(A[OFFSET(
            BLOCK_SIZE_M*by+A_SMEM_ROW_START+i,
            A_SMEM_COL,
            K)]);
        As[0][A_SMEM_COL][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index];
        As[0][A_SMEM_COL+1][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index+1];
        As[0][A_SMEM_COL+2][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index+2];
        As[0][A_SMEM_COL+3][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index+3];
    }
    #pragma unroll
    for(int i=0;i<BLOCK_SIZE_K;i+=B_SMEM_ROW_STRIDE){
        FETCH_FLOAT4(Bs[0][B_SMEM_ROW_START+i][B_SMEM_COL])=FETCH_FLOAT4(B[OFFSET(
            B_SMEM_ROW_START+i,
            BLOCK_SIZE_N*bx+B_SMEM_COL,
            N)]);
    }
    __syncthreads();
    //将数据从shared memory加载到寄存器中
    #pragma unroll
    for(int thread_y = 0;thread_y<THREAD_SIZE_M;thread_y+=4){
        FETCH_FLOAT4(reg_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_M*ty+thread_y]);
    }
    #pragma unroll
    for(int thread_x = 0;thread_x<THREAD_SIZE_N;thread_x+=4){
        FETCH_FLOAT4(reg_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_N*tx+thread_x]);
    }
    //大迭代K/BLOCK_SIZE_K次
    int write_stage_idx = 1;//控制写块的id
    int tile_idx = 0;//表示大循环的k值
    do{
        tile_idx += BLOCK_SIZE_K;
        //从global memory中prefetch加载到next tile中
        //先将global memory移动到寄存器
        if(tile_idx < K){
            #pragma unroll
            for(int i = 0;i < BLOCK_SIZE_M;i+=A_SMEM_ROW_STRIDE){
                int ldg_index = i/ A_SMEM_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index])=FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M*by+A_SMEM_ROW_START+i,
                    A_SMEM_COL+tile_idx,
                    K)]);
            }
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i+=B_SMEM_ROW_STRIDE){
                int ldg_index = i/B_SMEM_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index])=FETCH_FLOAT4(B[OFFSET(
                    tile_idx+B_SMEM_ROW_START+i,
                    BLOCK_SIZE_N*bx+B_SMEM_COL,
                    N)]);
            }
        }
        int load_stage_idx = write_stage_idx ^ 1;//读块是写快的取反
        #pragma unroll
        //小迭代 BLOCK_SIZE_K-1次小循环，总共完成BLOCK_K次小迭代，这里先进行BLOCK_K-1次小迭代
        //需要将下一轮小迭代用到的数据提前写入寄存器中
        //因为在最后一轮小迭代之前，需要将下一轮大迭代用到的数据搬运到共享内存和寄存器中
        //所以这里先进行BLOCK_SIZE_K-1次小迭代,最后一次小迭代结束之前需要搬运数据的操作。
        for(int j=0;j<BLOCK_SIZE_K-1;j++){
            //从shared memory中prefetch加载到next tile的寄存器中
            #pragma unroll
            //因为我们之前已经fetch第0行了，所以这里是j+1
            for(int thread_y = 0;thread_y<THREAD_SIZE_M;thread_y+=4){
                FETCH_FLOAT4(reg_a[(j+1)%2][thread_y])=FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_M*ty+thread_y]);
            }
            for(int thread_x = 0;thread_x<THREAD_SIZE_N;thread_x+=4){
                FETCH_FLOAT4(reg_b[(j+1)%2][thread_x])=FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_N*tx+thread_x]);
            }
            //计算C的结果
            #pragma unroll
            for(int thread_y = 0;thread_y<THREAD_SIZE_M;++thread_y){
                for(int thread_x = 0;thread_x<THREAD_SIZE_N;++thread_x){
                    r_c += reg_a[j%2][thread_y]*reg_b[j%2][thread_x];
                }
            }
        }
        //将下一轮大迭代用到的数据搬入共享内存中
        if(tile_idx < K){
            #pragma unroll
            for(int i = 0;i<BLOCK_SIZE_M;i+=A_SMEM_ROW_STRIDE){
                int ldg_index = i / A_SMEM_ROW_STRIDE * 4;
                As[write_stage_idx][A_SMEM_COL][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_SMEM_COL+1][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_SMEM_COL+2][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_SMEM_COL+3][A_SMEM_ROW_START+i] = ldg_a_reg[ldg_index+3];
            }
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i+=B_SMEM_ROW_STRIDE){
                int ldg_index = i/B_SMEM_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_SMEM_ROW_START+i][B_SMEM_COL])=FETCH_FLOAT(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
            write_stage_idx ^=1;
        }
        //将下一轮大迭代中第一轮小迭代用到的数据搬运到寄存器中，
        #pragma unroll
        for(int trhead_y = 0;thread_y<THREAD_SIZE_M;thread_y+=4){
            FETCH_FLOAT4(reg_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_M*ty+thread_y]);
        }
        #pragma unroll
        for(int thread_x = 0;thread_x<THREAD_SIZE_N;thread_x+=4){
            FETCH_FLOAT4(reg_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_N*tx+thread_x]);
        }
        #pragma unroll 
        for(int thread_y = 0;thread_y<THREAD_SIZE_M;++thread_y){
            for(int thread_x = 0;thread_x<THREAD_SIZE_N;++thread_x){
                r_c[thread_y][thread_x] += reg_a[1][thread_y]*reg_b[1][thread_x];
            }
        }
        //写会C中
        #pragma unroll
        FOR(int thread_y = 0;thread_y<THREAD_SIZE_M;++thread_y){
            #pragma unroll
            for(int thread_x = 0;thread_x<THREAD_SIZE_N;++thread_x){
                FETCH_FLOAT4(C[OFFSET(
                    BLOCK_SIZE_M*by + THREA_SIZE_M*ty+thread_y,
                    BLOCK_SIZE_N * bx + THREA_SIZE_N*tx+thread_x,
                    N)]) = FETCH_FLOAT4(r_c[thread_y][thread_x]);
            }
        }

    }
}



template<
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N,
    cont int THREAD_SIZE_M,
    const int THREAD_SIEZ_N,
    const bool DOUBLE_BUFFER>
__global__ void cutlass_matrixMul(const __restrict__float* a,const float* __restrict__ b,const float* __restrict__ c,int M,int K,int N){
     // A matrix configuration
  using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
  using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
  using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TilesShape          = Shape<_128,_128,_64>;                           // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
  using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder 

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TilesShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //

  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, Int<1>{}));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, Int<1>{}));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, Int<1>{}));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, Int<1>{}));

  block_A.reset(M * K);
  block_B.reset(K * N);
  block_C.reset(M * N);
  block_D.reset(M * N);

  //
  // Launch GEMM on the device
  //
 
  status = gemm_op({
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    block_A.get(),
    stride_A,
    block_B.get(),
    stride_B,
    {block_C.get(), stride_C, block_D.get(), stride_D, {alpha, beta}}
  });

  if (status != cutlass::Status::kSuccess) {
    return ;
  }
}
