#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math_contnts.h>
#include <type_traits>
using namespace std;

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);}

constexpr int WARP_SIZE = 32;
#define FINAL_MASK 0xffffffff

template<typename T> struct SumOp{
    __device__ __forceinline__ T operator()(const T &a,const T &b) const {
      return a+b;
    }
};
template<typename T> struct MaxOp{
  __device__ __forceinline__ T operator()(const T &a,const T &b) const {
    return a+b;
  }
};

template<template <typename> class ReductionOp,typename T,int thread_group_size>
__inline__ __device__ T WrapAllReduce(T val){
  for (int mask = thread_group_size/2;mask>0;mask/=2){
    val = ReductionOp<T>()(val,__shfl_xor_sync(FINAL_MASK,val,mask));
  }
  return val;
}

template<template <typename> class ReductionOp,typename T,int thread_group_size>
__inline__ __device__ T BlockAllReduce(T val){
  static __shared__ T shared[32];
  
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >>5;

  val = WrapAllReduce<ReductionOp<T>,T,thread_group_size>(val);

  if(lane==0){
    shared[wid] = val;
  }
  _syncthreads();

  if(std::is_same<ReductionOp<T>,SumOp<T>>::value){
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
  }else{
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 1e-20f;
  }

  val = WrapAllReduce<ReductionOp<T>,T,thread_group_size>(val);

  return val;
}
////////////////////////////////////////Optimized Reduce//////////////////////////////////////
//NDArray仅支持float数据所以只实现float
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
  #pragma unroll
  for(int mask = kWarpSize >> 1;mask>=1;mask>>=1){
  //__shfl_xor_sync的作用是在一个warp内的线程之间交换变量
    val += __shfl_xor_sync(FINAL_MASK,val,mask); 
  }
}
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val){
  #pragma unroll
  for(int mask = kWarpSize >> 1;mask>=1;mask>>=1){
    //在warp内的线程之间交换变量
    val =fmaxf(val,__shfl_xor_sync(FINAL_MASK,val,mask)); 
  }
}

//对一个block中的线程进行求和操作
//per block has 128 threads
//1D block, 1D grid
template<const int NUM_THREADS=1024>
__device__ __forceinline__ float block_reduce_sum(float val){
  constexpr int NUM_WARPS = (NUM_THREADS+WARP_SIZE -1)/WARP_SIZE;
  int warpid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_sum<WARP_SIZE>(val);
  if(lane==0) shared[warpid] = val
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum<NUM_WARPS>(val);
  return val;
}

//对一个block中的线程进行求最大值操作
template<const int NUM_THREADS=1024>
__device__ __forceinline__ float block_reduce_max(float val){
  constexpr int NUM_WARPS = (NUM_THREADS+WARP_SIZE-1)/WARP_SIZE;
  int warpid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;
  static __shared__  float shared[NUM_WARPS];

  val = warp_reduce_max<WARP_SIZE>(val);
  if(laneid==0) shared[warpid] = val;
  __syncthreads();
  val = (laneid<NUM_WARPS) ? shared[laneid]: 0.0f;
  val = warp_reduce_max<NUM_WARPS>(val);
  return val;
}

//Block reduce sum with float4
//grid(N/1024),block(1024)
//a's shape : (N*1),y = sum(a)
template<const int NUM_THREADS = 1024>
__global__ void block_all_reduce_sum_float4_kernel(float* a,float * y,int N){
  int tid = threadIdx.x;
  //float4向量化加载一次加载四个常量数据，所以*4
  int idx = (blockIdx.x*NUM_THREADS+tid)*4;
  constexpr int NUM_WARPS = (NUM_THREADS+WARP_SIZE-1)/WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];


  float4 reg_a = FLOAT4(a[idx]);
  float sum_of_four = (idx<N)?(reg_a.x+reg_a.y+reg_a.z+reg_a.w) : 0.0f;
  int wrapid = tid / WARP_SIZE;
  int laneid = tid % WARP_SIZE;
  //warp level reduce
  sum = warp_reduce_sum<WARP_SIZE>(sum);

  if(lane==0){
    reduce_smem[warpid] = sum;
  }
  __syncthreads();
  //在每个warp求和完成之后，第一个warp将之前所有warp的求和计算得到最终的结果。
  sum = (laneid<NUM_WARPS) ? reduce_smem[laneid] : 0.0f;
  //第一个warp
  if(warpid==0){
    sum = warp_reduce_sum<NUM_WARPS>(sum);
  }
  if(tid==0){
    atomicAdd(y,sum);
  }
}

//softmax kernel
//grid(N/1024),block(1024)
template<const int NUM_THREADS=1024>
__global__ void softmax_safe_kernel_v2(float* x,float* y,float* total,int N){
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  constexpr int NUM_WARPS = (NUM_THREADS+WARP_SIZE-1)/WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  __shared__ float s_max;

  float tmp = (idx < N) ? x[idx] : -1e20f;
  float max_val = block_reduce_max<NUM_THREADS>(tmp);
  if(tid==0){
    s_max = max_val;
  }
  __syncthreads();

  float exp_val = (idx < N) ? expf(tmp-s_max) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);

  if(tid==0){
    atomicAdd(total,sum);
  }
  //这里使用threadfence 在网格内进行内存同步，确保total求和数据可以被其他线层使用
  //适应syncthreads()也可以，但会造成多余的时间消耗。
  __threadfence();
  if(idx<N){
    y[idx] = exp_val/(*total);
  }
}

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template<const int NUM_THREADS = 1024/4>
__global__ void softmax_safe_v2_vec4(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4; 
  
  float4 reg_x = FLOAT4(x[idx]);

  float4 reg_max;
  reg_max.x = (idx<N) ? reg_x.x : -1e20f;
  reg_max.y = (idx<N) ? reg_x.y : -1e20f;
  reg_max.z = (idx<N) ? reg_x.z : -1e20f;
  reg_max.w = (idx<N) ? reg_x.w : -1e20f;
  float max_val = fmaxf(reg_max.x,fmaxf(reg_max.y,fmaxf(reg_max.z,reg_max.w)));
  float s_max = block_reduce_max<NUM_THREADS>(max_val);
  _syncthreads();

  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x-s_max) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y-s_max) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z-s_max) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w-s_max) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence  注意这里需要网格级别的内存同步
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y; 
  }
}


//final softmax kernel with warp block switch
template<const int NUM_THREADS=1024>  
void softmaxDispatch(){
  
}


















template <typename T>
__global__ void softmax_kernel(T* qk_buf,const int batch_size,const int head_num,const int seq_len
,const T scaler){
  int qk_offset = blockIdx.x*seq_len*seq_len;

}





