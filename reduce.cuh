#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/std/complex>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace column_reduce {
constexpr unsigned block_size = 256, warps = 8, tile_channels = 32;
enum class Mode { two_kernel, cooperative };
enum class Status { success, invalid_argument, overflow, insufficient_workspace, cooperative_unsupported, cuda_error };
struct Plan { uint64_t n=0, k=0, partitions=0, tiles=0, tasks=0; size_t workspace_bytes=0; unsigned blocks=0; int device=-1; Mode mode=Mode::two_kernel; };
inline const char* status_string(Status s) { switch(s) { case Status::success:return "success"; case Status::invalid_argument:return "invalid argument"; case Status::overflow:return "size overflow"; case Status::insufficient_workspace:return "insufficient or misaligned workspace"; case Status::cooperative_unsupported:return "cooperative launch unsupported"; default:return "CUDA error"; } }
inline bool mul(uint64_t a,uint64_t b,uint64_t& out) { if(b && a>UINT64_MAX/b) return false; out=a*b; return true; }
inline uint64_t ceil_div(uint64_t x,uint64_t y) { return x/y+(x%y!=0); }

template<class T> __device__ T shuffle_down(T v, int delta) { return __shfl_down_sync(0xffffffffu,v,delta); }
template<> __device__ inline cuda::std::complex<double> shuffle_down(cuda::std::complex<double> v,int delta) { return {__shfl_down_sync(0xffffffffu,v.real(),delta),__shfl_down_sync(0xffffffffu,v.imag(),delta)}; }

template<class InputT,class AccT,class Op> struct InputReader {
 const InputT* input; uint64_t n,k,partition,partitions; Op op;
 __device__ AccT read(uint64_t row,uint64_t c) const { return static_cast<AccT>(op(input[row*k+c])); }
 __device__ uint64_t count() const { return n; }
 __device__ uint64_t start(uint64_t r) const { return partition + r*partitions; }
 __device__ uint64_t step(uint64_t rows) const { return rows*partitions; }
};
template<class AccT> struct PartialReader {
 const AccT* data; uint64_t k,partitions;
 __device__ AccT read(uint64_t row,uint64_t c) const { return data[row*k+c]; }
 __device__ uint64_t count() const { return partitions; }
 __device__ uint64_t start(uint64_t r) const { return r; }
 __device__ uint64_t step(uint64_t rows) const { return rows; }
};
// All 256 threads participate in every call, including the barriers.
template<class AccT,class Reader> __device__ void reduce_tile(Reader reader, AccT* output,uint64_t tile,uint64_t k,AccT* shared) {
 unsigned lane=threadIdx.x%32, warp=threadIdx.x/32;
 unsigned width=1; uint64_t remaining=k-tile*32; while(width<32 && width<remaining) width*=2;
 unsigned ch=lane%width, row_in_warp=lane/width, rows_per_warp=32/width;
 uint64_t c=tile*32+ch, row=reader.start(warp*rows_per_warp+row_in_warp);
 uint64_t stride=reader.step(warps*rows_per_warp);
 AccT sum{};
 if(c<k) while(row<reader.count()) { sum += reader.read(row,c); if(reader.count()-row<=stride) break; row+=stride; }
 for(unsigned offset=width;offset<32;offset*=2) sum += shuffle_down(sum,offset);
 if(row_in_warp==0) shared[warp*32+ch]=sum;
 __syncthreads();
 if(warp==0 && lane<width && c<k) { AccT total{}; for(unsigned w=0;w<warps;++w) total+=shared[w*32+lane]; output[c]=total; }
 __syncthreads();
}
template<class InputT,class AccT,class Op> __global__ void first_kernel(const InputT* in,AccT* partial,uint64_t n,uint64_t k,uint64_t p,uint64_t tasks,Op op) {
 __shared__ double storage[warps*32*((sizeof(AccT)+sizeof(double)-1)/sizeof(double))]; AccT* shared=reinterpret_cast<AccT*>(storage);
 for(uint64_t t=blockIdx.x;t<tasks;) { uint64_t tile=t/p, part=t%p; InputReader<InputT,AccT,Op> r{in,n,k,part,p,op}; reduce_tile<AccT>(r,partial+part*k,tile,k,shared); if(tasks-t<=gridDim.x) break; t+=gridDim.x; }
}
template<class AccT> __global__ void second_kernel(const AccT* partial,AccT* out,uint64_t k,uint64_t p,uint64_t tiles) {
 __shared__ double storage[warps*32*((sizeof(AccT)+sizeof(double)-1)/sizeof(double))]; AccT* shared=reinterpret_cast<AccT*>(storage); PartialReader<AccT> r{partial,k,p};
 for(uint64_t t=blockIdx.x;t<tiles;) { reduce_tile<AccT>(r,out,t,k,shared); if(tiles-t<=gridDim.x) break; t+=gridDim.x; }
}
template<class InputT,class AccT,class Op> __global__ void cooperative_kernel(const InputT* in,AccT* out,AccT* partial,uint64_t n,uint64_t k,uint64_t p,uint64_t tasks,uint64_t tiles,Op op) {
 __shared__ double storage[warps*32*((sizeof(AccT)+sizeof(double)-1)/sizeof(double))]; AccT* shared=reinterpret_cast<AccT*>(storage);
 for(uint64_t t=blockIdx.x;t<tasks;) { uint64_t tile=t/p,part=t%p; InputReader<InputT,AccT,Op> r{in,n,k,part,p,op}; reduce_tile<AccT>(r,partial+part*k,tile,k,shared); if(tasks-t<=gridDim.x) break; t+=gridDim.x; }
 cooperative_groups::this_grid().sync();
 PartialReader<AccT> r{partial,k,p};
 for(uint64_t t=blockIdx.x;t<tiles;) { reduce_tile<AccT>(r,out,t,k,shared); if(tiles-t<=gridDim.x) break; t+=gridDim.x; }
}
template<class InputT,class AccT,class Op> Status make_plan(uint64_t n,uint64_t k,Mode mode,Plan& plan,cudaError_t* error=nullptr,unsigned sm_multiplier=4) {
 plan=Plan{}; plan.n=n; plan.k=k; plan.mode=mode; if(error) *error=cudaSuccess;
 if(!sm_multiplier) return Status::invalid_argument;
 if(!k) return Status::success;
 uint64_t elements, bytes, partials;
 if(!mul(n,k,elements) || !mul(elements,sizeof(InputT),bytes) || bytes>SIZE_MAX || !mul(k,sizeof(AccT),bytes) || bytes>SIZE_MAX) return Status::overflow;
 cudaError_t e=cudaGetDevice(&plan.device); if(e!=cudaSuccess) { if(error)*error=e; return Status::cuda_error; }
 cudaDeviceProp prop{}; e=cudaGetDeviceProperties(&prop,plan.device); if(e!=cudaSuccess) { if(error)*error=e; return Status::cuda_error; }
 if(prop.multiProcessorCount<=0 || prop.maxGridSize[0]<=0) return Status::invalid_argument;
 uint64_t limit=uint64_t(prop.multiProcessorCount)*sm_multiplier;
 limit=std::min<uint64_t>(limit,static_cast<uint64_t>(prop.maxGridSize[0]));
 if(mode==Mode::cooperative) {
  if(!prop.cooperativeLaunch) return Status::cooperative_unsupported;
  int active=0; e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,cooperative_kernel<InputT,AccT,Op>,block_size,0);
  if(e!=cudaSuccess) { if(error)*error=e; return Status::cuda_error; }
  if(active<=0) return Status::cooperative_unsupported;
  limit=std::min<uint64_t>(limit,uint64_t(active)*prop.multiProcessorCount);
 }
 // Channel tiles already supply independent work; split rows only to fill the grid.
 plan.tiles=ceil_div(k,tile_channels);
 plan.partitions=std::max<uint64_t>(1,std::min(ceil_div(n,block_size),ceil_div(limit,plan.tiles)));
 if(!mul(plan.partitions,k,partials) || !mul(partials,sizeof(AccT),bytes) || bytes>SIZE_MAX || !mul(plan.tiles,plan.partitions,plan.tasks)) return Status::overflow;
 plan.workspace_bytes=static_cast<size_t>(bytes);
 plan.blocks=static_cast<unsigned>(std::min<uint64_t>(plan.tasks,limit)); return plan.blocks?Status::success:Status::invalid_argument;
}
template<class InputT,class AccT,class Op> Status execute(const Plan& plan,const InputT* input,AccT* output,Op op,cudaStream_t stream,void* workspace,size_t workspace_bytes,cudaError_t* error=nullptr) {
 if(error)*error=cudaSuccess; if(!plan.k) return Status::success;
 if(!output || (plan.n && !input) || !plan.blocks || !plan.partitions || !plan.tasks) return Status::invalid_argument;
 if(!workspace || workspace_bytes<plan.workspace_bytes || reinterpret_cast<uintptr_t>(workspace)%alignof(AccT)) return Status::insufficient_workspace;
 int device; cudaError_t e=cudaGetDevice(&device); if(e!=cudaSuccess) { if(error)*error=e; return Status::cuda_error; } if(device!=plan.device) return Status::invalid_argument;
 AccT* partial=static_cast<AccT*>(workspace);
 if(plan.mode==Mode::two_kernel) {
  first_kernel<InputT,AccT,Op><<<plan.blocks,block_size,0,stream>>>(input,partial,plan.n,plan.k,plan.partitions,plan.tasks,op);
  e=cudaGetLastError(); if(e==cudaSuccess) {
   unsigned second_blocks=static_cast<unsigned>(std::min<uint64_t>(plan.blocks,plan.tiles));
   second_kernel<AccT><<<second_blocks,block_size,0,stream>>>(partial,output,plan.k,plan.partitions,plan.tiles); e=cudaGetLastError();
  }
 } else {
  auto n=plan.n,k=plan.k,p=plan.partitions,tasks=plan.tasks,tiles=plan.tiles;
  void* args[]={&input,&output,&partial,&n,&k,&p,&tasks,&tiles,&op};
  e=cudaLaunchCooperativeKernel((void*)cooperative_kernel<InputT,AccT,Op>,dim3(plan.blocks),dim3(block_size),args,0,stream);
 }
 if(e!=cudaSuccess) { if(error)*error=e; return Status::cuda_error; } return Status::success;
}
}
