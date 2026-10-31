#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <climits>
#include <limits>

// CUB adapters for the same row-major column reduction as reduce.cuh.
namespace cub_baseline {
enum class Mode { segmented, per_channel };
struct Plan { uint64_t n=0, k=0; size_t workspace_bytes=0; Mode mode=Mode::segmented; };

template<class Input, class Acc, class Op> struct Read {
 const Input* input; uint64_t n, k, channel; Op op; bool segmented;
 __host__ __device__ Acc operator()(int index) const {
  uint64_t j=static_cast<uint64_t>(index);
  uint64_t row=segmented ? j%n : j;
  uint64_t c=segmented ? j/n : channel;
  return static_cast<Acc>(op(input[row*k+c]));
 }
};
struct Offset {
 uint64_t n;
 __host__ __device__ int operator()(int segment) const { return static_cast<int>(uint64_t(segment)*n); }
};
template<class Acc> struct Add {
 __host__ __device__ Acc operator()(const Acc& a,const Acc& b) const { return a+b; }
};

template<class Input,class Acc,class Op>
cudaError_t invoke(const Plan& p,const Input* input,Acc* output,Op op,cudaStream_t stream,void* workspace,size_t& bytes) {
 using Count = cub::CountingInputIterator<int>;
 using Values = cub::TransformInputIterator<Acc,Read<Input,Acc,Op>,Count>;
 Count count(0);
 if(p.mode==Mode::segmented) {
  Values values(count,Read<Input,Acc,Op>{input,p.n,p.k,0,op,true});
  using Offsets = cub::TransformInputIterator<int,Offset,Count>;
  Offsets offsets(count,Offset{p.n});
  return cub::DeviceSegmentedReduce::Reduce(workspace,bytes,values,output,static_cast<int>(p.k),offsets,offsets+1,Add<Acc>{},Acc{},stream);
 }
 // During a storage query a single representative channel is sufficient: its
 // iterator type and item count are identical for every channel.
 uint64_t calls=workspace ? p.k : 1;
 for(uint64_t c=0;c<calls;++c) {
  Values values(count,Read<Input,Acc,Op>{input,p.n,p.k,c,op,false});
  cudaError_t e=cub::DeviceReduce::Reduce(workspace,bytes,values,output+c,static_cast<int>(p.n),Add<Acc>{},Acc{},stream);
  if(e!=cudaSuccess) return e;
 }
 return cudaSuccess;
}

template<class Input,class Acc,class Op>
cudaError_t make_plan(uint64_t n,uint64_t k,Mode mode,const Input* input,Acc* output,Op op,cudaStream_t stream,Plan& p) {
 p=Plan{}; p.n=n; p.k=k; p.mode=mode;
 // The iterator and CUB item/segment counts use int for toolkit compatibility.
 if(k>INT_MAX || n>INT_MAX || (n && k>uint64_t(INT_MAX)/n) ||
    (k && k>SIZE_MAX/sizeof(Acc)) || (n && k>SIZE_MAX/sizeof(Input)/n)) return cudaErrorInvalidValue;
 if(!k || !n) return cudaSuccess;
 if(!input || !output) return cudaErrorInvalidValue;
 size_t bytes=0; cudaError_t e=invoke(p,input,output,op,stream,nullptr,bytes);
 if(e==cudaSuccess) p.workspace_bytes=bytes;
 return e;
}

template<class Input,class Acc,class Op>
cudaError_t execute(const Plan& p,const Input* input,Acc* output,Op op,cudaStream_t stream,void* workspace,size_t bytes) {
 if(!p.k) return cudaSuccess;
 if(!output || (p.n && !input)) return cudaErrorInvalidValue;
 if(!p.n) return cudaMemsetAsync(output,0,static_cast<size_t>(p.k)*sizeof(Acc),stream);
 // CUB interprets a null pointer as a storage query, even when size is zero.
 if(!workspace || bytes<p.workspace_bytes) return cudaErrorInvalidValue;
 size_t available=bytes;
 return invoke(p,input,output,op,stream,workspace,available);
}
}
