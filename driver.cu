#include "reduce.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <string>
using C = cuda::std::complex<double>;
#define CUDA(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1); } } while(0)
struct Identity { template<class T> __host__ __device__ T operator()(T x) const { return x; } };
struct Norm { __host__ __device__ double operator()(C x) const { return cuda::std::norm(x); } };
struct Shift { double shift; __host__ __device__ double operator()(double x) const { return x+shift; } };
template<class T> T value(double x) { return T(x); }
template<> C value<C>(double x) { return C(x,x*0.37+0.125); }
template<class T> long double magnitude(T x) { return fabsl(static_cast<long double>(x)); }
template<> long double magnitude(C x) { return hypotl(x.real(),x.imag()); }
template<class T> long double real_part(T x) { return static_cast<long double>(x); }
template<> long double real_part(C x) { return x.real(); }
template<class T> long double imag_part(T) { return 0; }
template<> long double imag_part(C x) { return x.imag(); }
template<class Input,class Acc,class Op> void check_case(uint64_t n,uint64_t k,Op op) {
 using namespace column_reduce;
 Input* di=nullptr; Acc* dout=nullptr; void* ws=nullptr; cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
 size_t count=static_cast<size_t>(n*k); std::vector<Input> h(count); std::vector<Acc> out(k); std::vector<long double> ref_re(k),ref_im(k),scale(k);
 if(count) CUDA(cudaMalloc(&di,count*sizeof(Input))); if(k) CUDA(cudaMalloc(&dout,k*sizeof(Acc)));
 Plan plans[2]; size_t bytes=0;
 for(int m=0;m<2;++m) { auto s=make_plan<Input,Acc,Op>(n,k,m?Mode::cooperative:Mode::two_kernel,plans[m]); if(s==Status::cooperative_unsupported) continue; if(s!=Status::success) { fprintf(stderr,"plan: %s\n",status_string(s)); exit(1); } bytes=std::max(bytes,plans[m].workspace_bytes); }
 if(bytes) CUDA(cudaMalloc(&ws,bytes));
 for(int repeat=0;repeat<2;++repeat) {
  std::fill(ref_re.begin(),ref_re.end(),0); std::fill(ref_im.begin(),ref_im.end(),0); std::fill(scale.begin(),scale.end(),0);
  for(uint64_t i=0;i<n;++i) for(uint64_t c=0;c<k;++c) { double x=((i%2)?-1.:1.)*(1.0+(c%17)*0.17)+double((i*13+c*7+repeat*11)%101)*0.001; h[i*k+c]=value<Input>(x); Acc y=static_cast<Acc>(op(h[i*k+c])); ref_re[c]+=real_part(y); ref_im[c]+=imag_part(y); scale[c]+=magnitude(y); }
  if(count) CUDA(cudaMemcpyAsync(di,h.data(),count*sizeof(Input),cudaMemcpyHostToDevice,stream));
  for(int m=0;m<2;++m) { if(k && !plans[m].blocks) continue; cudaError_t err; auto s=execute(plans[m],di,dout,op,stream,ws,bytes,&err); if(s!=Status::success) { fprintf(stderr,"execute: %s (%s)\n",status_string(s),cudaGetErrorString(err)); exit(1); } if(k) CUDA(cudaMemcpyAsync(out.data(),dout,k*sizeof(Acc),cudaMemcpyDeviceToHost,stream)); CUDA(cudaStreamSynchronize(stream));
   for(uint64_t c=0;c<k;++c) { long double tolerance=1e-12L*(1+scale[c]); long double error=hypotl(real_part(out[c])-ref_re[c],imag_part(out[c])-ref_im[c]); if(error>tolerance) { fprintf(stderr,"FAIL n=%llu k=%llu c=%llu mode=%d error=%Lg tolerance=%Lg\n",(unsigned long long)n,(unsigned long long)k,(unsigned long long)c,m,error,tolerance); exit(1); } }
  }
 }
 if(ws) CUDA(cudaFree(ws)); if(di) CUDA(cudaFree(di)); if(dout) CUDA(cudaFree(dout)); CUDA(cudaStreamDestroy(stream));
}
void tests(bool quick) {
 const uint64_t ks[]={0,1,2,3,4,5,7,16,31,32,33,65,257}; const uint64_t ns[]={0,1,7,259,4097};
 for(auto k:ks) for(auto n:ns) { if(quick && n>259) continue; check_case<double,double>(n,k,Identity{}); check_case<float,double>(n,k,Identity{}); check_case<C,double>(n,k,Norm{}); check_case<C,C>(n,k,Identity{}); check_case<double,double>(n,k,Shift{2.5}); }
 puts("Correctness tests passed (unsupported cooperative launches skipped).");
}
void benchmark() {
 using namespace column_reduce; int device; cudaDeviceProp prop{}; int runtime,driver; CUDA(cudaGetDevice(&device)); CUDA(cudaGetDeviceProperties(&prop,device)); CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
 printf("GPU: %s; CUDA runtime: %d; driver: %d; nvcc: %d.%d.%d\n",prop.name,runtime,driver,__CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__);
 puts("Input=complex<double>, accumulator=double, transform=norm; bandwidth is input-effective, not DRAM bandwidth");
 puts("N,K,input_bytes,two_blocks,coop_blocks,block_size,partitions,workspace_bytes,two_us,coop_us,two_GB_s,coop_GB_s,two_over_coop");
 cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
 for(uint64_t n: {uint64_t(64),uint64_t(4096),uint64_t(262144)}) for(uint64_t k: {uint64_t(2),uint64_t(33),uint64_t(257)}) {
  size_t count=n*k; std::vector<C> h(count,C(0.5,0.25)); C* in; double* out; void* ws; CUDA(cudaMalloc(&in,count*sizeof(C))); CUDA(cudaMalloc(&out,k*sizeof(double))); CUDA(cudaMemcpyAsync(in,h.data(),count*sizeof(C),cudaMemcpyHostToDevice,stream));
  Plan p[2]; auto a=make_plan<C,double,Norm>(n,k,Mode::two_kernel,p[0]); auto b=make_plan<C,double,Norm>(n,k,Mode::cooperative,p[1]); if(a!=Status::success || (b!=Status::success && b!=Status::cooperative_unsupported)) { fprintf(stderr,"planning failed\n"); exit(1); }
  size_t bytes=std::max(p[0].workspace_bytes,p[1].workspace_bytes); CUDA(cudaMalloc(&ws,bytes)); double us[2]={0,0};
  for(int m=0;m<2;++m) { if(m && b==Status::cooperative_unsupported) continue; for(int i=0;i<5;++i) if(execute(p[m],in,out,Norm{},stream,ws,bytes)!=Status::success) exit(1); cudaEvent_t start,end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end)); CUDA(cudaEventRecord(start,stream)); for(int i=0;i<50;++i) if(execute(p[m],in,out,Norm{},stream,ws,bytes)!=Status::success) exit(1); CUDA(cudaEventRecord(end,stream)); CUDA(cudaEventSynchronize(end)); float ms; CUDA(cudaEventElapsedTime(&ms,start,end)); us[m]=ms*1000/50; CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end)); }
  printf("%llu,%llu,%zu,%u,%u,%u,%llu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n",(unsigned long long)n,(unsigned long long)k,count*sizeof(C),p[0].blocks,p[1].blocks,block_size,(unsigned long long)p[0].partitions,bytes,us[0],us[1],count*sizeof(C)/us[0]/1000,us[1]?count*sizeof(C)/us[1]/1000:0,us[1]?us[0]/us[1]:0);
  CUDA(cudaFree(ws)); CUDA(cudaFree(in)); CUDA(cudaFree(out));
 }
 CUDA(cudaStreamDestroy(stream));
}
int main(int argc,char** argv) { if(argc!=2) { fprintf(stderr,"usage: %s test|test-quick|bench\n",argv[0]); return 2; } std::string mode=argv[1]; if(mode=="test") tests(false); else if(mode=="test-quick") tests(true); else if(mode=="bench") benchmark(); else return 2; }
