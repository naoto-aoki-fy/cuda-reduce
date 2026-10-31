#include "reduce.cuh"
#include "cub_baseline.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <string>
#include <chrono>
#include <cerrno>
#include <climits>
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
 cub_baseline::Plan cp[2];
 for(int m=0;m<2;++m) { CUDA(cub_baseline::make_plan(n,k,m?cub_baseline::Mode::per_channel:cub_baseline::Mode::segmented,di,dout,op,stream,cp[m])); bytes=std::max(bytes,cp[m].workspace_bytes); }
 if(bytes) CUDA(cudaMalloc(&ws,bytes));
 for(int repeat=0;repeat<2;++repeat) {
  std::fill(ref_re.begin(),ref_re.end(),0); std::fill(ref_im.begin(),ref_im.end(),0); std::fill(scale.begin(),scale.end(),0);
  for(uint64_t i=0;i<n;++i) for(uint64_t c=0;c<k;++c) { double x=((i%2)?-1.:1.)*(1.0+(c%17)*0.17)+double((i*13+c*7+repeat*11)%101)*0.001; h[i*k+c]=value<Input>(x); Acc y=static_cast<Acc>(op(h[i*k+c])); ref_re[c]+=real_part(y); ref_im[c]+=imag_part(y); scale[c]+=magnitude(y); }
  if(count) CUDA(cudaMemcpyAsync(di,h.data(),count*sizeof(Input),cudaMemcpyHostToDevice,stream));
  for(int m=0;m<4;++m) { if(m<2) { if(k && !plans[m].blocks) continue; cudaError_t err; auto s=execute(plans[m],di,dout,op,stream,ws,bytes,&err); if(s!=Status::success) { fprintf(stderr,"execute: %s (%s)\n",status_string(s),cudaGetErrorString(err)); exit(1); } } else CUDA(cub_baseline::execute(cp[m-2],di,dout,op,stream,ws,bytes)); if(k) CUDA(cudaMemcpyAsync(out.data(),dout,k*sizeof(Acc),cudaMemcpyDeviceToHost,stream)); CUDA(cudaStreamSynchronize(stream));
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
 using namespace column_reduce;
 int device,runtime,driver; cudaDeviceProp prop{}; CUDA(cudaGetDevice(&device)); CUDA(cudaGetDeviceProperties(&prop,device)); CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
 printf("# GPU: %s; CUDA runtime: %d; driver: %d; nvcc: %d.%d.%d; CUB: %d\n",prop.name,runtime,driver,__CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__,CUB_VERSION);
 puts("# Input=complex<double>, accumulator=double, transform=norm; bandwidth is input-effective GB/s");
 puts("N,K,input_bytes,implementation,status,latency_us,input_GB_s,workspace_bytes,cub_calls,blocks,block_size,partitions,speedup_vs_cub_segmented,speedup_vs_cub_per_channel");
 cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
 const char* names[]={"custom_two_kernel","custom_cooperative","cub_segmented","cub_per_channel"};
 for(uint64_t n: {uint64_t(64),uint64_t(4096),uint64_t(262144)}) for(uint64_t k: {uint64_t(2),uint64_t(33),uint64_t(257)}) {
  size_t count=static_cast<size_t>(n*k), input_bytes=count*sizeof(C); std::vector<C> h(count,C(0.5,0.25)); C* in; double* out; void* ws=nullptr;
  CUDA(cudaMalloc(&in,input_bytes)); CUDA(cudaMalloc(&out,k*sizeof(double))); CUDA(cudaMemcpyAsync(in,h.data(),input_bytes,cudaMemcpyHostToDevice,stream));
  Plan p[2]; auto a=make_plan<C,double,Norm>(n,k,Mode::two_kernel,p[0]); auto b=make_plan<C,double,Norm>(n,k,Mode::cooperative,p[1]);
  if(a!=Status::success || (b!=Status::success && b!=Status::cooperative_unsupported)) { fprintf(stderr,"custom planning failed\n"); exit(1); }
  cub_baseline::Plan cp[2]; CUDA(cub_baseline::make_plan(n,k,cub_baseline::Mode::segmented,in,out,Norm{},stream,cp[0])); CUDA(cub_baseline::make_plan(n,k,cub_baseline::Mode::per_channel,in,out,Norm{},stream,cp[1]));
  size_t bytes=std::max(std::max(p[0].workspace_bytes,p[1].workspace_bytes),std::max(cp[0].workspace_bytes,cp[1].workspace_bytes));
  if(bytes) CUDA(cudaMalloc(&ws,bytes)); double us[4]={};
  for(int m=0;m<4;++m) {
   if(m==1 && b==Status::cooperative_unsupported) continue;
   auto run=[&]() { if(m<2) { cudaError_t err; auto status=execute(p[m],in,out,Norm{},stream,ws,bytes,&err); if(status!=Status::success) { fprintf(stderr,"execute: %s (%s)\n",status_string(status),cudaGetErrorString(err)); exit(1); } } else CUDA(cub_baseline::execute(cp[m-2],in,out,Norm{},stream,ws,bytes)); };
   for(int i=0;i<5;++i) run();
   cudaEvent_t start,end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end)); CUDA(cudaEventRecord(start,stream));
   for(int i=0;i<50;++i) run();
   CUDA(cudaEventRecord(end,stream)); CUDA(cudaEventSynchronize(end)); CUDA(cudaGetLastError()); float ms; CUDA(cudaEventElapsedTime(&ms,start,end)); us[m]=ms*1000/50; CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
  }
  for(int m=0;m<4;++m) {
   printf("%llu,%llu,%zu,%s,",(unsigned long long)n,(unsigned long long)k,input_bytes,names[m]);
   if(!us[m]) { puts("skipped,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"); continue; }
   printf("ok,%.3f,%.3f,%zu,",us[m],input_bytes/us[m]/1000,m<2?p[m].workspace_bytes:cp[m-2].workspace_bytes);
   if(m<2) printf("0,%u,%u,%llu,",p[m].blocks,block_size,(unsigned long long)p[m].partitions); else printf("%llu,N/A,N/A,N/A,",(unsigned long long)(m==2?1:k));
   if(m<2 && us[2]) printf("%.3f,",us[2]/us[m]); else printf("N/A,");
   if(m<2 && us[3]) printf("%.3f\n",us[3]/us[m]); else puts("N/A");
  }
  if(ws) CUDA(cudaFree(ws)); CUDA(cudaFree(in)); CUDA(cudaFree(out));
 }
 CUDA(cudaStreamDestroy(stream));
}
// Initialize once on the device so the fixed-size sweep needs no host input copy.
__global__ void initialize_fixed(C* input, uint64_t count) {
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 uint64_t stride=uint64_t(blockDim.x)*gridDim.x;
 for(;i<count;i+=stride) input[i]=C(0.5,0.25);
}

struct FixedOptions {
 uint64_t max_cub_k=32;
 int max_reps=3;
 double target_ms=0, budget_seconds=0;
 bool quick=false;
};

bool benchmark_fixed(const FixedOptions& opt) {
 using namespace column_reduce;
 // Quick mode keeps the same sweep, with a smaller input and one repetition by default.
 const uint64_t count=uint64_t(1)<<(opt.quick?24:30);
 const size_t input_bytes=static_cast<size_t>(count)*sizeof(C);
 const char* names[]={"custom_two_kernel","custom_cooperative","cub_segmented","cub_per_channel"};
 const int order[]={2,3,0,1}; // Baselines must be available when each custom row is emitted.
 const int warmups=opt.quick?0:1;
 auto begin=std::chrono::steady_clock::now();
 auto budget_expired=[&]() { return opt.budget_seconds>0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()>=opt.budget_seconds; };
 int device,runtime,driver; cudaDeviceProp prop{};
 CUDA(cudaGetDevice(&device)); CUDA(cudaGetDeviceProperties(&prop,device)); CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
 fprintf(stderr,"GPU: %s; CUDA runtime: %d; driver: %d; nvcc: %d.%d.%d; CUB: %d\n",prop.name,runtime,driver,__CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__,CUB_VERSION);
 fprintf(stderr,"Input=complex<double>, accumulator=double, transform=norm; bandwidth is input-effective GB/s\n");
 puts("N,K,input_bytes,implementation,status,reason,latency_us,input_GB_s,workspace_bytes,cub_calls,blocks,block_size,partitions,timed_repetitions,speedup_vs_cub_segmented,speedup_vs_cub_per_channel");
 fflush(stdout);
 cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
 bool success=true;
 fprintf(stderr,"Initializing %zu bytes of input for bench-fixed\n",input_bytes);
 fflush(stderr);
 C* in=nullptr; double* out=nullptr;
 cudaError_t setup=cudaMalloc(&in,input_bytes);
 if(setup==cudaSuccess) setup=cudaMalloc(&out,1024*sizeof(double));
 if(setup==cudaSuccess) {
  initialize_fixed<<<std::min(65535,prop.maxGridSize[0]),256,0,stream>>>(in,count);
  setup=cudaGetLastError();
  if(setup==cudaSuccess) setup=cudaStreamSynchronize(stream);
 }
 if(setup!=cudaSuccess) { fprintf(stderr,"fixed sweep input setup: %s\n",cudaGetErrorString(setup)); success=false; }
 for(uint64_t k=1;k<=1024;k*=2) {
  uint64_t n=count/k; double us[4]={};
  for(int m:order) {
   const char* reason=nullptr; const char* status="ok";
   if(m==3 && k>opt.max_cub_k) { status="skipped"; reason="too_many_cub_calls"; }
   else if(budget_expired()) { status="skipped"; reason="time_budget_exceeded"; }
   else if(setup!=cudaSuccess) { status="failed"; reason="input_setup_failed"; }
   fprintf(stderr,"N=%llu K=%llu implementation=%s cub_calls=%llu warmups=%d timed_repetitions=%s%d decision=%s%s%s\n",
    (unsigned long long)n,(unsigned long long)k,names[m],(unsigned long long)(m==2?1:m==3?k:0),warmups,opt.target_ms>0?"up_to_":"",opt.max_reps,
    reason?status:"run",reason?" reason=":"",reason?reason:"");
   fflush(stderr);
   Plan plan{}; cub_baseline::Plan cp{}; size_t bytes=0; void* ws=nullptr;
   cudaEvent_t start=nullptr,end=nullptr; cudaError_t err=cudaSuccess; int reps=0; double total_ms=0;
   if(!reason) {
    if(m<2) {
     auto st=make_plan<C,double,Norm>(n,k,m?Mode::cooperative:Mode::two_kernel,plan,&err);
     if(st==Status::cooperative_unsupported) { status="skipped"; reason="cooperative_unsupported"; }
     else if(st!=Status::success) { status="failed"; reason="planning_failed"; fprintf(stderr,"%s planning: %s (%s)\n",names[m],status_string(st),cudaGetErrorString(err)); }
     bytes=plan.workspace_bytes;
    } else {
     err=cub_baseline::make_plan(n,k,m==2?cub_baseline::Mode::segmented:cub_baseline::Mode::per_channel,in,out,Norm{},stream,cp);
     if(err!=cudaSuccess) { status="failed"; reason="planning_failed"; }
     bytes=cp.workspace_bytes;
    }
   }
   if(!reason) err=cudaMalloc(&ws,std::max<size_t>(bytes,1));
   auto run=[&]() -> cudaError_t {
    if(m<2) { cudaError_t e=cudaSuccess; auto st=execute(plan,in,out,Norm{},stream,ws,bytes,&e); return st==Status::success?cudaSuccess:(e==cudaSuccess?cudaErrorInvalidValue:e); }
    return cub_baseline::execute(cp,in,out,Norm{},stream,ws,bytes);
   };
   for(int i=0;!reason && i<warmups && err==cudaSuccess;++i) {
    if(budget_expired()) { status="skipped"; reason="time_budget_exceeded"; break; }
    err=run(); if(err==cudaSuccess) err=cudaStreamSynchronize(stream);
   }
   if(!reason && err==cudaSuccess) err=cudaEventCreate(&start);
   if(!reason && err==cudaSuccess) err=cudaEventCreate(&end);
   while(!reason && err==cudaSuccess && reps<opt.max_reps) {
    if(budget_expired()) { if(!reps) { status="skipped"; reason="time_budget_exceeded"; } break; }
    err=cudaEventRecord(start,stream);
    if(err==cudaSuccess) err=run();
    if(err==cudaSuccess) err=cudaEventRecord(end,stream);
    if(err==cudaSuccess) err=cudaEventSynchronize(end);
    float ms=0; if(err==cudaSuccess) err=cudaEventElapsedTime(&ms,start,end);
    if(err==cudaSuccess) { total_ms+=ms; ++reps; }
    if(opt.target_ms>0 && total_ms>=opt.target_ms) break;
   }
   std::vector<double> result(k);
   if(!reason && err==cudaSuccess) err=cudaMemcpyAsync(result.data(),out,k*sizeof(double),cudaMemcpyDeviceToHost,stream);
   if(!reason && err==cudaSuccess) err=cudaStreamSynchronize(stream);
   if(!reason && err==cudaSuccess) for(double x:result) if(!std::isfinite(x) || std::abs(x-double(n)*0.3125)>1e-12*(1+double(n)*0.3125)) { status="failed"; reason="correctness_failed"; break; }
   if(err!=cudaSuccess) { status="failed"; reason="cuda_error"; fprintf(stderr,"N=%llu K=%llu %s: %s\n",(unsigned long long)n,(unsigned long long)k,names[m],cudaGetErrorString(err)); }
   // Drain any work submitted before an error before releasing its resources.
   cudaError_t drain=cudaStreamSynchronize(stream);
   if(drain!=cudaSuccess && !reason) { status="failed"; reason="cuda_error"; fprintf(stderr,"%s synchronization: %s\n",names[m],cudaGetErrorString(drain)); }
   if(start) cudaEventDestroy(start); if(end) cudaEventDestroy(end); if(ws) cudaFree(ws);
   if(status[0]=='f') success=false;
   printf("%llu,%llu,%zu,%s,%s,%s,",(unsigned long long)n,(unsigned long long)k,input_bytes,names[m],status,reason?reason:"");
   if(reason) printf("N/A,N/A,N/A,N/A,N/A,N/A,N/A,%d,N/A,N/A\n",reps);
   else {
    us[m]=total_ms*1000/reps;
    printf("%.3f,%.3f,%zu,",us[m],input_bytes/us[m]/1000,bytes);
    if(m<2) printf("0,%u,%u,%llu,",plan.blocks,block_size,(unsigned long long)plan.partitions);
    else printf("%llu,N/A,N/A,N/A,",(unsigned long long)(m==2?1:k));
    printf("%d,",reps);
    if(m<2 && us[2]>0) printf("%.3f,",us[2]/us[m]); else printf("N/A,");
    if(m<2 && us[3]>0) printf("%.3f\n",us[3]/us[m]); else puts("N/A");
   }
   fflush(stdout);
  }
 }
 if(in) cudaFree(in); if(out) cudaFree(out); CUDA(cudaStreamDestroy(stream));
 return success;
}

bool parse_positive(const char* s, double& value) {
 char* end=nullptr; errno=0; value=strtod(s,&end);
 return *s && *end=='\0' && errno!=ERANGE && std::isfinite(value) && value>0;
}
int main(int argc,char** argv) {
 if(argc<2) { fprintf(stderr,"usage: %s test|test-quick|bench|bench-fixed [--quick] [--max-cub-k K] [--max-reps N] [--target-ms MS] [--budget-seconds S]\n",argv[0]); return 2; }
 std::string mode=argv[1];
 if(mode=="bench-fixed") {
  FixedOptions opt; bool reps_set=false;
  for(int i=2;i<argc;++i) {
   std::string arg=argv[i];
   if(arg=="--quick") { opt.quick=true; continue; }
   if((arg=="--max-cub-k" || arg=="--max-reps" || arg=="--target-ms" || arg=="--budget-seconds") && i+1<argc) {
    double v; if(!parse_positive(argv[++i],v)) { fprintf(stderr,"invalid value for %s\n",arg.c_str()); return 2; }
    if(arg=="--max-cub-k") { if(v>1024 || v!=floor(v)) { fprintf(stderr,"max-cub-k must be an integer from 1 to 1024\n"); return 2; } opt.max_cub_k=static_cast<uint64_t>(v); }
    else if(arg=="--max-reps") { if(v>INT_MAX || v!=floor(v)) { fprintf(stderr,"max-reps must be a positive integer\n"); return 2; } opt.max_reps=static_cast<int>(v); reps_set=true; }
    else if(arg=="--target-ms") opt.target_ms=v;
    else opt.budget_seconds=v;
   } else { fprintf(stderr,"unknown or incomplete option: %s\n",arg.c_str()); return 2; }
  }
  if(opt.quick && !reps_set) opt.max_reps=1;
  return benchmark_fixed(opt)?0:1;
 }
 if(argc!=2) { fprintf(stderr,"options are only supported for bench-fixed\n"); return 2; }
 if(mode=="test") tests(false); else if(mode=="test-quick") tests(true); else if(mode=="bench") benchmark(); else { fprintf(stderr,"unknown mode: %s\n",mode.c_str()); return 2; }
}
