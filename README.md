# Interleaved column transform reduction

`reduce.cuh` provides two explicit CUDA implementations of `out[c] = sum_i AccT(op(in[i*K+c]))`. It requires CUDA Runtime, Cooperative Groups and libcu++; it does not use CUB or Thrust. The included driver tests double identity, float to double, complex-double norm to double, and complex-double identity. The supported accumulator shuffle types are `float`, `double`, and `cuda::std::complex<double>`; add an appropriate `shuffle_down` specialization for other types. Transforms must be device-callable, copyable as kernel arguments, and independent of evaluation order.

## Build and run

Install a CUDA toolkit with libcu++ and a compatible NVIDIA driver/GPU. Select an architecture supported by **your installed nvcc**: A100 `sm_80`, H100 `sm_90`, B200 `sm_100` (requires a sufficiently recent toolkit). The default is `sm_100`.

```sh
nvcc --version
nvidia-smi
make ARCH=sm_80
./reduce_demo test
./reduce_demo bench
./reduce_demo bench-fixed  # fixed 2^30 elements, K = 1..1024
compute-sanitizer --tool memcheck ./reduce_demo test-quick
compute-sanitizer --tool synccheck ./reduce_demo test-quick
```

No GPU, compiler version or benchmark results are assumed. The driver prints the runtime, driver, compiler and GPU versions. The benchmark times whole operations in one stream using preallocated memory, five warmups and 50 event-timed repetitions, excluding transfers and allocation. It reports each method in a separate CSV row, including workspace size and the custom configurations. Bandwidth uses input bytes only and is not measured DRAM traffic. Unsupported cooperative execution is reported as skipped with `N/A` values. CUB comes from the installed CUDA Toolkit/CCCL; it is not vendored.

The `bench-fixed` mode targets large-memory GPUs such as GB200 and sweeps powers of two from K=1 through 1024 with N*K fixed at 2^30. It initializes and reuses one 16 GiB device input buffer; no host-sized copy is allocated. Each implementation uses its own temporary workspace, runs the same warmups and event timing as `bench`, and checks every output against the known constant-input sum. Allocation, planning, and launch failures produce `skipped` rows with diagnostics on stderr, so other cases can continue; a correctness mismatch produces a `failed` row and a nonzero exit status. The per-channel timing includes all K CUB reductions. The standard `bench` mode is unchanged.

## Usage

```cpp
#include "reduce.cuh"
struct Scale {
  double factor;
  __device__ double operator()(float x) const { return factor * x; }
};
// Given device pointers input and output, uint64_t n, k, and cudaStream_t stream:
column_reduce::Plan plan;
cudaError_t cuda_error;
auto status = column_reduce::make_plan<float, double, Scale>(
    n, k, column_reduce::Mode::two_kernel, plan, &cuda_error);
// Check status (including cooperative_unsupported for Mode::cooperative).
void* workspace = nullptr;
if (plan.workspace_bytes) cudaMalloc(&workspace, plan.workspace_bytes);
status = column_reduce::execute(plan, input, output, Scale{2.0}, stream,
                                workspace, plan.workspace_bytes, &cuda_error);
// Check status; wait for stream completion before reading output or freeing buffers.
```

Planning queries the current device; reuse a plan only for the same input/accumulator/transform types, device, dimensions, and mode. The plan is an immutable contract after successful planning. Execution is asynchronous, uses the supplied stream, and does not allocate, transfer, or synchronize. CUDA errors from asynchronous execution can surface on later stream synchronization. `K=0` is a no-op; `N=0, K>0` writes zero through the same two phases, and input may be null. Nonempty channel sets require output and workspace; insufficient or incorrectly aligned workspace is rejected. The workspace needs `alignof(AccT)` alignment (cudaMalloc satisfies it); input, output, and workspace must not overlap. Keep all buffers alive until stream completion. Concurrent calls require separate output and workspace, unless stream dependencies serialize them. The caller must ensure the input allocation covers `N*K` elements and output covers `K`.

## Mapping and synchronization

A 256-thread block has eight warps. Each task handles one channel tile of up to 32 and one row partition. Within a warp, `width` is the next power of two at least as large as the tile, capped at 32; lane modulo width selects a channel and lane divided by width selects a row. Warps cover different rows. For large K, adjacent lanes access adjacent channels for coalescing; for K=1, all lanes cooperate on rows; intermediate K trades some lane utilization for contiguous channel accesses. This is a general compromise, not a claim of optimality for every K.

Each thread accumulates in registers using a row grid-stride loop. Shuffle steps of `width` combine only matching channels; the first row group writes each warp's result to shared memory and the first warp combines eight warp sums. Full warps and uniform block barriers are required: the API always launches exactly 256 threads. Padding contributes zero and never invokes the transform. Complex shuffles explicitly move real and imaginary components.

The workspace contains `P*K` accumulator values in `[partition][channel]` order, with `P = max(1, min(32, ceil(N/256)))`. Blocks grid-stride over `ceil(K/32)*P` tasks, so each partial is written on every call, including empty rows. The second phase reads these partials directly and grid-strides over channel tiles. Two-kernel mode always launches both kernels for K>0. Cooperative mode uses the same phases in one kernel, with one unconditional grid synchronization between them. Its one-dimensional physical grid is limited by occupancy of the instantiated kernel and the device multiprocessor count; both modes use zero dynamic shared memory. No atomics are used. Fixed configurations give fixed reduction grouping, but floating point results need not match sequential sums, other configurations, or other architectures bitwise.

Tests compare against long-double CPU accumulation of the transformed values, with an absolute tolerance of `1e-12 * (1 + sum(abs(transformed values)))`, accounting for cancellation and different summation orders. They include nonzero-at-zero transforms, irregular tails, channel-specific values, alternating signs, and repeated workspace use.

## CUB comparison baselines

`cub_baseline.cuh` adds two device-wide CUB reductions without changing `reduce.cuh`. Both fuse the transform and conversion to the accumulator type into a virtual input iterator, and use an explicit addition and zero initializer, including for complex accumulators. The segmented baseline makes a virtual column-major view: index `j` reads row `j % N`, channel `j / N` from the original row-major input. Iterator-generated offsets delimit each channel, so no transpose, transformed array, or offset array is allocated. One `DeviceSegmentedReduce::Reduce` call produces all channels. The per-channel baseline uses a strided iterator and calls `DeviceReduce::Reduce` once per channel on the same stream and workspace. Empty rows are handled by asynchronously zeroing the output; zero channels are a no-op.

Planning queries CUB workspace size with the actual input/output pointers and stream; allocate that workspace before execution and keep the plan, buffers, and transform types consistent. Execution performs no allocation or host synchronization. Workspace, input, and output must remain valid until the stream completes, and simultaneous operations need separate workspace and output or stream ordering. The adapters conservatively limit `N`, `K`, and `N*K` to `INT_MAX` for compatibility with CUB's integer iterator/count interfaces; larger cases return `cudaErrorInvalidValue`. The CUDA Toolkit must provide CUB and libcu++.

The custom kernels map threads over tiles of interleaved channels. The segmented virtual view instead causes strided physical reads, and its segment count affects available parallelism. The per-channel method also reads strided data and performs K separate reductions, potentially increasing launch overhead. CUB selects internal policies according to device, type, problem size, and CCCL version. These are complete algorithms for the same operation, not isolated measurements of CUB's warp reduction. Differences can reflect coalescing, launch counts, and parallelism as well as arithmetic; a slower segmented result does not establish that CUB device-wide reduction is inherently slower.

The benchmark prints GPU, CUDA and CUB versions as comment lines, then long-format CSV. `cub_calls` counts CUB API invocations per complete operation. Speedup is CUB latency divided by custom latency; values above one favor the custom method. Speedups appear on custom rows, and `N/A` denotes an inapplicable or skipped value. For example, with placeholders rather than measurements:

```text
N,K,input_bytes,implementation,status,latency_us,input_GB_s,workspace_bytes,cub_calls,blocks,block_size,partitions,speedup_vs_cub_segmented,speedup_vs_cub_per_channel
64,2,2048,custom_two_kernel,ok,<us>,<GB/s>,<bytes>,0,<blocks>,256,<partitions>,<ratio>,<ratio>
64,2,2048,custom_cooperative,skipped,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A
64,2,2048,cub_segmented,ok,<us>,<GB/s>,<bytes>,1,N/A,N/A,N/A,N/A,N/A
64,2,2048,cub_per_channel,ok,<us>,<GB/s>,<bytes>,2,N/A,N/A,N/A,N/A,N/A
```
