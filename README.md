# Interleaved column transform reduction

`reduce.cuh` provides two explicit CUDA implementations of `out[c] = sum_i AccT(op(in[i*K+c]))`. It requires CUDA Runtime, Cooperative Groups and libcu++; it does not use CUB or Thrust. The included driver tests double identity, float to double, complex-double norm to double, and complex-double identity. The supported accumulator shuffle types are `float`, `double`, and `cuda::std::complex<double>`; add an appropriate `shuffle_down` specialization for other types. Transforms must be device-callable, copyable as kernel arguments, and independent of evaluation order.

## Build and run

Install a CUDA toolkit with libcu++ and a compatible NVIDIA driver/GPU. Select an architecture supported by **your installed nvcc**: A100 `sm_80`, H100 `sm_90`, B200 `sm_100` (requires a sufficiently recent toolkit). The default is `sm_80`.

```sh
nvcc --version
nvidia-smi
make ARCH=sm_80
./reduce_demo test
./reduce_demo bench
compute-sanitizer --tool memcheck ./reduce_demo test-quick
compute-sanitizer --tool synccheck ./reduce_demo test-quick
```

No GPU, compiler version or benchmark results are assumed. The driver prints the runtime, driver, compiler and GPU versions. The benchmark times whole operations in one stream using preallocated memory, five warmups and 50 event-timed repetitions, excluding transfers and allocation. It prints both configurations and the ratio of two-kernel time to cooperative time. Its bandwidth uses input bytes only and is not measured DRAM traffic. The logical partitions match; physical grid sizes can differ. Unsupported cooperative execution is reported in the test/benchmark output as skipped (benchmark values are zero).

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
