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

No GPU, compiler version or benchmark results are assumed. CUB comes from the installed CUDA Toolkit/CCCL; it is not vendored. The standard `bench` mode uses five warmups and 50 event-timed repetitions.

`bench-fixed` sweeps K=1 through 1024 with N*K fixed at 2^30 complex elements (16 GiB). It initializes one device input and reuses it for every implementation. Allocation, planning, initialization, and correctness checks are outside the timed region. Each invocation is timed with CUDA events in the same stream, and synchronized before the next invocation. Bandwidth is input bytes divided by elapsed time, not measured DRAM traffic.

By default each configuration gets one warmup and three timed repetitions. The per-channel CUB baseline is skipped for K > 32 because it launches K reductions per invocation; the other implementations run where supported. Planning, allocation, execution, and correctness errors are reported as `failed`, while policy and unsupported cooperative configurations are `skipped`. Failures produce a nonzero exit status. Every row includes a reason for failures or skips and the actual timed repetition count. Progress and device information go to stderr; stdout is CSV, flushed after each row. CUB baselines run first so custom rows can include speedups; an unavailable baseline produces `N/A`.

```sh
./reduce_demo bench-fixed --quick                     # 2^24 elements, no warmup, one repetition
./reduce_demo bench-fixed --sm-multiplier 8           # target eight first-stage blocks per SM
./reduce_demo bench-fixed --max-cub-k 1024            # run all K values for every supported method
./reduce_demo bench-fixed --max-reps 10 --target-ms 500
./reduce_demo bench-fixed --budget-seconds 120 > results.csv
```

`--max-cub-k K` accepts 1 through 1024; `--max-reps N` sets the maximum timed invocations (default 3, or 1 in quick mode unless explicitly set). With `--target-ms`, measurement stops after the cumulative CUDA event time reaches the target or the maximum count is reached. Without it, the maximum count is used. Counts can differ between implementations; latency is the mean per invocation. `--budget-seconds` is a wall-clock budget beginning before device setup. It is checked between synchronized invocations and configurations; a running invocation is allowed to finish, and a partially measured configuration is reported with its actual count. Remaining configurations are skipped. The budget is therefore not a hard deadline, especially during initialization or an individual slow invocation. Quick mode reduces the input to 2^24 elements (256 MiB), so its results are not directly comparable to the normal fixed-size sweep.

The custom first phase targets eight blocks per SM by default, with enough row partitions to fill that grid when channel tiles alone do not provide sufficient work. `bench-fixed --sm-multiplier N` accepts a positive integer through `UINT_MAX` to change this target. Both benchmarks report the SM count and requested target; the CSV `blocks` column is the actual first-stage grid (the cooperative grid in cooperative mode), and `partitions` is the row partition count. At large K, both CUB baselines use strided reads, which can reduce memory coalescing. Actual performance depends on the GPU and toolkit.

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
// Optional arguments change the default multiplier of 8 and accumulator count of 1:
// make_plan<float, double, Scale>(n, k, mode, plan, &cuda_error, 8, 4);
// Check status (including cooperative_unsupported for Mode::cooperative).
void* workspace = nullptr;
if (plan.workspace_bytes) cudaMalloc(&workspace, plan.workspace_bytes);
status = column_reduce::execute(plan, input, output, Scale{2.0}, stream,
                                workspace, plan.workspace_bytes, &cuda_error);
// Check status; wait for stream completion before reading output or freeing buffers.
```

Planning queries the current device; reuse a plan only for the same input/accumulator/transform types, device, dimensions, mode, multiplier, and accumulator count. The optional `unsigned sm_multiplier` argument defaults to 8, preserving existing call sites; pass `nullptr` for the error pointer if only setting the multiplier. Zero is rejected with `invalid_argument`. The final `unsigned accumulators` argument accepts 1 (default), 2, or 4; other values are rejected, including for K=0. The selected count applies to the first stage for K=1/2/4 in both modes; other K and the final reduction use one accumulator. The plan is an immutable contract after successful planning. Execution is asynchronous, uses the supplied stream, and does not allocate, transfer, or synchronize. CUDA errors from asynchronous execution can surface on later stream synchronization. With valid multiplier and accumulator settings, `K=0` is a no-op; `N=0, K>0` writes zero through the same two phases, and input may be null. Nonempty channel sets require output and workspace; insufficient or incorrectly aligned workspace is rejected. The workspace needs `alignof(AccT)` alignment (cudaMalloc satisfies it); input, output, and workspace must not overlap. Keep all buffers alive until stream completion. Concurrent calls require separate output and workspace, unless stream dependencies serialize them. The caller must ensure the input allocation covers `N*K` elements and output covers `K`.

## Mapping and synchronization

A 256-thread block has eight warps. Each task handles one channel tile of up to 32 and one row partition. Within a warp, `width` is the next power of two at least as large as the tile, capped at 32; lane modulo width selects a channel and lane divided by width selects a row. Warps cover different rows. For large K, adjacent lanes access adjacent channels for coalescing; for K=1, all lanes cooperate on rows; intermediate K trades some lane utilization for contiguous channel accesses. This is a general compromise, not a claim of optimality for every K.

Each thread accumulates in registers using a row grid-stride loop. With 2 or 4 accumulators, an unrolled group of consecutive loop iterations adds to distinct partial sums, which are combined before the existing warp reduction. Each load is guarded at the tail, so missing rows contribute zero without invoking the transform. Host dispatch selects a compiled kernel variant; cooperative planning queries that same variant for occupancy. Increasing the count can increase registers or cause spills and can change the cooperative grid size. Benchmark before choosing a count; summation order and floating point results can change. Shuffle steps of `width` combine only matching channels; the first row group writes each warp's result to shared memory and the first warp combines eight warp sums. Full warps and uniform block barriers are required: the API always launches exactly 256 threads. Padding contributes zero and never invokes the transform. Complex shuffles explicitly move real and imaginary components.

For K=1, partition `p` receives a contiguous row range: with `q=N/P` and `rem=N%P`, its beginning is `p*q + min(p,rem)` and its length is `q + (p<rem)`. These adjacent, nonoverlapping ranges cover exactly N rows, and lengths differ by at most one. The products and sums used for boundaries are bounded by N. Threads use local row indices `threadIdx.x + j*256`; the beginning is added only for valid loads, so inactive threads cannot overflow a global row index near the integer limit. Adjacent lanes read consecutive elements, including complex-double inputs. Division and modulo for the range are outside the element loop. Empty partitions contribute zero without calling the transform. For K=2 and K=4, `ContiguousColumnReader` uses the same ranges and reads `(range.begin + local_row)*K + channel`. Each thread starts at `warp*(32/K) + lane/K` and advances by `256/K` rows. Adjacent lanes read consecutive elements across rows and channels. Both execution modes share these readers and the existing block reduction, partial layout, and final reduction. Other K retain the interleaved mapping `p + local_row*P`. Partition counts and launch planning follow the same formulas; cooperative occupancy is queried for the selected compiled kernel.

Let `T = ceil(K/32)` and let `L` be the requested `num_SMs * sm_multiplier` block count capped by the device's one-dimensional CUDA grid limit. Cooperative mode also caps `L` at `active_blocks_per_SM * num_SMs`, using occupancy of the instantiated cooperative kernel with 256 threads and zero dynamic shared memory. The workspace contains `P*K` accumulator values in `[partition][channel]` order, with `P = max(1, min(ceil(N/256), ceil(L/T)))` for K>0. This removes the old 32-partition cap, adds only enough row partitions to supply the target grid, and uses one partition when channel tiles already fill the grid. Small inputs cap row partitioning at `ceil(N/256)`; empty rows use one partition. Workspace and task sizes retain overflow checks.

The first phase launches `min(T*P, L)` blocks of exactly 256 threads, grid-striding over `T*P` tasks, so each partial is written on every call, including empty rows. For N=2^30 and K=1, two-kernel mode launches `num_SMs * 8` first-stage blocks by default when grid and work limits permit: 1216 blocks on a 152-SM GPU. Its second kernel launches `min(plan.blocks, T)` blocks to read these partials and grid-stride over channel tiles; for K=1 that is one block. Two-kernel mode always launches both kernels for K>0. Cooperative mode uses the same phases in one kernel, with one unconditional grid synchronization between them; its resident grid stays unchanged for both phases. Both modes use zero dynamic shared memory. No atomics are used. Fixed configurations give fixed reduction grouping, but floating point results need not match sequential sums, other multipliers, other configurations, or other architectures bitwise.

Tests compare against long-double CPU accumulation of the transformed values, with an absolute tolerance of `1e-12 * (1 + sum(abs(transformed values)))`, accounting for cancellation and different summation orders. They include nonzero-at-zero transforms, irregular tails, channel-specific values, alternating signs, and repeated workspace use. Planning checks in both `test` and `test-quick` cover N=2^30/K=1 without allocating the input, multipliers 1/4/7/UINT_MAX, small/empty inputs, channel tiles that already fill the grid, overflow, and cooperative occupancy limits. Full tests also check million-row inputs for K=1/2/3/4 and 65,549 rows for K=33 with multipliers 1/4/7 in both supported execution modes. `bench` includes K=1, and `bench-fixed` exercises the full N=2^30/K=1 execution (or N=2^24 in quick mode).

Partition boundary checks additionally cover N=0/1, N<P, uneven splits, and boundaries up to UINT64_MAX without allocating input. Execution cases at N=255/256/257/511/512/513/769 cover K=1/2/4 with float, double, complex norm, complex identity, and a nonzero-at-zero transform in both supported modes. These use the existing cancellation-sensitive reference checks and workspace reuse. Both test modes also exercise N=8193 with K=2/4, shifted double and complex identity inputs, and multipliers 1/4/7.

## K=1 measurement status

The supplied GB200 measurements for 2^30 complex-double elements, norm-to-double, CUDA 13.3, and multiplier 8 are reference observations, not measurements reproduced here:

| Implementation | Original K=1 latency | Contiguous K=1 latency |
|---|---:|---:|
| CUB per-channel DeviceReduce | 2.248 ms | Unchanged baseline; not rerun |
| Custom two-kernel | 4.026 ms | Not measured |
| Custom cooperative | 4.006 ms | Not measured |

The contiguous reader isolates the coalescing hypothesis. No speedup, regression result, register-pressure result, or explanation of the remaining measured gap to CUB is established yet. K=2/4 now also use contiguous row partitions; unchanged mappings for other K do not guarantee unchanged performance: compiled register usage and cooperative occupancy still need comparison. Selectable 1/2/4 accumulator variants are implemented for K=1/2/4; a separate K=1 kernel remains deferred. No accumulator variant has been measured or classified as a win or regression.

For the K=2/4 extension, `make` was attempted but could not compile because `nvcc` is unavailable in the implementation environment. CUDA correctness tests, GPU benchmarks, sanitizer runs, and Nsight profiling remain unverified here.

To compare the original revision with this change, build each checkout with the same toolkit, flags, and GPU, then run the following once per checkout, saving separate CSV files:

```sh
make ARCH=sm_100
./reduce_demo bench-fixed --sm-multiplier 8 --max-reps 20 > results.csv
```

This retains the existing timing methodology and sweeps K=1/2/4/8/16/32/64/128/256/512/1024 with 20 timed invocations per supported configuration. Compare `latency_us`, `input_GB_s`, `partitions`, and `blocks` for both custom modes; bandwidth is decimal GB/s (divide by 1000 for TB/s). Repeat sweeps to assess noise. Do not use `--quick`, `--target-ms`, or a time budget for this comparison, since those can change the input size or repetition count. Where Nsight Compute is available, profile equivalent configurations separately for global load sectors/coalescing, DRAM and L2 throughput, instructions/stalls, registers, and spills. Keep profiler timings separate from ordinary event timings.

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

## Independent accumulator comparison

`bench-fixed --accumulators 1|2|4` selects the custom first-stage variant for K=1/2/4 (default 1) and records the count on stderr. Save a separate CSV and log for each count; CUB baselines use their existing implementation. Both correctness modes cover all counts with empty inputs, row boundaries, uneven tails, channel-specific signed values, float-to-double, complex norm, complex identity, nonzero-at-zero transforms, and workspace reuse. Planning checks validate count rejection and occupancy for each compiled cooperative variant.

```sh
make -B ARCH=sm_100 NVCCFLAGS='-O3 -std=c++17 -lineinfo -Xptxas=-v' > build.log 2>&1
./reduce_demo test-quick
./reduce_demo test
for a in 1 2 4; do
  ./reduce_demo bench-fixed --accumulators "$a" --sm-multiplier 8 --max-reps 20 > "acc${a}.csv" 2> "acc${a}.log"
done
```

Compare K=1/2/4 CUDA-event times, ptxas register/spill reports, and actual cooperative grid sizes under the same GPU settings. Repeat in reverse count order to check ordering effects. These commands still run both CUB baselines for the complete K sweep. The accumulator implementation environment has no `nvcc` or GPU tools; CUDA compilation, execution, sanitizers, register/spill checks, and performance measurements remain unverified.

Host validation of the extracted accumulation loop passed for counts 1/2/4 with float, double, and complex-double sums, exact row coverage, inactive channels, tails, and UINT64_MAX row boundaries. This does not validate CUDA compilation or device behavior.
