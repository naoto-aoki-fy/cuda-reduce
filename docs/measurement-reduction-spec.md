# Measurement reduction shared contract

This contract applies across all six implementation steps. Step 1 provides only configuration, input generation, ownership, fixtures, and the CPU reference.

### Mathematical operation

- The input is a read-only device array of `cuda::std::complex<double>`.
- Let `N = 2^q`, where `q` is the number of state qubits.
- Let `k = num_measure_qubits`, with `1 <= k <= min(8, q)`.
- Let `B = 2^k`.
- The ordered list `positions[0..k-1]` contains distinct state-index bit positions in `[0, q)`.
- `positions[t]` maps to bit `t` of the output bin. In particular, the first supplied position maps to the least significant bin bit.
- For state index `J`, define:

  `bin(J) = OR over t of (((J >> positions[t]) & 1ULL) << t)`

- Compute one double-precision output per bin:

  `output[b] = sum of norm(a[J]) over all J with bin(J) == b`

- Here, `norm(a[J])` means `real(a[J])^2 + imag(a[J])^2`.
- Do not normalize the state, sample a measurement outcome, or collapse the state.

### Runtime parameters and indexing

- `num_measure_qubits` and the measured positions must remain runtime parameters.
- Do not instantiate templates or dispatch to kernels specialized on the measurement count or bin count.
- Templates for storage policies, layouts, or CUB block sizes are permitted.
- Use 64-bit arithmetic for state sizes, global thread IDs, strides, indices, offsets, and workspace element counts.
- Promote operands before multiplication. Check host-side arithmetic for overflow before allocating memory or constructing launch dimensions.
- Validate positions before shifting. Reject duplicate or out-of-range positions.

### Required implementations

The completed project will provide:

- Independent custom reductions using either two reduction kernels or a cooperative kernel with `grid.sync()`.
- Per-thread accumulator storage in registers, shared memory, or preallocated global memory.
- Thread-major and bin-major layouts for shared and global storage.
- Separate CUB BlockReduce and DeviceSegmentedReduce baselines.

The custom implementations must not call CUB reduction primitives.

### Atomic policy: strict

No device-side atomic addition is permitted in the executed measurement/reduction path, including device routines called through libraries.

This includes `atomicAdd`, atomic fetch-add wrappers, equivalent generated atomic-add instructions, and CAS-based emulation of atomic accumulation.

Audit the selected CUB algorithms and the implementation of cooperative synchronization. Do not silently exempt library internals. If a required vendor implementation violates this policy, record the affected configuration as `unsupported` with evidence and an explicit reason.

Do not replace a required `grid.sync()` with a custom spin barrier to work around this restriction.

### Launch and benchmark contract

- For custom reduction kernels, use exactly `G = SM_count * sm_multiplier` blocks, including the second reduction kernel.
- Default `sm_multiplier`: 8.
- Default block size: 256; make it configurable and document supported values.
- Do not silently change the requested block count, block size, storage backend, or problem size.
- The full benchmark input is `N = 2^31` complex doubles: 32 GiB for the state vector alone.
- Benchmark measurement counts are 1 through 8.
- The standard contiguous position sets begin at:
  - Low: `0`
  - Middle: `floor((q - k) / 2)`
  - High: `q - k`
- Positions within these standard sets are listed in ascending order.
- Benchmark multipliers are `{1, 2, 4, 8, 16}`.
- Measured repetitions are configurable, defaulting to 1.
- Warm-up count is configurable; use and document a default of 3.
- Required per-invocation initialization and all reduction work belong inside the timed operation.
- Allocation and random input generation remain outside timing.

## Step 1 configuration and ownership

The CLI accepts value pairs. `--positions` is a comma-separated, ordered list;
its length is the runtime `num_measure_qubits`. There is no separate count that
can disagree with the supplied list. Defaults:

| Option | Default | Constraints |
| --- | --- | --- |
| `--qubits` | 10 | 1 through 63, subject to checked allocation size |
| `--positions` | `0` | 1 through min(8,q) distinct positions in [0,q) |
| `--block-size` | 256 | 128, 256, 512 |
| `--sm-multiplier` | 8 | Positive; benchmark sweep remains {1,2,4,8,16} |
| `--repetitions` | 1 | Positive |
| `--warmups` | 3 | Zero or greater |
| `--seed` | 20261031 | Unsigned 64-bit integer |
| `--chunk-elements` | 65536 | Positive; CPU validation chunk capacity |

Only decimal unsigned integer syntax is accepted. Unknown options, missing
values, empty position tokens, duplicates and overflow are errors. Zero warmups
is allowed. The small default is for validation; the full benchmark remains
q=31 and is not run in this step. Block size, multiplier, warmups and repetitions
are stored for later steps; the Step 1 driver performs input generation and CPU
validation once and reports GPU reductions as `not_run`.

`checked_add`, `checked_mul`, `power_of_two`, and `allocation_bytes` check
arithmetic before allocation. `launch_blocks` computes exactly SM_count times
sm_multiplier and rejects a result beyond the CUDA grid-x limit (INT_MAX).
Future launch sites must also check device-specific block and grid limits,
shared memory limits, and cooperative residency, reporting unsupported requests
without changing the settings. `workspace_elements` promotes all three
operands to the shared uint64_t size type and checks both products. For example,
q=31 means 2147483648 elements and 34359738368 state bytes; these calculations
never pass through signed 32-bit integers.

CUDA and cuRAND calls throw descriptive errors on failure. The move-only
`DeviceBuffer` and scoped generator release owned resources during both normal
return and exception unwinding. Destructors do not throw; cleanup return codes
are ignored. Device synchronization at the end of generation checks asynchronous
execution errors before the state is returned. There are no reduction kernels,
reduction-library calls, or cooperative synchronization implementations in Step 1.
The strict atomic audit must be completed for each selected reduction and
synchronization implementation in later steps.

## Reproducible input

The host cuRAND API uses `CURAND_RNG_PSEUDO_PHILOX4_32_10`,
`CURAND_ORDERING_PSEUDO_DEFAULT`, the configured seed, and initial offset zero.
`curandGenerateUniformDouble` supplies double-precision uniform draws in (0,1].
The generator persists across calls while constructing a state. Generation
chunks contain 2^20 amplitudes, except the final smaller chunk. Each call
requests twice the amplitude count. Since N is a power of two with q>=1,
requests contain a multiple of four doubles, including small inputs.

Within each chunk, draw 2*j becomes the real component of state[offset+j],
and draw 2*j+1 becomes its imaginary component. A mapping kernel always uses
256 blocks and 256 threads, with 64-bit grid-stride indexing. These generation
settings are independent of the reduction block size, SM multiplier, measured
positions, repetitions, warmups, and CPU chunk size. Changing them does not
regenerate or modify an already constructed state. The driver generates one
read-only device state, then passes that same allocation to validation. Future
methods must reuse that allocation.

Repeatability means identical state bytes for the same q, seed, generator
configuration, cuRAND version and device environment. Cross-version or
cross-device bitwise reproducibility is not promised. Changing q can change the
cuRAND call sequence; prefix equivalence across different q is not promised.
Generation, temporary draw allocation, and state allocation are outside all
future reduction timing regions. A full-size run needs the 32 GiB state plus
up to 16 MiB of temporary device draws, before any reduction workspace.

## Independent CPU reference and fixtures

`CpuReference` accepts contiguous chunks with absolute 64-bit offsets and
rejects gaps, overlap, null nonempty chunks, overflow, out-of-state ranges, and
premature finalization. Each component is promoted to long double before
squaring; Neumaier compensated accumulation is performed independently per bin.
Compensation remains in use on platforms where long double is not wider than
double. Input values are expected to be finite, as in the generated inputs.
Neither normalization nor collapse is performed.

`reference_device_state` copies bounded chunks from the existing device state.
It uses an explicit component conversion rather than assuming ABI compatibility
between CUDA and host complex types. Host memory is two arrays of at most
min(chunk_elements,N) complex values, plus O(B) reference sums. There is no
second full-size host state. Chunk size affects host memory and transfer count,
not the mathematical mapping or accumulation order.

`deterministic_fixtures()` exposes q=3 fixtures and analytic expected results:

| Fixture | Ordered positions | Nonzero amplitudes (index: value) | Expected bins |
| --- | --- | --- | --- |
| Zeros | [2,0] | None | [0,0,0,0] |
| Single | [2,0] | 5: 3+4i | [0,0,0,25] |
| Multiple | [2,0] | 0: 1; 1: 2i; 2: 3; 4: 3+4i; 5: 2+3i; 7: 1+i | [10,25,4,15] |
| Reversed | [0,2] | Same as multiple | [10,4,25,15] |

## Result representation

`Result` defaults to `not_run`. Status meanings are:

- `success`: the requested operation completed successfully.
- `unsupported`: the requested configuration cannot be executed under the
  contract; the reason must explain the restriction and include evidence for
  vendor atomic-policy violations.
- `failure`: an attempted operation failed, including invalid input or runtime errors.
- `not_run`: the operation was not attempted; this is never a successful measurement.

Elapsed time and DRAM profiling bytes use optional values. An absent value means
unavailable; zero is reserved for an actually observed zero. Future metrics must
follow the same rule. No profiling values are collected by the Step 1 driver.

## Step 2: global-memory independent reduction

`GlobalReduction` implements both global accumulator layouts with two custom
kernels, each launched with exactly G blocks and the configured 128, 256 or 512
threads. Measurement count and positions remain runtime parameters. Thread g
initializes its B accumulators and traverses J=g+T*j once. Each bin is then
reduced within the block by full-mask warp shuffles and a shared-memory reduction
of warp totals. These supported block sizes contain only full warps; every thread,
including threads without input, participates in the collectives. A trailing
block barrier protects shared scratch reuse across bins.

Partials always use `partials[bin*G+block_id]`, independently of accumulator
layout. The second kernel owns bins via `bin=blockIdx.x; bin<B; bin+=G`, distributes
the G partials across its threads and uses the same block reduction. Same-stream
kernel ordering supplies the inter-kernel dependency. All partials and outputs
are overwritten before consumption, so they need no separate clear operation.

Per-workspace device allocations are 8*T*B accumulator bytes, 8*G*B partial bytes,
8*B output bytes and k*sizeof(unsigned) position bytes. Both kernels separately
use 128 shared scratch bytes per block. All allocation products are checked and
device/kernel launch limits are validated; no requested setting is adjusted.
CUDA allocation exhaustion and launch resource limits produce unsupported status;
invalid input, arithmetic overflow, execution errors and output mismatches remain
failures. Workspace construction errors throw; the driver classifies them.

The reduction source uses no atomic operations, CUB, or device library reduction
routines. Synchronization consists only of warp shuffles, block barriers and
same-stream kernel ordering; no cooperative or spin barrier is present. Source
audit is complete; generated device code has not been audited without nvcc.

`run` returns average CUDA-event milliseconds across requested repetitions, after
the requested warmups. Initialization and both kernel launches are inside the
timed operation. Workspace allocation, host output allocation, event creation,
input generation, position upload and output download are outside timing. CPU
comparison uses |GPU-reference| <= 1e-12 + 1e-11*|reference| and rejects nonfinite
GPU values. Profiling bytes remain unavailable. No full benchmark is run in
Step 2; see README for validation commands and environment limitations.

## Step 3: shared storage and register evidence

The existing two-stage reduction, runtime bin extraction, exact G blocks per
kernel, finalization, timing, and strict atomic policy also apply to Step 3.
`GlobalReduction(config, layout, storage)` preserves the original global default.
Shared first-stage accumulators are block-local, with thread ID t and block
size S: thread-major `sums[t*B+bin]`, bin-major `sums[bin*S+t]`. Every invocation
initializes all B doubles per thread and reads each assigned amplitude once.
Partials and output are overwritten as in Step 2. Global accumulator allocation
is zero for shared and register storage; other workspace products remain checked.

The shared first stage uses `8*S*B` dynamic bytes plus the exact kernel's queried
static allocation, currently 128 scratch bytes. Dynamic storage is declared as
extern shared doubles, so its base and length have eight-byte alignment with no
additional padding. Query device default/opt-in shared limits and both exact
kernel attributes. When the request exceeds the default and fits the device
opt-in budget, request cudaFuncAttributeMaxDynamicSharedMemorySize as necessary
before rejection. Do not lower an already larger kernel allowance: previously
constructed workspaces must remain usable. Reject over-limit requests with
requested dynamic bytes, static bytes, default/opt-in limits, and the applicable
kernel dynamic limit. Check exact kernel max threads and occupancy for both
stages; zero active blocks is unsupported. Record registers/thread, local
bytes/thread, shared bytes and theoretical occupancy. No silent adaptation is
allowed. Shared support is device/build dependent, per the README byte table.

The register candidate has eight named scalar double accumulators and supports
runtime k=1..3 within its fixed capacity; no measurement-specific instantiation
or dispatch exists. Layout is not applicable. Runtime k=4..8 is unsupported.
Scalar source, zero reported spills, and kernel localSizeBytes alone are not
proof of register residency. Each actual binary and architecture requires ptxas
resource/spill reports and PTX/SASS analysis tracing the bin accumulators and
all relevant local operations, including called functions. Record compiler
version, flags, architecture, source/binary hashes and per-kernel findings.
Spilling or direct local bin allocation violates this backend's contract.

`make cuda-evidence CUDA_ARCH=sm_XX` captures evidence for a separate driver
build with the requested architecture and retained intermediates. Capture is
not certification. Until reviewed evidence is bound to the actual build, every
in-capacity register request is unverified and rejected without benchmarking.
There are currently no certified register architectures; follow-up certification
must add the build-bound gate and correctness tests before permitting timing.

GPU tests preserve all Step 2 cases and add supported shared layouts against CPU
and global output, repeated reuse with changed input, constructor resource
rejection with requested-byte diagnostics, and unverified/capacity register
rejection. Representative sanitizer invocations use the existing sanitize target.
Actual host/GPU results and the environment's support matrix are recorded in
README. No cooperative launch or CUB implementation is part of Step 3.

## Step 4: gated cooperative reduction

`ReductionMethod::cooperative` selects one combined kernel per storage backend;
global/shared layouts remain runtime choices. Two-kernel execution remains the
default. Shared device routines implement initialization, 64-bit traversal,
custom block partials and unique block-stride final bin ownership in both paths.
The combined kernel performs one unconditional `this_grid().sync()` between
partial writes and finalization, including threads/blocks without useful work.
Its launch API is `cudaLaunchCooperativeKernel`, with exactly the requested
`G=SM_count*sm_multiplier` blocks and supported block size.

`reduction_resources(config, storage, method)` inspects the exact combined kernel
for cooperative requests, checks device support and launch limits, applies
shared-memory opt-in, and queries occupancy using its actual dynamic allocation.
The maximum resident grid is `active_blocks_per_SM*SM_count`; G above that bound
is rejected on the host. The resource result is an eligibility calculation, not
an atomic-policy or register certification. Constructor gates run before device
buffer allocation, and eligibility is rechecked before cooperative enqueue.
Register qualification of the two-stage kernel never qualifies the combined
kernel; each needs independent evidence.

The strict atomic policy is unchanged. No synchronization-only exception is
introduced. All cooperative constructors currently reject execution: examined
vendor CUDA 12.9.79 synchronization uses atomic addition, and no build-bound
atomic-free synchronization certification exists. The source inspection,
package/header hashes and unavailable generated-code evidence are recorded in
`docs/cooperative-sync-audit.md`. In-capacity register requests also remain
unverified for the combined kernel; larger counts exceed capacity.

The driver reports each method separately and contains CPU and matching
two-kernel comparisons for executable cooperative outputs. Evidence capture
retains the selected vendor preprocessing as well as PTX/SASS. Tests were
explicitly skipped for Step 4; no runtime correctness or numerical residency
results are claimed in the absence of CUDA tools/hardware. CUB and the complete
benchmark matrix remain deferred.


## Step 5: separate CUB baselines

`BlockPrimitive::custom` remains the default and instantiates only the independent
custom reduction primitive. `BlockPrimitive::cub` instantiates matched CUB kernels
at S=128/256/512, preserving runtime measurement count, positions, accumulator
storage/layout and two-kernel/cooperative structure. Resource checks inspect the
actual selected CUB kernels. Register residency and cooperative synchronization
qualification remain independent gates.

`SegmentedReduction` executes `DeviceSegmentedReduce::Sum` on the original state
through a signed 64-bit random-access gather/norm iterator and signed 64-bit
segment-offset iterator. It prepares no N-element array and queries reusable
storage outside timing. The driver labels custom tuning parameters n/a and runs
this method once. Installed CUB versions are accepted when the required APIs
compile, without a version whitelist or backend substitution. The strict
atomic-free requirement remains unchanged; the selected build's audit status is
unverified until its implementation is reviewed. Historical 2.8.2 source evidence
does not certify arbitrary installed headers or generated code. Baseline definitions, offset/atomic call-path
evidence, version identification, unsupported cases and unrun validation are in
[the Step 5 report](cub-baselines.md). Tests were skipped by user instruction;
no full benchmark matrix was executed.


## Step 6: benchmark execution and reporting

The full preset, per-active-workspace memory preflight, raw per-invocation event
samples, multi-configuration chunked CPU references, fixed-tolerance bin/total
validation, resource fields and JSON/CSV schema are defined in
[the benchmark guide](benchmark.md). It also records the exact Make commands,
separate metric-discovering profiling workflow and actual environment evidence.
The mathematical and strict atomic contracts above remain unchanged. Tests were
explicitly skipped for this implementation; CUDA execution remains unverified.
