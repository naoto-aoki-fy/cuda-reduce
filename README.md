# Quantum measurement reduction

Steps 1–6 implement configuration, deterministic cuRAND input, a bounded CPU
reference, and independent two-kernel reductions with per-thread global-memory
accumulators in thread-major and bin-major layouts, both shared-memory layouts,
and a fixed-capacity register candidate. The six-step contract is in
[docs/measurement-reduction-spec.md](docs/measurement-reduction-spec.md).
Step 6 adds the complete benchmark matrix, per-repetition event timing, full-input
CPU validation, JSON/CSV output and separate profiling tools; see
[benchmark commands, result definitions and execution evidence](docs/benchmark.md).
Step 4 adds gated cooperative kernels. Step 5 adds matched CUB BlockReduce and
lazy-gather DeviceSegmentedReduce baselines; see [definitions, source audit and
results](docs/cub-baselines.md).

Build with GNU Make and a C++17 compiler:

```sh
make host
```

This builds `build/libmeasurement_core.a` without CUDA dependencies. With a CUDA
toolkit providing nvcc, libcudart, cuRAND, and `<cuda/std/complex>`:

```sh
make cuda
./build/measurement_validate --qubits 10 --positions 7,1 --seed 20261031 --chunk-elements 128
```

`make` always builds the host library and also builds the CUDA driver when nvcc
is found. `CXX`, `CXXFLAGS`, `NVCC`, and `NVCCFLAGS` may be overridden; for example,
`make cuda NVCCFLAGS='-O2 -arch=sm_80'`. CUDA execution requires an appropriate GPU.
The driver reuses one generated state for all methods, prints CPU reference bins,
and checks each GPU output with absolute tolerance 1e-12 plus relative tolerance
1e-11. It reports status and CUDA-event mean milliseconds per invocation after
the configured warmups (default 3) and repetitions (default 1). Timing includes
per-thread accumulator initialization and both kernels; allocation, position
upload, input generation, CPU reference, and result copying are outside timing. Profiling bytes remain
unavailable. Runtime errors and numerical mismatches are failures; device/kernel
resource limits and CUDA allocation exhaustion are explicit unsupported results.

Supported block sizes are **128, 256, 512**, validated explicitly. Both kernels
launch exactly `G = SM_count * sm_multiplier` blocks on the same stream. Each
thread reads its grid-stride input elements once. Bin extraction uses the runtime
ordered positions. Reductions use full-warp shuffles followed by shared scratch,
with barriers before scratch reuse. The two-kernel reduction path uses no library reductions or atomics.

For global storage, `T = G * block_size`, `B = 2^k`, reusable device workspace is:

- Accumulators: `8*T*B` bytes, `sums[g*B+bin]` or `sums[bin*T+g]`.
- Partials: `8*G*B` bytes, always `partials[bin*G+block]`.
- Output: `8*B` bytes; positions: `k*sizeof(unsigned)` bytes.
- Separate shared scratch: 16 doubles (128 bytes) per block in either kernel.

Every invocation initializes all accumulators and overwrites all partials and
outputs. `GlobalReduction` retains its original global default; its optional
`AccumulatorStorage` argument selects shared or register storage. It owns reusable
buffers; `execute` enqueues the two kernels and `run` performs warmups, timing, synchronization and result download.
Use a workspace serially on its construction device. Simultaneous operations
need separate workspaces. The state allocation is read-only during reduction.
All buffer products are checked before workspace allocation. Workspace excludes
the state (`16*N` bytes), generation temporaries, and CUDA runtime/event overhead.

Validation commands:

```sh
make test-host
make test-cuda
make sanitize
```

The GPU suite checks analytic fixtures, all k=1..8, all supported block sizes,
both layouts, arbitrary/reversed positions, N smaller than T, grid-stride tails,
and three invocations per workspace (NaN poison, reuse, then zero input). It also
checks generated input with warmups and multiple repetitions on the same buffers.
Transfers and reduction are tested with a nonblocking stream. The sanitizer
target runs representative small cases through memcheck, synccheck and racecheck;
an unsupported GPU test exits 77 and is not counted as a pass.

Host checks pass in the implementation environment. nvcc, nvidia-smi and
Compute Sanitizer are unavailable, so CUDA compilation, GPU correctness, timings,
and sanitizer results remain unverified. No 32 GiB benchmark was run.

Step 3 shared storage allocates no global accumulator buffer. Each thread owns
B double accumulators at `sums[t*B+bin]` or `sums[bin*S+t]`, initialized on every
invocation. Dynamic shared bytes are `8*S*B`, already aligned to eight bytes;
the exact kernel's static scratch (currently 128 bytes) is added for limit
checks. The constructor queries the selected device, both kernel attributes,
and occupancy, requesting dynamic shared opt-in before rejecting a feasible
request. Diagnostics include requested bytes and device/kernel limits. No block
size adjustment, tiling, backend substitution or extra state pass is performed.
The driver prints registers/thread, local bytes/thread, static/dynamic shared
bytes, maximum kernel threads and theoretical occupancy for each stage.

Shared dynamic requirements in KiB (add actual static scratch):

| Block size | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| 256 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| 512 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |

The register candidate uses eight named double scalars, independent of runtime
k, and runtime switches for accumulation and reduction. Its capacity is eight
bins (k=1..3), with layout **not applicable**; k=4..8 is unsupported by capacity.
Scalar source alone does not establish residency. All k=1..3 register requests
are currently rejected as **residency unverified**, before allocation or timing.
There is deliberately no runtime switch to override this evidence gate.

Capture evidence for an explicit architecture on a CUDA host, for example:

```sh
make cuda-evidence CUDA_ARCH=sm_80 NVCCFLAGS='-O2'
```

This builds a separate actual driver under `build/evidence/sm_80/`, retaining
compiler intermediates, verbose ptxas resource/spill output, PTX, SASS, resource
usage, compiler/tool versions, source hashes, full command/flags, and binary
hash in a manifest. It does not certify residency or enable register timing.
Audit `register_first_stage` in every embedded architecture: trace all eight
accumulators through initialization, input traversal, bin updates and block
reduction. Inspect local-address operations in PTX and local loads/stores in
SASS, including called device functions. Distinguish any unrelated local use
from bin storage; zero spills alone is insufficient. Record the findings against
the manifest/binary hash. Spilled or directly local accumulators are unsupported;
missing or ambiguous evidence remains unverified. Enabling a verified kernel
requires a reviewed, build/architecture-bound evidence gate and GPU correctness
and resource tests. No architecture is certified by this change.

Step 3 validation status in this environment (2026-10-31):

| Method | Requested configurations | Result here |
| --- | --- | --- |
| Global, both layouts | S=128/256/512, k=1..8 | CUDA unavailable; preserved implementation |
| Shared, both layouts | S=128/256/512, k=1..8 | CUDA unavailable; device support unverified |
| Registers, layout n/a | S=128/256/512, k=1..3 | Residency unverified; timing disabled |
| Registers, layout n/a | k=4..8 | Unsupported: exceeds eight-bin capacity |

`make test-host` passed. `make test-cuda` and `make sanitize` could not build
because nvcc is absent. Evidence capture also rejected missing nvcc. The GPU model could not be
identified with the available tools; no CUDA execution was possible, so there
is no measured device support matrix, compiler/SASS evidence, or sanitizer result. The GPU suite now compares supported shared cases to both
the CPU and existing global results, checks rejection diagnostics and constructor
rejection, and repeats dirty-workspace and changed-input cases. Quick sanitizer
cases include k=1,3,6,8, covering small storage, capacity boundary, opt-in candidates
and oversized requests on typical devices. Unsupported shared configurations
are printed explicitly and skipped; supported runtime failures fail the suite.

Step 4 adds `ReductionMethod::cooperative` as the optional fourth constructor
argument to `GlobalReduction`; the default remains `two_kernel`. The driver
reports both methods for each backend/layout. Cooperative kernels share the
custom accumulation and block reduction routines with the two-kernel path,
write all bin-major partials, call `cooperative_groups::this_grid().sync()` once,
then finalize with `bin=blockIdx.x; bin<B; bin+=G`. Every thread reaches the grid
collective, including idle input threads and blocks owning no final bins.
Measurement count and ordered positions remain runtime inputs.

Before allocation or launch, eligibility checks query cooperative device support,
apply shared-memory opt-in to the actual combined kernel, query its attributes
and occupancy with the requested block size and dynamic bytes, and calculate:

- Requested blocks: `G = SM_count * sm_multiplier`.
- Resident block limit: `R = SM_count * active_blocks_per_SM(combined kernel, S, dynamic_bytes)`.
- Eligible residency: `G <= R`; oversized requests throw `Unsupported` on the host.

Diagnostics print G, SM count, R, and the combined kernel's resource data.
No grid/block adjustment or backend fallback is made. The enqueue path uses
`cudaLaunchCooperativeKernel` with exactly G blocks and S threads. Register
qualification is independent for the combined kernel and remains unverified;
k=4..8 still exceeds its fixed capacity. The existing event timing and workspace
ownership apply, with all cooperative reduction work in one launch.

**All cooperative execution is currently disabled.** The specification retains
its strict atomic policy with no synchronization exception. Inspection of
NVIDIA CUDA runtime 12.9.79 shows atomic addition inside `grid.sync()`, conflicting
with that policy. Other builds have no reviewed atomic-free synchronization
certification. See [the synchronization audit](docs/cooperative-sync-audit.md)
for package/header hashes, call path, source locations, and generated-code limits.
`make cuda-evidence CUDA_ARCH=sm_XX` now also retains the preprocessed reduction
source, including the selected vendor synchronization implementation. Evidence
capture cannot enable execution. If a future build is certified, the driver
compares each executable cooperative output with both the CPU reference and the
matching successful two-kernel output before reporting success.

Step 4 results in this environment (2026-10-31):

| Combination | Support / observed result |
| --- | --- |
| Two-kernel global/shared, both layouts | Preserved; CUDA build/runtime unverified here |
| Cooperative global/shared, both layouts, k=1..8 | Unsupported: strict synchronization audit gate; device limits may reject earlier |
| Cooperative registers, k=1..3, layout n/a | Unsupported: combined residency unverified and synchronization policy gate |
| Registers, either method, k=4..8 | Unsupported: eight-bin capacity |

`make host` succeeds (library already current). `make cuda` stops because nvcc
is unavailable. Tests and sanitizers were skipped at the user's request,
including small-input, repeated-run, arbitrary-position and oversized-residency
checks. No actual correctness results, numeric device eligibility calculations,
PTX/SASS findings, or timings can be claimed for this step. No known-invalid
cooperative launch was attempted and no full benchmark matrix was added.

Step 5 uses `make cuda` with the installed toolkit's CCCL/CUB headers or headers
selected through `CPPFLAGS`. There is no CUB version whitelist: versions including
3.3.4 can execute when the required APIs compile. Incompatible APIs produce compiler
errors without selecting a fallback. The driver reports actual CUB, available CCCL,
CUDA header/compiler versions and selected algorithms, then compares both baselines
with the same CPU reference. GPU architecture and build flags are recorded in
benchmark metadata. The strict atomic-free requirement remains in force; the
selected build's atomic-free audit status is **unverified**. API compatibility,
numerical validation and timing do not establish atomic-free execution. See
[the CUB policy and historical source evidence](docs/cub-baselines.md).
`BlockPrimitive::cub` selects matched BlockReduce kernels at S=128/256/512,
with independent resource inspection and register/cooperative gates.
`SegmentedReduction` runs once with custom tuning fields n/a and signed 64-bit
lazy gather and segment offsets, including the exclusive N=2^31 endpoint.

`make host` succeeds; `make cuda` cannot compile because nvcc is unavailable.
Tests were skipped at the user's request. Added mapping/boundary checks and
`make test-cub` GPU checks remain unrun. No GPU correctness, resource, timing or
sanitizer results are claimed, and no full benchmark matrix was run.


Step 6 full benchmark (32 GiB input plus active workspace):

```sh
make benchmark-full BENCHMARK_ARGS='--block-sizes 256 --repetitions 1 --warmups 3'
```

The default full preset retains all 2424 requested cases, including unsupported
ones. Results are written to `build/full.json` and `build/full.csv`. Use
`make benchmark-smoke` for a separately labelled small-input sweep. See the
[Step 6 guide](docs/benchmark.md) for validation, memory preflight, timing,
profiling, reproducible builds and limitations. Step 6 tests were skipped as
requested; `make host` succeeded and `make cuda` was blocked by missing nvcc.
No full-size run, GPU validation or timing results are available here.
