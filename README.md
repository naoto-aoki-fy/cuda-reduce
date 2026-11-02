# Quantum measurement reduction

Steps 1–6 implement configuration, deterministic cuRAND input, a bounded CPU
reference, and independent two-kernel reductions with per-thread global-memory
accumulators in thread-major and bin-major layouts, both shared-memory layouts,
and a fixed-capacity register candidate. The six-step contract is in
[docs/measurement-reduction-spec.md](docs/measurement-reduction-spec.md).
Step 6 adds the complete benchmark matrix, per-repetition event timing, full-input
CPU validation, JSON/CSV output and separate profiling tools; see
[benchmark commands, result definitions and execution evidence](docs/benchmark.md).
Step 5 adds matched CUB BlockReduce and
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

Supported block sizes are **32, 64, 128, 256, 512, 1024**, validated explicitly. Both kernels
launch exactly `G = SM_count * sm_multiplier` blocks on the same stream. Each
thread reads its grid-stride input elements once. Bin extraction uses the runtime
ordered positions. Reductions use full-warp shuffles followed by shared scratch,
with barriers before scratch reuse. The custom two-kernel reduction path uses no library reductions or atomics.

For global storage, `T = G * block_size`, `B = 2^k`, reusable device workspace is:

- Accumulators: `8*T*B` bytes, `sums[g*B+bin]` or `sums[bin*T+g]`.
- Partials: `8*G*B` bytes, always `partials[bin*G+block]`.
- Output: `8*B` bytes; positions: `k*sizeof(unsigned)` bytes.
- Separate shared scratch: 32 doubles (256 bytes) per block in either custom kernel,
  enough for one sum from each warp of a 1024-thread block.

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
Host coverage includes k=9..16 position bands, reversed ordering, gather mapping
at q=31 and bounded CPU references. CUB GPU coverage adds segmented reduction
for k=9..16 on bounded input; register rejection checks include these counts.
Transfers and reduction are tested with a nonblocking stream. The sanitizer
target runs representative small cases through memcheck, synccheck and racecheck;
an unsupported GPU test exits 77 and is not counted as a pass.

Host checks pass in the implementation environment. nvcc, nvidia-smi and
Compute Sanitizer are unavailable, so CUDA compilation, GPU correctness, timings,
and sanitizer results remain unverified. No 32 GiB benchmark was run.

Step 3 shared storage allocates no global accumulator buffer. Each thread owns
B double accumulators at `sums[t*B+bin]` or `sums[bin*S+t]`, initialized on every
invocation. Dynamic shared bytes are `8*S*B`, already aligned to eight bytes;
the exact kernel's static scratch (currently 256 bytes for custom kernels) is added for limit
checks. The constructor queries the selected device, both kernel attributes,
and occupancy, requesting dynamic shared opt-in before rejecting a feasible
request. Diagnostics include requested bytes and device/kernel limits. No block
size adjustment, tiling, backend substitution or extra state pass is performed.
The driver prints registers/thread, local bytes/thread, static/dynamic shared
bytes, maximum kernel threads and theoretical occupancy for each stage.

Shared dynamic requirements in KiB (add actual static scratch):

| Block size | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| 64 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| 128 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| 256 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| 512 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| 1024 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |

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

Storage validation status in this environment (2026-11-02):

| Method | Requested configurations | Result here |
| --- | --- | --- |
| Global, both layouts | S=32/64/128/256/512/1024, k=1..8 | CUDA unavailable; device execution unverified |
| Shared, both layouts | S=32/64/128/256/512/1024, k=1..8 | CUDA unavailable; device support unverified |
| Registers, layout n/a | S=32/64/128/256/512/1024, k=1..3 | Residency unverified; timing disabled |
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

Custom and CUB BlockReduce reductions use two kernels exclusively. Each launches
exactly the requested G blocks on the same stream; stream ordering makes the
partials visible to the final stage. There is no grid-wide synchronization
kernel or alternative reduction strategy. The strict atomic-accumulation
prohibition and necessary block barriers remain in force.

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
`BlockPrimitive::cub` selects matched BlockReduce kernels at S=32/64/128/256/512/1024,
with independent resource inspection and a register residency gate.
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

The default full preset sweeps measured-qubit counts k=1..16 at fixed q=31 and
retains all 2448 requested cases, including unsupported ones. Results are written
to `build/full.json` and `build/full.csv`. Use
`make benchmark-smoke` for a separately labelled small-input sweep. See the
[Step 6 guide](docs/benchmark.md) for validation, memory preflight, timing,
profiling, reproducible builds and limitations. Step 6 tests were skipped as
requested; `make host` succeeded and `make cuda` was blocked by missing nvcc.
No full-size run, GPU validation or timing results are available here.

To run the same full benchmark restricted to k=1,2,3,4,5,6:

```sh
make benchmark-full-k1-6 BENCHMARK_ARGS='--block-sizes 256 --repetitions 1 --warmups 3'
```

This uses the `full-k1-6` preset with the same fixed q=31 input, position bands,
methods, tuning parameters, timing and validation as `full`. It retains all 918
requested cases by default, including unsupported ones. Results go to the separate
directory `build/full-k1-6/` as `results.json` and `results.csv`; override the prefix
with `BENCHMARK_K1_6_RESULTS=path/to/results`. The `benchmark-full` target retains its k=1..16 sweep.

To run only k=4 with the same full-size input and measurement settings:

```sh
make benchmark-full-k4 BENCHMARK_ARGS='--block-sizes 256 --repetitions 1 --warmups 3'
```

The `full-k4` preset keeps q=31, all three position bands and every method,
storage/layout choice and SM multiplier from `full`. It retains all 153 requested
cases by default, including unsupported configurations, with the same timing,
validation and reporting behavior. Results go to `build/full-k4/results.json`
and `build/full-k4/results.csv`; override the prefix with
`BENCHMARK_K4_RESULTS=path/to/results`. Existing full benchmark targets are unchanged.

Validation after removing the single-kernel synchronization variant (2026-11-02):
`make` and `make test-host` pass, including both C++ host suites and nine Python
tests. All four staged JSON plans generate successfully in dry-run mode.
`make cuda`, `make test-cuda`, `make sanitize`, and `make cuda-evidence` are
blocked by missing nvcc. CUDA compilation, GPU correctness, sanitizer checks,
and device-code audits remain unverified; register timing remains disabled.
