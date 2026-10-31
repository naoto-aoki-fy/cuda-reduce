# Quantum measurement reduction

Steps 1–3 implement configuration, deterministic cuRAND input, a bounded CPU
reference, and independent two-kernel reductions with per-thread global-memory
accumulators in thread-major and bin-major layouts, both shared-memory layouts,
and a fixed-capacity register candidate. The six-step contract is in
[docs/measurement-reduction-spec.md](docs/measurement-reduction-spec.md).
Cooperative reductions and CUB baselines remain deferred.

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
with barriers before scratch reuse. No reduction path calls libraries or atomics.

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
