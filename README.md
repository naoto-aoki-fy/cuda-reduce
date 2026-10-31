# Quantum measurement reduction

Steps 1–2 implement configuration, deterministic cuRAND input, a bounded CPU
reference, and independent two-kernel reductions with per-thread global-memory
accumulators in thread-major and bin-major layouts. The six-step contract is in
[docs/measurement-reduction-spec.md](docs/measurement-reduction-spec.md).
Other storage backends, cooperative reductions, and CUB baselines remain deferred.

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
The driver reuses one generated state for both layouts, prints CPU reference bins,
and checks each GPU output with absolute tolerance 1e-12 plus relative tolerance
1e-11. It reports status and CUDA-event mean milliseconds per invocation after
the configured warmups (default 3) and repetitions (default 1). Timing includes
accumulator zeroing and both kernels; allocation, position upload, input generation,
CPU reference, and result copying are outside timing. Profiling bytes remain
unavailable. Runtime errors and numerical mismatches are failures; device/kernel
resource limits and CUDA allocation exhaustion are explicit unsupported results.

Supported block sizes are **128, 256, 512**, validated explicitly. Both kernels
launch exactly `G = SM_count * sm_multiplier` blocks on the same stream. Each
thread reads its grid-stride input elements once. Bin extraction uses the runtime
ordered positions. Reductions use full-warp shuffles followed by shared scratch,
with barriers before scratch reuse. No reduction path calls libraries or atomics.

For `T = G * block_size`, `B = 2^k`, reusable device workspace is:

- Accumulators: `8*T*B` bytes, `sums[g*B+bin]` or `sums[bin*T+g]`.
- Partials: `8*G*B` bytes, always `partials[bin*G+block]`.
- Output: `8*B` bytes; positions: `k*sizeof(unsigned)` bytes.
- Separate shared scratch: 16 doubles (128 bytes) per block in either kernel.

Every invocation initializes all accumulators and overwrites all partials and
outputs. `GlobalReduction` owns reusable buffers; `execute` enqueues the two
kernels and `run` performs warmups, timing, synchronization and result download.
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
