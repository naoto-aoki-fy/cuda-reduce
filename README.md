# Quantum measurement reduction foundations

Step 1 implements configuration, checked arithmetic, deterministic cuRAND input,
CUDA ownership, bounded CPU reference validation, analytic fixtures, and result
statuses. The persistent six-step contract is in
[docs/measurement-reduction-spec.md](docs/measurement-reduction-spec.md).
GPU reduction backends and benchmarks are not implemented yet.

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
The driver prints CPU reference bins, reports GPU reductions as `not_run`, and
reports profiling metrics as unavailable. It does not time any reductions.

Tests were skipped at the user's request. No full benchmark was run. The current
environment provides g++ and Make, but nvcc and nvidia-smi are not on PATH;
CUDA compilation, repeatable GPU generation, and GPU execution are unverified.
The deterministic fixture API is available for subsequent validation. Boundary
checks can be exercised through the checked arithmetic API without allocating a
large state.
