# Cooperative synchronization audit, 2026-10-31

The specification's strict atomic policy applies to synchronization internals.
No exception has been authorized. Cooperative kernels are implemented but gated
before workspace allocation and timing; no atomic-free build is certified.

## Vendor source inspected

The inspected artifact is NVIDIA's `nvidia-cuda-runtime-cu12` distribution,
version **12.9.79**, obtained from its
[PyPI release](https://pypi.org/project/nvidia-cuda-runtime-cu12/12.9.79/).
This is downloaded vendor source evidence, not an installed toolkit, and does
not identify any future build's compiler, architecture or binary.

Wheel filename:
`nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`

SHA-256:
`25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3`

All paths below are relative to `nvidia/cuda_runtime/include/` in that wheel:

| File | SHA-256 |
| --- | --- |
| `cooperative_groups.h` | `cb6705c5ae9efac684140f5a5b6d99b93c22d30b9dede107abb5cddefdf42d3d` |
| `cooperative_groups/details/helpers.h` | `2bd8efc675dce7ee85ba6d4a1b811028926b559e018473920096d9478b832f47` |
| `cooperative_groups/details/sync.h` | `11de0af50acf662e3775d4aa670bf55fc346fc24ec5d4a304909dda14bfcd8b5` |

The inspected call path is `grid_group::sync()` in `cooperative_groups.h`,
then `details::grid::sync()` in `helpers.h:380-382`, then
`details::sync_grids_arrive()` and `sync_grids_wait()` in `sync.h`.

In `sync.h:68-92`, all threads enter a block barrier; the master thread of each
block updates a shared global arrival counter. The SM70-or-newer branch at
line 84 uses inline PTX `atom.add.release.gpu.u32`; the other branch at line 88
uses `atomicAdd`. This is an atomic-add instruction in vendor source, including
on the modern release/acquire path. The master block uses a phase-flipping
increment; other blocks add one. The wait routine at lines 96-109 polls the phase
with acquire loads on modern targets, uses fences on older targets, and ends
with a block barrier. These operations implement grid synchronization, but the
atomic additions still violate the specification. No internal operation is
exempted. All storage/layout combinations using this implementation are
unsupported, irrespective of accumulator atomics or cooperative residency.

## Generated-code evidence and execution gate

There is no nvcc, cuobjdump, installed CUDA header tree, or identified CUDA GPU
in this environment. No actual combined-kernel PTX/SASS could be produced or
inspected. The inline PTX above is source evidence, not a generated-code audit.
No cooperative configuration was executed or timed; tests were skipped on user
instruction. Numeric SM/residency results are unavailable.

For an actual CUDA build, use GNU Make:

```sh
make cuda-evidence CUDA_ARCH=sm_80 NVCCFLAGS='-O2'
```

The evidence tool builds a separate driver and records the source hashes,
compiler/tool versions, flags, architecture, binary hash, ptxas diagnostics,
PTX, SASS and resource usage. Step 4 also records
`cooperative-preprocessed.ii`, retaining the selected vendor implementation.
Review all three `cooperative_stage` storage instantiations and any called
device routines in each embedded architecture. Trace the executed path from
grid synchronization through the counter update; inspect atomic add/reduction
instructions and any CAS emulation, rather than treating unrelated atomic
instructions elsewhere in the binary as proof. Record symbol names and line or
instruction offsets against the manifest's binary hash. Register accumulators
need a separate combined-kernel local-memory/spill audit.

Until that review supplies an acceptable build-bound certification, constructors
reject with an explicit unsupported reason. The examined vendor source has a
known policy conflict; unknown versions remain unverified. Capturing evidence,
zero spills or passing two-stage qualification never enables cooperative timing.
No override, replacement barrier or two-kernel fallback is provided.
