# Step 5: CUB baseline definitions and audit

## Version-independent compatibility policy

The installed CUDA/CCCL toolchain supplies CUB (or headers selected through
`CPPFLAGS`). There is no version whitelist or minimum-version gate. CUB 3.3.4
and other releases can compile and execute both baselines when the required
public APIs are compatible. API incompatibilities fail compilation directly;
there is no replacement with custom kernels or silent version-based test skip.
Resource limits and register residency gates remain independent.

Runtime diagnostics and benchmark metadata report actual CUB, CCCL when its
version header/macro is available, CUDA header and compiler versions. Existing
build metadata records compiler flags, and device metadata and kernel attributes
record GPU compute capability and binary/PTX architectures.

The strict atomic-free requirement forbids atomic addition and atomic-based
accumulation, including inside library calls. The selected build's atomic-free
audit status is **unverified**; removing the version gate is not certification.
These properties require separate evidence:

- API compatibility: compilation against the selected headers.
- Numerical correctness: CPU-reference comparisons and CUDA error checks,
  including retained 64-bit mapping/offset boundary regressions.
- Atomic-free implementation verification: review of the installed headers,
  selected policies, generated kernels and called functions for each architecture.
- Performance: actual event timings under the unchanged benchmark methodology.

The historical source inspection below concerns only NVIDIA's unmodified
[CCCL v2.8.2 sources](https://github.com/NVIDIA/cccl/tree/v2.8.2/cub/cub)
(`CUB_VERSION=200802`). It does not certify installed headers, other releases,
patched vendor distributions or generated device code. Internal policy and
one-CTA-per-segment findings below apply only to those inspected sources;
the installed DeviceSegmentedReduce policy is left to CUB.

## A: matched BlockReduce pipeline

Select `GlobalReduction(config, layout, storage, BlockPrimitive::cub)`.
The default primitive remains custom. Accumulation/traversal bodies are shared
source with a compile-time reduction policy: S=0 instantiates only the custom
shuffle/barrier primitive; S=32/64/128/256/512/1024 instantiates only
`cub::BlockReduce<double,S,cub::BLOCK_REDUCE_WARP_REDUCTIONS>::Sum`.
No measurement count or bin count is a template argument.

For each storage/layout, input norm calculation, ordered runtime positions,
per-thread accumulators, layouts, G=SM_count*sm_multiplier, partial layout and
final bin ownership match the custom implementation exactly. Global/shared
storage preserves both layouts; the eight named register scalars have layout n/a
and runtime k=1..3 capacity. There is no substitution or size adjustment.
Both reduction stages use CUB. Every bin reduction ends with a block barrier
before temporary storage can be reused. The two kernels launch on the same
stream, whose execution order supplies the dependency between stages.

Resource inspection selects the actual CUB kernel instantiation for the requested
S and storage: static shared bytes, registers, local bytes, maximum block
size and occupancy come from that kernel, including the final stage. Shared
accumulators still require 8*S*B dynamic bytes, in addition to the queried CUB
static storage; no custom scratch-size assumption is used. Dynamic opt-in and
per-kernel occupancy are recomputed from these values. Workspace is otherwise
8*T*B global accumulator bytes when applicable, 8*G*B partial bytes, 8*B output
bytes and k*sizeof(unsigned) position bytes; T=G*S.

Register qualification remains independently unverified for these CUB-containing
kernels. All register execution is gated; k>3 reports capacity exceeded.
Eligibility queries may reject device/resource limits before the residency gate.

## B: DeviceSegmentedReduce

`SegmentedReduction` calls the public `cub::DeviceSegmentedReduce::Sum` overload
with lazy random-access input and offset iterators. Their index/difference types
are signed int64_t, and segment offset values are int64_t. Only the segment count
B<=256 uses int. The CUB-owned launch configuration is left unchanged. Storage,
layout, grid-finalization mode, block size and sm_multiplier are **not applicable**;
the driver runs this baseline once for the selected state and positions.

With L=N/B, logical v yields bin=v/L and remaining=v%L. Bin bit t is deposited
at positions[t]; successive remaining bits fill unmeasured positions in ascending
order. The input iterator dereferences the original state at that index and
computes real*real+imag*imag in double. Segment b spans [b*L,(b+1)*L).
This is a bijection because the two disjoint sets of bit positions reconstruct
all q input bits, with the supplied order determining bin-bit significance.
No N-element mapping, norm or reorder buffer exists. Only a small by-value
position/mask descriptor is prepared before timing. Norm and index mapping are
executed by CUB's reduction kernel on every input read.

Temporary storage is queried and allocated before timing, then reused on the
caller's stream. Each call overwrites all B outputs. Both baselines use CUDA
events after warmups and average over repetitions, including initialization and
all reduction kernels. Allocation, descriptor/position preparation, input
creation, output download and independent compensated CPU reference are outside
timing. Mismatches and nonfinite results are failures with elapsed time cleared.

## Historical 2.8.2 offset and atomic source audit

All file paths here are relative to the linked upstream `cub/cub/` tree.

- `device/device_segmented_reduce.cuh:389-422`: Sum derives OffsetT from the
  common **value type** of the begin/end iterators. Our two offset iterators
  therefore select int64_t; Sum supplies addition and a zero initial value.
- `device/dispatch/dispatch_reduce.cuh:148-191`: DeviceSegmentedReduceKernel
  loads both endpoints into OffsetT, passes OffsetT to AgentReduce, and assigns
  one segment to each block. Thread zero stores the completed aggregate.
  `DispatchSegmentedReduce::InvokePasses` at 941-995 launches B blocks, with
  no cross-block accumulation, counters or atomic synchronization. Its storage
  query returns one byte in this release; the implementation still uses the API
  query rather than assuming that size.
- `agent/agent_reduce.cuh:359-450` carries OffsetT through ConsumeRange and
  tile offsets. `grid/grid_even_share.cuh:204-209` retains OffsetT endpoints and
  tile stride. Only a final partial-tile count, strictly below TILE_ITEMS, is
  narrowed to int; absolute offsets and the exclusive endpoint never are.
  Consequently N=2147483648 remains representable throughout this selected
  path. There is no signed-32-bit range limit imposed by this Sum overload.
  N beyond INT64_MAX is explicitly unsupported by our signed iterator contract;
  ordinary Config/allocation limits may reject such requests earlier.
- `device/dispatch/tuning/tuning_reduce.cuh:200-290` selects
  BLOCK_REDUCE_WARP_REDUCTIONS for every policy (300/350/600/1000);
  SegmentedReducePolicy aliases ReducePolicy. Architecture and type select
  CUB's launch shape, not a custom tuning label.
- `agent/agent_reduce.cuh:166-168` enables vectorized loads only for native
  pointer inputs, so this lazy iterator takes the scalar iterator path with
  64-bit indexing. `device/dispatch/kernels/reduce.cuh:89-92` combines the
  initial value and block aggregate and stores the result directly.
- `block/specializations/block_reduce_warp_reductions.cuh` uses warp reductions,
  shared warp aggregates, a block barrier and thread-zero aggregation.
  `warp/specializations/warp_reduce_shfl.cuh:329-362` implements double addition
  with shuffle and add instructions. Neither selected baseline's reduction path
  performs atomic addition or CAS accumulation. The nondeterministic atomic
  BlockReduce algorithm is never selected.

This establishes upstream 2.8.2 **source** compatibility and atomic policy for the
inspected algorithms; it is not a successful CUDA build or device-code audit,
and does not verify the selected build.
`make cuda-evidence CUDA_ARCH=sm_XX` retains CUB preprocessing,
compiler/ptxas output, resource usage, PTX/SASS and source/binary hashes for a
future actual build. Audit every selected kernel and called function, including
all eight register accumulators, before changing the register residency gate.

## Results and unsupported cases

| Item | Actual result in this environment |
| --- | --- |
| `make host` | Succeeds; existing host library current |
| `make cuda` | Cannot compile: nvcc unavailable (Make error 127) |
| Host mapping/offset tests | Added; not run at user request |
| GPU CPU-reference comparisons, repeated reuse, stream checks | Added; not run at user request |
| Sanitizers, CUDA timings, occupancy numbers, PTX/SASS | Not run / unavailable |
| Any installed CUB, including 3.3.4 | No version rejection; API compatibility, GPU correctness and atomic-free execution unverified here |
| CUB registers, k=1..3 | Unsupported: actual CUB kernel residency unverified |
| CUB registers, k=4..8 | Unsupported: eight-double capacity |
| CUB shared or launch resource excess | Unsupported with actual requested/kernel/device limit diagnostics |
| Signed iterator endpoint >INT64_MAX | Unsupported; no narrowing or N adjustment |

The added host checks enumerate q=1..8 and all supported k, low/middle/high,
noncontiguous and reversed positions; they check coverage, bin ordering and
per-bin sums against CpuReference. Arithmetic checks around 2^31 use synthetic
iterators and q=31/32 descriptors without state allocation. `make test-cub`
adds small q=3/8 cases for every k, both layouts and all supported block sizes,
nonblocking streams, warmups/repetitions and changed-input workspace reuse;
resource and policy exclusions are printed. `make test-cuda` includes it.
These are pending checks, not reported passes. No full benchmark matrix ran.

## Inspected upstream file hashes

SHA-256 hashes identify downloaded source evidence only; no vendor source is
vendored into the project.

```text
49fee4e41576d175824cbf98c39eb64b3dfda94ed1830223035073c48282de8e  version.cuh
fb882e89d64ccca62c429869410f89f9293beb9a327c99e40b18702c4f686843  device_segmented_reduce.cuh
66815a5fec15beb0dcb74538455918db724a5d20cbf292bb194b80a18daca48e  device_dispatch_dispatch_reduce.cuh
89a35825f293cfb43b0c283497381eca3fcb857e269a40bfc05476622644a3f1  device_dispatch_tuning_tuning_reduce.cuh
1f9e7e05f58db9be8ef8b94b41a81db5e892e134e98f9c152364b6d6276e1051  device_dispatch_kernels_reduce.cuh
326ee290d6dabeed3768f331314b36c23d165d6c07b7bff68cc1d6c51d5a780e  agent_agent_reduce.cuh
9df3bad2e0759b27dfe6691a8c51de105691ff350fbbf00aea9af87d0948ae1c  grid_grid_even_share.cuh
965f778629276cae6015f308119284a3a58b843125d7a8cf86d6d54dbe5024ea  block_block_reduce.cuh
d9fcdfe35341774a595c1c88b3ed49fd7a79fb19d2b29a41a28ca71473f609d3  block_specializations_block_reduce_warp_reductions.cuh
0cefb0f0b4b951ce1bcc4239e001fe5fda81942b404b3d6e8d6b5d73da9cf3b7  warp_warp_reduce.cuh
d4b3db062dfe3879eacd59915238b772b6a1e07ea9d67a18449933ba8d14e4d1  warp_specializations_warp_reduce_shfl.cuh
```

## Version-policy change validation

Tests, including `make test-host`, `make test-cub`, `make test-cuda`, and
`make benchmark-smoke`, were skipped at the user's request for this change.
Existing host q=31/32 logical-index and offset regressions and GPU CPU-reference,
reuse, stream, resource and policy checks are retained. No CUB 3.3.4 compilation,
GPU execution, numerical pass, atomic-free certification or timing is claimed.
