# Step 6: benchmark and validation

Build exclusively with Make, C++17 and a CUDA toolkit providing cuRAND and
cuda::std::complex. CUB versions, including 3.3.4, are accepted when the required
BlockReduce and DeviceSegmentedReduce APIs compile; API incompatibilities fail
compilation without a fallback. Select the real GPU architecture explicitly, for example:

```sh
make cuda NVCCFLAGS='-O2 -arch=sm_80 -lineinfo'
```

The runtime reports actual CUB, available CCCL, CUDA header/compiler, runtime/driver and cuRAND versions,
GPU name, compute capability and SM count. A generated build header records
compiler/version, full flags, Git revision, SHA-256 of sources and build tools,
and a source/flags build ID. Runtime kernel attributes also record binary and
PTX architecture versions. Build metadata is refreshed by Make and does not
rely on a clean Git tree. The generated metadata header is a dependency of all CUDA objects, so source
and flag changes rebuild those objects together. For a complete retained compiler,
PTX/SASS and binary-hash record, use the evidence target below.

## Small suite and smoke benchmark

Run the retained deterministic, randomized, reuse, mapping and boundary suites
before a production benchmark:

```sh
make test-host
make test-cuda
make sanitize
make benchmark-smoke BENCHMARK_ARGS='--qubits 10 --warmups 1 --repetitions 2'
```

Tests are explicit Make targets; the benchmark does not secretly run them.
`--preset single` (the default) preserves an arbitrary ordered `--positions`
configuration and one multiplier, running all backends and strategies. For example:

```sh
./build/measurement_validate --qubits 10 --positions 7,1 --seed 20261031 --chunk-elements 128 --results build/single
```

`--preset smoke` sweeps k=1..min(8,q), low/middle/high contiguous bands and all
five SM multipliers on a smaller input (q=10 by default). This is explicitly a
smoke workload, never a full-size measurement.

## Full benchmark

```sh
make benchmark-full BENCHMARK_ARGS='--block-sizes 128,256,512 --repetitions 5 --warmups 3 --seed 20261031 --chunk-elements 65536'
```

The full preset fixes q=31, N=2147483648, and the state alone consumes
34359738368 bytes (32 GiB). It rejects `--qubits`; it never substitutes a smaller
state. Both matrix presets own `--positions` and `--sm-multiplier` and reject
those overrides. `--block-sizes` accepts a distinct comma-separated selection
from 128,256,512; it defaults to `--block-size` (256). Measured repetitions
default to 1 and warmups to 3.

Each of 24 measurement configurations (k=1..8 times three position bands) has,
for each block size and multiplier, 20 rows: custom/CUB BlockReduce times two
strategies times five storage/layout choices (registers with layout n/a, shared
with two layouts, global with two layouts). Add one DeviceSegmentedReduce row
per measurement configuration, independent of custom tuning. The default full
matrix therefore contains **2424 rows**; three block sizes produce **7224**.
Even coincident position bands on a small smoke input retain separate labels.
Unsupported requests remain rows with explicit diagnostics. Segmented custom
fields and register layout are JSON null (not applicable).

The input is generated exactly once with the specified seed, Philox generator,
default cuRAND ordering, offset zero and fixed generation chunks. Every method
uses that same immutable allocation. Generation preflights state plus up to
16 MiB of draws. After draws are freed, each active variant preflights its
workspace against current free memory, with the state already resident:

- Global accumulators: `8 * G * S * B` bytes; shared/register: zero global accumulator bytes.
- Partials: `8 * G * B`; output: `8 * B`; positions: `4 * k`.
- Segmented: output plus the exact CUB temporary-storage query (at least one allocated temporary byte).

Rows record workspace and total live state-plus-workspace bytes where the
resource query succeeds. Oversized shared-memory or residency requests can be
rejected earlier. Only one variant's workspace is live at a time. CUDA context,
cuRAND internal allocations, events and allocator overhead are not predicted;
allocation failures remain unsupported with their diagnostics. Available memory
can change between preflight and allocation. Host validation is bounded by two
complex chunks plus O(sum of bins across reference configurations), and measured
output retention is O(repetitions * bins) for the active variant.

## Timing and numerical validation

A controlled nonblocking stream and CUDA events surround each complete
invocation. Every repetition initializes its own accumulators, reads the state,
computes norms and bin mapping, and runs both reduction kernels (or the complete
cooperative/library operation). CUB segmented input is a lazy gather/norm iterator;
no data-dependent norms or reorder buffers are prepared outside timing.
Allocation, position upload, input generation, warmups, CPU work and output
validation transfers are outside event intervals. Per-repetition synchronization
and output download happen after the stop event; the operation is then run again.
Results from previous invocations are never substituted.

Raw milliseconds, arithmetic mean, min, median and max are retained. Sample
standard deviation is null for one repetition; no confidence interval is
invented. Input-effective bandwidth is
`N * sizeof(cuda::std::complex<double>) / (mean_ms / 1000) / 1e9` GB/s.
This counts input bytes only. It is **not total DRAM bandwidth**: global
accumulators and other workspace traffic can be substantial. No estimated
workspace-throughput number is reported.

All actual device-generated amplitudes are copied in bounded host chunks once.
Independent `CpuReference` objects compute every required position reference in
the same traversal, using long-double component squaring and Neumaier compensated
bin sums. Each reference is reused across all tuning and method cases. Full-input
CPU work at q=31 can take considerable time. `--chunk-elements` bounds transfer
and host storage, not the number of validated elements.

Every bin of **every measured repetition** must satisfy
`abs(GPU - reference) <= 1e-12 + 1e-11 * abs(reference)` and be finite.
The sum of all output bins must satisfy the same absolute/relative rule against
the CPU total norm (sum of compensated reference bins). Maximum absolute error
and maximum relative error are reported across repetitions, where the latter is
`max(abs(error) / max(abs(reference), 1e-12))`. Total-norm absolute error is also
reported. Tolerances are fixed and never widened. Eligible cooperative results
retain the additional comparison against the matching two-kernel output.

`--validation skip` explicitly disables CPU reference/validation, records
`validation_status=skipped` and must not be described as a validated benchmark.
Timing status remains independent of numerical validation: a numerical failure
keeps its raw elapsed times and `timing_status=success`, while overall status and
validation fail. Unexpected execution errors stop further launches; remaining
rows are not_run. Expected resource/policy restrictions are unsupported.
A setup failure is recorded in run metadata and leaves cases not_run; an
unsupported state/driver setup makes the workload unsupported. Exit status is
nonzero for unexpected failures or numerical mismatches. All-unsupported runs
can exit zero: inspect the summary instead of treating exit zero as GPU success.

## Output and resource/profiling evidence

`--results PREFIX` writes PREFIX.json and PREFIX.csv; the directory must already
exist. Make targets use build/full and build/smoke. JSON has metadata, results
and a supported/unsupported/failure/not_run summary. CSV stores the same row
fields, with run metadata repeated using the `run_` prefix. CSV cells contain
JSON tokens (including quoted strings, arrays and null); decode them as JSON
when typed values are needed. JSON null means unavailable or n/a, never an
observed zero. Rows are flushed as they complete; an interrupted JSON stream
may lack its closing array/summary, while completed CSV rows remain readable.

Custom and BlockReduce stage fields record actual registers/thread,
local bytes/thread, static/dynamic shared bytes/block, active blocks/SM and
theoretical occupancy. These are queried from the exact kernel and block size.
CUB DeviceSegmentedReduce's private kernel attributes are unavailable through
this driver; obtain per-kernel resources in a separate profiler run. Spill
loads/stores are explicitly unavailable through runtime APIs and are compiler
information, not measured runtime traffic:

```sh
make cuda-evidence CUDA_ARCH=sm_80 NVCCFLAGS='-O2'
```

Inspect the retained build.log ptxas spill-load/store messages, resources.txt,
PTX and SASS per kernel. The manifest binds compiler flags, source and binary
hashes. Evidence capture does not certify register residency or synchronization.

The profiling helper first queries the installed Nsight Compute metric catalog,
then requests only supported launch-resource, occupancy, DRAM read/write and local load/store
sector metrics. Metric units remain those in ncu's CSV; local sectors are not
compiler spill counts. Unavailable tools, counters or permissions are recorded
in the profile manifest and never replaced with zeros.

```sh
make profile PROFILE_ARGS='--case 6 --output build/profile-smoke -- --preset smoke --qubits 10'
# Same case ordering/input settings as the desired full matrix:
make profile PROFILE_ARGS='--case 6 --output build/profile-full -- --preset full'
```

Case IDs are zero-based and follow the loops described above: reference band,
block-size list order, multiplier, primitive (custom then CUB), storage
(registers, shared, global), layout and strategy (two_kernel, cooperative).
With defaults, case 6 is custom two-kernel global thread-major at multiplier 1;
the segmented row follows all custom tuning rows for that reference.
Choose a supported row from a normal result file. The script retains metric
discovery, version, command, binary hash, per-kernel metrics.csv, ncu report,
application output and profile manifest. The profiler starts disabled; a CUDA profiler API range encloses only the selected
reduction, excluding generation and setup. Profiling mode executes one complete
operation without event timing, without warmups, and without CPU validation;
unselected rows are not_run. Profile data stays in separate artifacts and is
never merged into headline event timings. An application-unsupported selected
case can produce no kernels: inspect the run JSON and profiler warnings before
interpreting any report.

## Atomic policy and actual implementation-environment evidence

No atomic-add exception has been introduced. Cooperative methods remain
unsupported because no build has certified atomic-free grid synchronization;
the examined CUDA vendor implementation violates the policy. Register requests
remain unsupported until binary/architecture-bound residency evidence exists;
k>3 additionally exceeds capacity. CUB methods have no version gate and record
atomic-free audit status as **unverified for the selected build**. The strict
atomic-free requirement still applies: compatible APIs, passing CPU-reference
comparisons and performance measurements are separate from implementation
verification. Review the installed source and generated device code for every
selected architecture; numerical correctness alone cannot prove absence of
atomics. See cooperative-sync-audit.md and cub-baselines.md.
Do not interpret the presence of all requested rows as support for every row.

For this Step 6 implementation, tests and sanitizers were **skipped at the
user's explicit request**, overriding the task's request to run the suite.
`make host` succeeded. `make cuda` failed before compilation because nvcc is
absent. GPU tools and Nsight Compute are also unavailable here. Consequently,
CUDA compilation, all GPU correctness checks, full-size validation, actual
resource metrics, profiling and all benchmark times remain unverified. No
performance numbers or successful device validation are claimed. The commands
above are the exact remaining execution path on a suitably provisioned GPU.
