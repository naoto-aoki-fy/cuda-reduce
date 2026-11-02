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
configuration and one multiplier, running all backends with two-kernel reductions. For example:

```sh
./build/measurement_validate --qubits 10 --positions 7,1 --seed 20261031 --chunk-elements 128 --results build/single
```

`--preset smoke` sweeps k=1..min(8,q), low/middle/high contiguous bands and all
five SM multipliers on a smaller input (q=10 by default). This is explicitly a
smoke workload, never a full-size measurement.

## Full benchmark

```sh
make benchmark-full BENCHMARK_ARGS='--block-sizes 32,64,128,256,512,1024 --repetitions 5 --warmups 3 --seed 20261031 --chunk-elements 65536'
```

The full preset fixes q=31, N=2147483648, and the state alone consumes
34359738368 bytes (32 GiB). It rejects `--qubits`; it never substitutes a smaller
state. All matrix presets own `--positions` and `--sm-multiplier` and reject
those overrides. `--block-sizes` accepts a distinct comma-separated selection
from 32,64,128,256,512,1024; it defaults to `--block-size` (256). Measured repetitions
default to 1 and warmups to 3.

Each of 48 measurement configurations (k=1..16 times three position bands) has,
for each block size and multiplier, 10 rows: custom/CUB BlockReduce times
five storage/layout choices (registers with layout n/a, shared
with two layouts, global with two layouts). Add one DeviceSegmentedReduce row
per measurement configuration, independent of custom tuning. The default full
matrix therefore contains **2448 rows**; three block sizes produce **7248**,
and all six produce **14448**. The default block size remains 256.
Here k counts measured qubits, while the state size stays fixed at q=31.
The smoke preset retains its k=1..min(8,q) sweep. Larger k can exceed shared
memory or global workspace limits; these requests remain explicit unsupported
rows, as do register requests beyond the existing eight-bin capacity (k>3).
Even coincident position bands on a small smoke input retain separate labels.
Unsupported requests remain rows with explicit diagnostics. Segmented custom
fields and register layout are JSON null (not applicable).

To select exactly the k=4 subset of `full`:

```sh
make benchmark-full-k4 BENCHMARK_ARGS='--block-sizes 256 --repetitions 1 --warmups 3'
# Optional output prefix:
make benchmark-full-k4 BENCHMARK_K4_RESULTS=build/k4-run/results
```

This target uses `--preset full-k4` and defaults to
`build/full-k4/results.json` and `build/full-k4/results.csv`. It fixes q=31
(the same 32 GiB input), rejects `--qubits`, and retains low/middle/high bands,
all reduction methods, storage/layout choices, block sizes and SM multipliers.
Timing, validation, scheduling and result reporting use the shared full benchmark
implementation. There are 153 requested rows with the default block size, or 903
with all six sizes. Unsupported configurations, including every k=4 register
request, remain explicit rows. `make test-host` verifies the exact subset with
defaults and overrides. `benchmark-full` (k=1..16) and `benchmark-full-k1-6`
(k=1..6) retain their existing behavior and output locations.

For block-size comparisons, group CSV/JSON rows by workload (q and ordered
positions), method, storage, layout, strategy and SM multiplier, then compare
`mean_ms`, `median_ms` or `raw_elapsed_ms` across `block_size`. Input generation,
warmups and repetitions are shared settings for the sweep. DeviceSegmentedReduce
has a null block size and runs once per workload. Device/kernel thread limits,
shared memory and occupancy checks report unsupported rows with reasons; they
never substitute a different block size. The JSON planner accepts the same six
sizes in both `candidates.block_sizes` and explicit `include` entries; its omitted
`block_sizes` default remains `[128, 256, 512]`.

`make test-host` checks CLI and JSON planning for all six sizes. On a CUDA host,
run `make test-cuda` for numerical comparisons, dirty workspace reuse and idle
thread/grid-stride edge cases, including 32 and 1024 threads. `make sanitize`
also includes all six sizes in the custom-kernel quick checks.

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
library operation). CUB segmented input is a lazy gather/norm iterator;
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
reported. Tolerances are fixed and never widened.

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
hashes. Evidence capture does not certify register residency or atomic-free library execution.

The profiling helper first queries the installed Nsight Compute metric catalog,
then requests only supported launch-resource, occupancy, DRAM read/write and local load/store
sector metrics. Metric units remain those in ncu's CSV; local sectors are not
compiler spill counts. Unavailable tools, counters or permissions are recorded
in the profile manifest and never replaced with zeros.

```sh
make profile PROFILE_ARGS='--case 3 --output build/profile-smoke -- --preset smoke --qubits 10'
# Same case IDs/input settings as the desired full matrix:
make profile PROFILE_ARGS='--case 3 --output build/profile-full -- --preset full'
```

Case IDs are zero-based plan indices and follow the loops described above: reference band,
block-size list order, multiplier, primitive (custom then CUB), storage
(registers, shared, global) and layout.
With defaults, case 3 is custom two-kernel global thread-major at multiplier 1;
the segmented ID follows all custom tuning IDs for that reference. Execution is
scheduled independently, so result rows need not arrive in case-ID order.
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

No atomic-add exception has been introduced. Reductions use same-stream kernel
ordering between stages and retain the necessary block barriers. Register requests
remain unsupported until binary/architecture-bound residency evidence exists;
k>3 additionally exceeds capacity. CUB methods have no version gate and record
atomic-free audit status as **unverified for the selected build**. The strict
atomic-free requirement still applies: compatible APIs, passing CPU-reference
comparisons and performance measurements are separate from implementation
verification. Review the installed source and generated device code for every
selected architecture; numerical correctness alone cannot prove absence of
atomics. See [the CUB audit](cub-baselines.md).
Do not interpret the presence of all requested rows as support for every row.

The expanded infrastructure is host-tested (including preset compatibility,
configuration validation, simulated interruption and resume). This implementation
environment has no nvcc or GPU tools. CUDA compilation, device correctness,
sanitizers, actual resource metrics, profiling and benchmark timings remain
unverified. No performance data has been collected here.

## Opt-in version 1 benchmark plans

The commands above keep their tuning defaults and planned case IDs. Case IDs
changed after removal of the former alternative strategy; select
profiling IDs from new results. Existing result files remain historical evidence;
start a new result prefix when an old journal fails the resume fingerprint check. Only
`--matrix-config` selects the new Python standard-library planner and execution
wrapper. Basic planning and journal recovery need Python 3 on a POSIX system,
and do not build CUDA or query a device:

```sh
make benchmark-full BENCHMARK_RESULTS=build/pilot \
  BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --dry-run'
make benchmark-full BENCHMARK_RESULTS=build/pilot \
  BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --validation skip --resume'
# Reconstruct recorded rows even without a binary or GPU:
make benchmark-full BENCHMARK_RESULTS=build/pilot BENCHMARK_ARGS='--recover'
```

A dry run writes PREFIX.plan.json with every ordered workload, full candidate
configuration, workload_id, candidate_id, case_key, counts, input bytes, workspace
lower bounds, predictable policy rejections and resource estimates. No state is
allocated and no kernels are launched. SM-dependent workspace, actual shared
scratch, occupancy and segmented temporary storage require runtime checks;
unknown values are null. Inspect the plan file to see individual configurations.
`--results` in the Python command is equivalent to BENCHMARK_RESULTS in Make.

Example schema (all field names are strict; unknown fields are errors):

```json
{
  "version": 1,
  "q": [12, 16, 20],
  "k": [1, 3, 6],
  "positions": {
    "patterns": ["low", "middle", "high", "random"],
    "random_count": 4,
    "seed": 20261031,
    "reverse": true
  },
  "candidates": {
    "methods": ["custom", "cub", "segmented"],
    "storage": ["shared", "global"],
    "layouts": ["thread_major", "bin_major"],
    "block_sizes": [32, 64, 128, 256, 512, 1024],
    "sm_multipliers": [1, 2, 4, 8, 16],
    "strategies": ["two_kernel"]
  },
  "measurement": {
    "seed": 20261031,
    "scheduling_seed": 20261101,
    "validation": "full",
    "warmups": 3,
    "repetitions": 5,
    "chunk_elements": 65536,
    "memory_reserve_bytes": 536870912,
    "workspace_limit_bytes": 8589934592,
    "case_budget_ms": 0,
    "run_budget_seconds": 0
  }
}
```

`q` accepts a nonempty list in [1,59], bounded by uint64 input-byte arithmetic;
large requests are subject to runtime feasibility. `k` accepts a nonempty list
in [1,min(16,q)] or the string `"all"` for all valid widths independently per q.
Explicit invalid q/k combinations fail matrix generation; no input is downsized.
Repeated workload specifications are deduplicated by exact ordered positions,
q and input seed. This opt-in deduplication does not affect legacy presets.

Patterns include low/middle/high, `all_contiguous` (each start from 0 to q-k),
`alternating` (even bits, then odd bits if needed), `mixed` (ceil(k/2) low bits
and the remaining highest bits), `distant` (k-1 low bits and the highest bit),
`random`, and `explicit`. Explicit arrays use, for example,
`"patterns":["explicit"], "explicit":[[7,1,4],[4,1,7]]`; arrays apply to matching
k values and must be valid for every selected q. A requested q/k pair with no
matching position arrays is rejected. Random arrays are fixed-seed
ordered samples with noncontiguous sets whenever 1<k<q. For k=1 or k=q a
noncontiguous set is impossible. `reverse:true` adds reversed arrays. Arrays
are never sorted; permutations with different output-bin semantics retain
different identities. Duplicate arrays retain the first pattern label. Category
labels do not affect identity.

Candidate filters expand only the selected dimensions. `cub` means CUB
BlockReduce; `segmented` means CUB DeviceSegmentedReduce and generates one
candidate per workload regardless of other dimensions. Registers have no layout.
Optional `registers` storage remains explicitly unsupported under the current
residency audit. Only `two_kernel` is accepted in `strategies` and included
candidates' `strategy` fields; other strategy values are configuration errors.
The output `strategy` field remains `two_kernel` (null for segmented) for schema
compatibility. No policy gate is relaxed.

For shortlists use `candidates.include` instead of Cartesian filters:

```json
{"include": [
  {"method":"custom", "storage":"global", "layout":"thread_major",
   "block_size":256, "sm_multiplier":4, "strategy":"two_kernel"},
  {"method":"segmented"}
]}
```

There are no built-in winners. The shortlist in the example Stage B/C/D files
is illustrative; edit it based on separately validated Stage A results. Included
candidates specify all applicable tuning fields. Measurement defaults are seed
20261031, legacy scheduling seed 20261101, 1 repetition, 3 warmups, 65536 host chunk
elements, full validation, a 512 MiB memory reserve, unlimited workspace and no
budgets. Each measurement key can also be overridden on the command line using
hyphens, for example `--repetitions 9 --case-budget-ms 1000`.
`scheduling_seed` remains accepted for configuration compatibility; the deterministic
shortest-first scheduler does not use it.

## Independent benchmark stages

| Stage | Configuration | Coverage |
| --- | --- | --- |
| A | configs/benchmark-full-pilot.json | q=20,28; k=1,3,5,6,8,12,14,16; low/middle/high; 48 workloads, 121 candidates each (5808 cases) |
| B | configs/benchmark-full-coverage.json | q=12,16,20,24,26,28; all valid widths; low/middle/high, deduplicated; editable shortlist |
| C | configs/benchmark-full-positions.json | q=20,28; k=3,6,12; every contiguous start, structured/random patterns and reversals |
| D | configs/benchmark-full-large.json | q=29..33 with resource checks; editable shortlist |

Select a file independently with `--matrix-config`. Legacy `benchmark-full`
does not execute these stages automatically. Stage A files deliberately request
potentially infeasible configurations so the dataset retains rejection rows.

## Expanded execution, resources and budgets

Each q has one child process with one immutable `(q,seed)` input reused by all
pending candidates and one traversal for all required CPU references. Candidates
execute in estimated shortest-first order within each q, and smaller q groups
execute first regardless of their configuration order. Resume skips completed
cases, preserves the relative schedule of pending cases, and initializes the
same state again for each q with remaining work. Input and position generation
retain their independent seeds.

The same heuristic schedules the C++ `single`, `smoke`, `full`, `full-k1-6`, and `full-k4`
presets. It estimates relative work units, not milliseconds, without calibration:

- Let `N=2^q`, `B=2^k`, and `T=128 * sm_multiplier * block_size`. The fixed
  128-SM proxy keeps planning independent of the GPU and its current free memory.
- For custom/CUB block reductions, cost per invocation is
  `N*(16+2*k+s) + T*B*(s+8*p) + 16*128*sm_multiplier*B`, where storage weight
  `s` is 16 for global, 4 for shared, and 1 for registers; `p` is 1 for custom
  and 1.1 for CUB. This models input reads/bit extraction, accumulator traffic,
  per-bin block reductions and second-stage partials.
- For segmented reduction, cost is `N*(24+4*k) + 16*B`, allowing for virtual
  gather addressing and strided input reads. Both costs are multiplied by
  `warmups+repetitions`. Ties retain original plan order.

These weights are rough assumptions, not measured backend rankings. Occupancy,
cache effects, actual SM count, position-dependent coalescing, resource rejections,
allocation and CPU validation can change wall-clock order. Setup/reference work
still precedes execution and may dominate a short run. Timing, validation,
register audits, case coverage and allocation lifetimes are unchanged. Each
two-kernel reduction still executes its stages together; no cross-case baseline
dependency exists in the current driver. Python supplies an already scheduled
case plan, which the C++ child consumes in the supplied order.

Original case IDs, workload/candidate hashes, and `--profile-case` selection are
unchanged by scheduling. Additive JSON/CSV fields distinguish the orders:
`case_id` is the plan index, `scheduled_index` is the zero-based scheduled position,
and `execution_index` counts actual case attempts (null for unattempted rows and
Python preflight rejections). C++ counts within one invocation; Python counts
across child processes and resumed attempts in the journal. A retry receives a
new execution index; the journal retains earlier attempts while exports retain
the latest row per case. `estimated_cost` records the heuristic work units and
metadata records `scheduling=estimated_shortest_first_v1` (C++ children report
`planner_order`). Dry-run cases remain in plan order with explicit scheduled
indices. Journal record order includes preflight rejections as well as child
results. Exports after recovery may follow journal order; use the explicit fields
to reconstruct planned or actual execution order. Existing fingerprints reject
resuming journals from an older planner/build; use a new result prefix.

The planner first queries actual GPU properties, free memory and build metadata
without allocating a state. Each candidate is checked against input generation
scratch (up to 16 MiB), workspace, available memory, and the configured reserve.
No GPU-name resource lookup is used. Predictable rejections are retained without
allocating the state. The driver repeats memory checks immediately before state
allocation and before each workspace allocation. It queries kernel resources
and exact CUB segmented temporary-storage bytes. The workspace limit covers
positions, accumulators, partials, output and library temporary buffers; it
excludes the immutable input. The reserve is additional free headroom. The
estimated workspace formulas are the same as the legacy formulas above.

Shared-memory estimates in the planner are lower bounds; runtime kernel queries
also account for scratch, registers, occupancy and device launch limits.
`workspace_bytes` is the queried allocation requirement, not an allocator's
physical-granularity measurement. CUDA context, events, cuRAND internals and
allocator overhead are not exactly predicted; memory availability can change.
Unexpected CUDA allocation failures retain their diagnostics and are labeled
`failure_category=unexpected_allocation_failure`, separately from predictable
`resource_or_policy_rejection`. Legacy allocation status remains unsupported.
Unexpected execution/validation failures remain failures with nonzero exit.

`case_budget_ms` limits the measured repetition phase, including completed
output downloads; warmups and setup are excluded. It stops after a completed
repetition and retains actual samples and validated outputs. At least one
repetition completes. `timing_status=stopped` distinguishes a partial repetition
set from success; the reason explains the budget. `run_budget_seconds` is a soft
wall-clock budget checked between q groups and cases, including driver setup.
No in-flight CUDA kernel, operation or reference traversal is interrupted, so
budgets may be exceeded by one operation/setup phase. Unattempted cases remain
not_run. Resume retries not_run/missing/stopped cases; it does not combine partial
samples with a new attempt. A stopped case restarts its requested repetitions,
so increase/remove its budget with a new prefix for a longer experiment.

## Durable journal and expanded result schema

PREFIX.journal.jsonl is append-only: one header followed by one complete result
per finished case/attempt. Each record is flushed and fsynced. A prefix has an
exclusive POSIX lock, preventing concurrent writers. Existing journals require
`--resume` or a new prefix; `--resume` also starts a new journal if none exists.
A torn unterminated final line is discarded on resume; malformed complete lines
are errors. Latest attempts replace earlier attempts in exports, preserving all
raw attempts in the journal. Completed success/unsupported/failure cases are
skipped. Missing, not_run and budget-stopped cases are eligible for rerun.

Resume compares canonical configuration and measurement settings, GPU UUID/PCI
identifier and architecture, driver/runtime/library versions, compiler/build
metadata, binary SHA-256 and planner SHA-256. Changes reject resume. Initially
free memory may vary and is excluded from compatibility checks. Original run
metadata remains in the header. Changing seeds, validation, repetitions,
budgets, config or source/build requires a new prefix. Separate GPU results must
use separate prefixes. `--recover` only exports existing journal records; it
never queries CUDA or invents missing cases. It is useful after a crash. Final
normal exports also include explicit pending rows for all planned cases.

Final JSON retains the `metadata`, `results`, `summary` structure and CSV retains
JSON-token cells and `run_` metadata fields. Both exports use atomic replacement;
the journal is authoritative if interruption occurs between the two replacements.
Expanded exports retain legacy row fields and add:

- SHA-256 `workload_id` (q, ordered positions, input seed), `candidate_id`
  (applicable algorithm/tuning fields), `case_key` (the pair), and benchmark_run_id.
  `case_id` remains a zero-based plan index, independent of execution order.
- q, N, bins, exact positions, position_pattern, input_bytes, completed_repetitions,
  timing_mode (`single_invocation`), workspace estimates/lower bound, feasibility
  and resource_rejection_reason. Unsupported or unavailable metrics remain null.
- GPU UUID/PCI identity, L2 and shared-memory limits, initial free and total
  memory, runtime/driver/library versions, compiler flags/build/source revision.
  Kernel metadata and all original timing/numerical fields remain available.
- N_per_bin, total_threads, v2_total_threads, input_over_L2,
  accumulator_over_L2, and bins_over_gcd_bins_threads (only for the exact ascending
  low-contiguous mapping). Ratios are null when their inputs are unavailable.

Sample standard deviation is null for fewer than two samples. Successful timing
with `validation_status=skipped` is explicitly unvalidated. Resource feasibility,
numerical correctness, timing completion and atomic-policy audit are separate
properties. Profiler metrics remain separate; no counters or confidence intervals
are fabricated.

## Commands on each GPU

Run each command independently on its named machine, with a toolkit supporting
that installed GPU. `-arch=native` compiles for the locally visible GPU; actual
binary architecture is retained in result rows. Use CUDA_VISIBLE_DEVICES to
select one device if the machine has several. No multi-device reduction is
performed; this is the same plan executed separately on five GPU types.

```sh
# NVIDIA A100 machine
CUDA_VISIBLE_DEVICES=0 make benchmark-full NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_RESULTS=build/a100-pilot BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --validation skip --resume'
# NVIDIA GH200 machine
CUDA_VISIBLE_DEVICES=0 make benchmark-full NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_RESULTS=build/gh200-pilot BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --validation skip --resume'
# NVIDIA GB200 machine
CUDA_VISIBLE_DEVICES=0 make benchmark-full NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_RESULTS=build/gb200-pilot BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --validation skip --resume'
# NVIDIA RTX PRO 6000 Blackwell machine
CUDA_VISIBLE_DEVICES=0 make benchmark-full NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_RESULTS=build/rtx-pro-6000-pilot BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --validation skip --resume'
# NVIDIA RTX 3060 machine
CUDA_VISIBLE_DEVICES=0 make benchmark-full NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_RESULTS=build/rtx3060-pilot BENCHMARK_ARGS='--matrix-config configs/benchmark-full-pilot.json --validation skip --resume'
```

For each machine, first run `make test-host`, `make test-cuda` and `make sanitize`
with matching NVCCFLAGS, then a fully validated small/medium plan or smoke run:

```sh
make benchmark-smoke NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_ARGS='--qubits 12 --validation full --repetitions 3'
# Edit a copy of Stage B to representative medium q/k/candidates, then run:
make benchmark-full NVCCFLAGS='-O2 -arch=native -lineinfo' BENCHMARK_RESULTS=build/validated-medium BENCHMARK_ARGS='--matrix-config configs/benchmark-full-coverage.json --validation full --resume'
```

Full CPU-reference validation remains available at every size. It can take
considerable time, so keep validation experiments separate from large skip-mode
runs. Inspect statuses and numerical errors rather than relying solely on exit
zero, especially for resource-rejected plans.

## Limitations and deferred features

No selector, model or kernel optimization is included. Repeated-invocation timed
batches are deferred; timings remain one complete invocation per event interval.
Host-only dry runs do not predict exact CUB storage or device kernel residency.
The wrapper is POSIX-only (advisory locking), uses one GPU per process and has
soft budgets. CPU reference memory/time and repetition-output host memory are
not governed by the device-memory limit. Legacy direct-binary results retain
their original streaming behavior; durable resume is opt-in through the planner.
All runtime GPU results and cross-architecture comparisons still require the
commands above to be executed on the actual devices.
