#!/usr/bin/env python3
"""Versioned benchmark plans and durable results; Python standard library only."""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value, name, low=0, high=(1 << 64) - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}]')
    return value


def keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'unknown fields or invalid object in {name}')


def choices(value, allowed, name):
    if not isinstance(value, list) or not value or any(v not in allowed for v in value):
        raise ValueError(f'{name}: expected nonempty selection from {allowed}')
    if len({canonical(v) for v in value}) != len(value):
        raise ValueError(f'{name}: duplicate entries')
    return value


def positions(q, k, spec):
    keys(spec, ['patterns', 'explicit', 'random_count', 'seed', 'reverse'], 'positions')
    patterns = choices(spec.get('patterns', ['low', 'middle', 'high']),
                       ['low', 'middle', 'high', 'all_contiguous', 'alternating', 'mixed', 'distant', 'random', 'explicit'], 'patterns')
    integer(spec.get('seed', 20261031), 'position seed')
    count = integer(spec.get('random_count', 3), 'random_count', 1, 10000)
    if type(spec.get('reverse', False)) is not bool:
        raise ValueError('reverse must be boolean')
    result = []
    for pattern in patterns:
        if pattern in ('low', 'middle', 'high'):
            start = {'low': 0, 'middle': (q-k)//2, 'high': q-k}[pattern]
            result.append((pattern, list(range(start, start+k))))
        elif pattern == 'all_contiguous':
            result.extend((f'contiguous_{s}', list(range(s, s+k))) for s in range(q-k+1))
        elif pattern == 'alternating':
            result.append((pattern, (list(range(0, q, 2)) + list(range(1, q, 2)))[:k]))
        elif pattern == 'mixed':
            low = (k+1)//2
            result.append((pattern, list(range(low)) + list(range(q-(k-low), q))))
        elif pattern == 'distant':
            result.append((pattern, list(range(k-1)) + [q-1]))
        elif pattern == 'random':
            rng = random.Random(identity([spec.get('seed', 20261031), q, k]))
            for i in range(count):
                # Keep sampled ordering: sorting would change bin semantics.
                p = rng.sample(range(q), k)
                if k > 1 and q > k:
                    for _ in range(64):
                        if max(p) - min(p) > k-1:
                            break
                        p = rng.sample(range(q), k)
                    else:
                        p = list(range(k-1)) + [q-1]
                        rng.shuffle(p)
                result.append((f'random_{i}', p))
        else:
            explicit = spec.get('explicit', [])
            if not isinstance(explicit, list) or not explicit:
                raise ValueError('explicit pattern requires position arrays')
            for p in explicit:
                if not isinstance(p, list) or not p or len(p) > min(16, q):
                    raise ValueError('invalid explicit position count')
                for bit in p:
                    integer(bit, 'position', 0, q-1)
                if len(set(p)) != len(p):
                    raise ValueError('duplicate explicit position')
                if len(p) == k:
                    result.append(('explicit', p))
    if spec.get('reverse', False):
        result += [(label + '_reversed', list(reversed(p))) for label, p in result]
    seen = set()
    for label, p in result:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            yield label, p


METHODS = ['custom', 'cub', 'segmented']
STORAGE = ['shared', 'global', 'registers']
LAYOUTS = ['thread_major', 'bin_major']
STRATEGIES = ['two_kernel', 'cooperative']


def candidates(spec):
    keys(spec, ['methods', 'storage', 'layouts', 'block_sizes', 'sm_multipliers', 'grid_block_counts', 'strategies', 'include'], 'candidates')
    if 'include' in spec:
        if len(spec) != 1 or not isinstance(spec['include'], list) or not spec['include']:
            raise ValueError('include must be a nonempty list and cannot be mixed with filters')
        result = []
        for entry in spec['include']:
            keys(entry, ['method', 'storage', 'layout', 'block_size', 'sm_multiplier', 'grid_blocks', 'strategy'], 'included candidate')
            method = entry.get('method')
            choices([method], METHODS, 'method')
            if method == 'segmented':
                if len(entry) != 1:
                    raise ValueError('segmented has no custom tuning fields')
                result.append(dict(method=method))
            else:
                required = {'method', 'storage', 'block_size', 'strategy'}
                if ('sm_multiplier' in entry) == ('grid_blocks' in entry):
                    raise ValueError('candidate requires exactly one of sm_multiplier or grid_blocks')
                if not required <= set(entry) or (entry['storage'] != 'registers' and 'layout' not in entry):
                    raise ValueError('included candidate missing tuning fields')
                choices([entry['storage']], STORAGE, 'storage')
                choices([entry['strategy']], STRATEGIES, 'strategy')
                integer(entry['block_size'], 'block_size', 128, 512)
                choices([entry['block_size']], [128, 256, 512], 'block_size')
                if 'grid_blocks' in entry:
                    integer(entry['grid_blocks'], 'grid_blocks', 1)
                else:
                    integer(entry['sm_multiplier'], 'sm_multiplier', 1, (1 << 31)-1)
                if entry['storage'] == 'registers':
                    if 'layout' in entry:
                        raise ValueError('register layout is not applicable')
                else:
                    choices([entry['layout']], LAYOUTS, 'layout')
                result.append(entry.copy())
    else:
        methods = choices(spec.get('methods', METHODS), METHODS, 'methods')
        storage = choices(spec.get('storage', ['shared', 'global']), STORAGE, 'storage')
        layouts = choices(spec.get('layouts', LAYOUTS), LAYOUTS, 'layouts')
        blocks = choices(spec.get('block_sizes', [128, 256, 512]), [128, 256, 512], 'block_sizes')
        for block in blocks:
            integer(block, 'block_size', 128, 512)
        if 'grid_block_counts' in spec and 'sm_multipliers' in spec:
            raise ValueError('grid_block_counts cannot be combined with sm_multipliers')
        grid_mode = 'grid_block_counts' in spec
        grid_key = 'grid_blocks' if grid_mode else 'sm_multiplier'
        counts = spec.get('grid_block_counts' if grid_mode else 'sm_multipliers', [1, 2, 4, 8, 16])
        if not isinstance(counts, list) or not counts:
            raise ValueError('grid counts must be a nonempty list')
        for count in counts:
            integer(count, grid_key, 1, (1 << 64)-1 if grid_mode else (1 << 31)-1)
        if grid_mode and len(set(counts)) != len(counts):
            raise ValueError('duplicate grid counts')
        strategies = choices(spec.get('strategies', ['two_kernel']), STRATEGIES, 'strategies')
        result = []
        for method in methods:
            if method == 'segmented':
                result.append(dict(method=method))
                continue
            for b in blocks:
                for m in counts:
                    for st in storage:
                        for layout in ([None] if st == 'registers' else layouts):
                            for strategy in strategies:
                                c = dict(method=method, storage=st, block_size=b, strategy=strategy)
                                c[grid_key] = m
                                if layout is not None:
                                    c['layout'] = layout
                                result.append(c)
    return list({identity(c): c for c in result}.values())


DEFAULTS = dict(seed=20261031, scheduling_seed=20261101, repetitions=1, warmups=3,
                chunk_elements=65536, validation='full', memory_reserve_bytes=512*1024**2,
                workspace_limit_bytes=(1 << 64)-1, case_budget_ms=0, run_budget_seconds=0)


def plan(config, overrides=None):
    keys(config, ['version', 'q', 'k', 'positions', 'candidates', 'measurement'], 'configuration')
    if type(config.get('version')) is not int or config['version'] != 1:
        raise ValueError('configuration version must be 1')
    settings = dict(DEFAULTS)
    keys(config.get('measurement', {}), DEFAULTS, 'measurement')
    settings.update(config.get('measurement', {}))
    settings.update(overrides or {})
    for name, value in settings.items():
        if name == 'validation':
            if value not in ('full', 'skip'):
                raise ValueError('validation must be full or skip')
        else:
            integer(value, name, 1 if name in ('repetitions', 'chunk_elements') else 0)
    qs = config.get('q')
    if not isinstance(qs, list) or not qs:
        raise ValueError('q must be a nonempty list')
    for q in qs:
        integer(q, 'q', 1, 59)  # 16*N must fit uint64_t.
    ks = config.get('k', 'all')
    if ks != 'all':
        if not isinstance(ks, list) or not ks:
            raise ValueError('k must be all or a nonempty list')
        for k in ks:
            integer(k, 'k', 1, 16)
        if any(k > q for q in qs for k in ks):
            raise ValueError('k exceeds q; use k="all" for q-dependent valid widths')
    cs = candidates(config.get('candidates', {}))
    workloads = {}
    for q in qs:
        for k in (range(1, min(16, q)+1) if ks == 'all' else ks):
            generated = list(positions(q, k, config.get('positions', {})))
            if not generated:
                raise ValueError(f'no position arrays match q={q}, k={k}')
            for category, p in generated:
                w = dict(q=q, positions=p, seed=settings['seed'])
                wid = identity(w)
                workloads.setdefault(wid, dict(w, workload_id=wid, k=k, N=1 << q, bins=1 << k,
                                               input_bytes=16 << q, position_pattern=category))
    if not workloads:
        raise ValueError('position specifications generated no workloads')
    cases = []
    for w in workloads.values():
        for c in cs:
            cid = identity(c)
            cases.append(dict(workload=w, candidate=c, candidate_id=cid,
                              case_key=identity([w['workload_id'], cid])))
    # Shuffle inside each state group at execution, keeping state reuse.
    return settings, list(workloads.values()), cases


def estimate(case, device, settings):
    w, c = case['workload'], case['candidate']
    reason = None
    b, k = w['bins'], w['k']
    workspace = grid = threads = accumulator = None
    if c.get('storage') == 'registers':
        reason = 'register capacity exceeded' if k > 3 else 'register residency audit unverified'
    elif c.get('strategy') == 'cooperative':
        reason = 'cooperative synchronization strict atomic policy unverified'
    requested = c.get('grid_blocks')
    minimum_grid = requested if requested is not None else c.get('sm_multiplier')
    lower_bound = 8*b + 1 if c['method']=='segmented' else (
        (8*minimum_grid*c['block_size']*b if c['storage']=='global' else 0)
        + 8*minimum_grid*b + 8*b + 4*k)
    if lower_bound > settings['workspace_limit_bytes']:
        reason = reason or 'workspace lower bound exceeds memory limit'
    if c['method'] != 'segmented' and (device or requested is not None):
        grid = requested if requested is not None else device['SM_count'] * c['sm_multiplier']
        if grid > (1 << 31)-1:
            reason = reason or 'requested grid exceeds CUDA grid-x limit'
        threads = grid * c['block_size']
        accumulator = 8 * threads * b if c['storage'] == 'global' else 0
        workspace = accumulator + 8 * grid * b + 8*b + 4*k
        if workspace > settings['workspace_limit_bytes']:
            reason = reason or 'workspace memory limit exceeded'
        if device and c['storage'] == 'shared' and 8*c['block_size']*b > device['shared_memory_optin_per_block']:
            reason = reason or 'shared accumulator alone exceeds device shared memory limit'
    if device:
        available = device['initial_free_device_bytes'] - settings['memory_reserve_bytes']
        generation = w['input_bytes'] + 16*min(w['N'], 1 << 20)
        if generation > available or w['input_bytes'] + (workspace or lower_bound) > available:
            reason = reason or 'device memory preflight: input/workspace/reserve exceeds available bytes'
    return dict(workspace_lower_bound_bytes=lower_bound, estimated_workspace_bytes=workspace, estimated_blocks=grid, total_threads=threads,
                accumulator_bytes=accumulator, resource_feasibility='unsupported' if reason else 'runtime_check_required',
                resource_rejection_reason=reason)


def append_record(file, record):
    file.write(canonical(record) + '\n')
    file.flush()
    os.fsync(file.fileno())


def read_journal(path):
    """Only an unterminated final record may be discarded; other corruption fails."""
    data = path.read_bytes()
    complete = data.rfind(b'\n') + 1
    records = [json.loads(line) for line in data[:complete].splitlines()]
    if not records or records[0].get('type') != 'header':
        raise ValueError('journal missing complete header')
    rows = {}
    for entry in records[1:]:
        if entry.get('type') != 'result':
            raise ValueError('unknown journal record')
        rows[entry['row']['case_key']] = entry['row']
    return records[0], rows, complete


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def export_results(prefix, header, rows):
    ordered = list(rows.values())
    counts = dict.fromkeys(('success', 'unsupported', 'failure', 'not_run'), 0)
    counts.update(collections.Counter(row['status'] for row in ordered))
    counts['requested_cases'] = header['metadata'].get('candidate_count', len(ordered))
    atomic_json(Path(str(prefix)+'.json'), dict(metadata=header['metadata'], results=ordered, summary=dict(counts)))
    path = Path(str(prefix)+'.csv')
    tmp = Path(str(path)+'.tmp')
    # Preserve legacy CSV conventions: each cell is a JSON token.
    fields = sorted({key for row in ordered for key in row} |
                    {'run_'+key for key in header['metadata']})
    with tmp.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in ordered:
            expanded = dict(row, **{'run_'+k: v for k, v in header['metadata'].items()})
            writer.writerow([canonical(expanded.get(k)) for k in fields])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def base_row(case, settings, run_id, index):
    w, c = case['workload'], case['candidate']
    row = dict(case_id=index, case_key=case['case_key'], workload_id=w['workload_id'],
                candidate_id=case['candidate_id'], benchmark_run_id=run_id, preset='matrix',
                q=w['q'], N=w['N'], k=w['k'], bins=w['bins'], positions=w['positions'],
                band=w['position_pattern'], position_pattern=w['position_pattern'], seed=w['seed'],
                state_bytes=w['input_bytes'], input_bytes=w['input_bytes'],
                method={'custom': 'custom', 'cub': 'CUB BlockReduce', 'segmented': 'CUB DeviceSegmentedReduce'}[c['method']],
                strategy=c.get('strategy'), storage=c.get('storage'), layout=c.get('layout'),
                block_size=c.get('block_size'), sm_multiplier=c.get('sm_multiplier'), grid_blocks=c.get('grid_blocks'),
                repetitions=settings['repetitions'], warmups=settings['warmups'], chunk_elements=settings['chunk_elements'],
                status='not_run', reason='', timing_status='not_run', timing_mode='single_invocation',
                raw_elapsed_ms=[], completed_repetitions=0, validation_status='skipped' if settings['validation']=='skip' else 'not_run',
                validation_reason='', absolute_tolerance=1e-12, relative_tolerance=1e-11,
                mean_ms=None, median_ms=None, min_ms=None, max_ms=None, sample_stddev_ms=None,
                max_absolute_error=None, max_relative_error=None, total_norm_absolute_error=None,
                reference_total_norm=None, input_effective_GB_per_s=None, blocks=None,
                workspace_bytes=None, total_live_device_bytes=None, resources=None,
                N_per_bin=w['N']/w['bins'], profiling_run=False, profiling_status='not_run',
                compiler_spill_status='unavailable; use make cuda-evidence',
                runtime_metrics_status='unavailable; separate profiler run required')
    for name in ('compiler_spill_loads', 'compiler_spill_stores', 'achieved_occupancy', 'dram_read_bytes', 'dram_write_bytes'):
        row[name] = None
    for stage in (1, 2):
        for name in ('binary_architecture', 'ptx_architecture', 'registers_per_thread', 'local_bytes_per_thread',
                     'static_shared_bytes_per_block', 'dynamic_shared_bytes_per_block', 'active_blocks_per_sm', 'theoretical_occupancy'):
            row[f'stage{stage}_{name}'] = None
    return row


def enrich(row, device):
    if row.get('blocks') is not None and row.get('block_size') is not None:
        t = row['blocks'] * row['block_size']
        row['total_threads'] = t
        row['v2_total_threads'] = (t & -t).bit_length()-1
        row['bins_over_gcd_bins_threads'] = (row['bins']//math.gcd(row['bins'], t)
                                            if row['positions'] == list(range(row['k'])) else None)
    l2 = device.get('L2_bytes')
    row['input_over_L2'] = row['input_bytes']/l2 if l2 else None
    a = row.get('accumulator_bytes')
    row['accumulator_over_L2'] = a/l2 if l2 and a is not None else None


def completed(row):
    return row['status'] in ('success', 'unsupported', 'failure') and row['timing_status'] != 'stopped'


def run(args):
    prefix = Path(args.results)
    journal = Path(str(prefix)+'.journal.jsonl')
    if args.recover:
        header, rows, _ = read_journal(journal)
        export_results(prefix, header, rows)
        print(f'Recovered {len(rows)} rows')
        return 0
    if not args.matrix_config:
        raise ValueError('--matrix-config is required')
    config = json.loads(Path(args.matrix_config).read_text())
    if args.grid_block_counts is not None:
        spec = config.setdefault('candidates', {})
        if 'include' in spec or 'sm_multipliers' in spec or 'grid_block_counts' in spec:
            raise ValueError('--grid-block-counts requires candidates without include or grid/SM count filters')
        spec['grid_block_counts'] = [int(v) for v in args.grid_block_counts.split(',')]
    overrides = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k, None) is not None}
    settings, workloads, cases = plan(config, overrides)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        output = dict(version=1, workload_count=len(workloads), candidate_count=len(cases),
                      distinct_candidates=len({c['candidate_id'] for c in cases}), settings=settings,
                      cases=[dict(c, **estimate(c, None, settings)) for c in cases])
        atomic_json(Path(str(prefix)+'.plan.json'), output)
        print(f"workloads={len(workloads)} candidates={len(cases)}; {prefix}.plan.json")
        return 0
    binary = Path(args.binary).resolve()
    query = subprocess.run([str(binary), '--device-info', '1'], check=True, text=True, capture_output=True)
    device = json.loads(query.stdout)
    # Free memory is deliberately excluded from compatibility; it varies per run.
    environment = {k: v for k, v in device.items() if k not in ('initial_free_device_bytes',)}
    fingerprint = identity(dict(config=config, settings=settings, device=environment,
                                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows = {}
    if args.resume and journal.exists():
        header, rows, complete = read_journal(journal)
        if header['fingerprint'] != fingerprint:
            raise ValueError('incompatible resume: configuration, settings, GPU, runtime or build changed')
        expected = {c['case_key'] for c in cases}
        if set(rows) - expected:
            raise ValueError('journal contains unknown cases')
        with journal.open('r+b') as f:
            f.truncate(complete)
    else:
        if journal.exists():
            raise ValueError('journal already exists; use --resume or a new --results prefix')
        header = dict(type='header', version=1, fingerprint=fingerprint,
                      metadata=dict(device, benchmark_run_id=str(uuid.uuid4()), settings=settings,
                                    configuration=config, workload_count=len(workloads), candidate_count=len(cases),
                                    startup_status='success', startup_reason=''))
        with journal.open('x') as f:
            append_record(f, header)
    run_id = header['metadata']['benchmark_run_id']
    index = {c['case_key']: i for i, c in enumerate(cases)}
    groups = collections.defaultdict(list)
    started = time.monotonic()
    for case in cases:
        old = rows.get(case['case_key'])
        if old and completed(old):
            continue
        groups[case['workload']['q']].append(case)
    with journal.open('a') as progress:
        def record(case, row):
            row.update(case_key=case['case_key'], workload_id=case['workload']['workload_id'],
                       candidate_id=case['candidate_id'], benchmark_run_id=run_id, case_id=index[case['case_key']])
            append_record(progress, dict(type='result', row=row))
            rows[case['case_key']] = row
        for q, pending in groups.items():
            if settings['run_budget_seconds'] and time.monotonic()-started >= settings['run_budget_seconds']:
                break
            random.Random(identity([settings['scheduling_seed'], q])).shuffle(pending)
            runnable = []
            for case in pending:
                resource = estimate(case, device, settings)
                if resource['resource_rejection_reason']:
                    row = base_row(case, settings, run_id, index[case['case_key']])
                    row.update(resource, status='unsupported', timing_status='unsupported',
                               reason=resource['resource_rejection_reason'], failure_category='resource_or_policy_rejection')
                    enrich(row, device)
                    record(case, row)
                else:
                    runnable.append(case)
            if not runnable:
                continue
            with tempfile.TemporaryDirectory(prefix='measurement-') as directory:
                path = Path(directory)/'cases.tsv'
                with path.open('w') as f:
                    for case in runnable:
                        w, c = case['workload'], case['candidate']
                        f.write(' '.join([','.join(map(str, w['positions'])), w['position_pattern'], c['method'],
                                          c.get('storage', 'global'), c.get('layout', 'thread_major'), c.get('strategy', 'two_kernel'),
                                          str(c.get('block_size', 256)), str(c.get('sm_multiplier', 1)), str(c.get('grid_blocks', 0))])+'\n')
                command = [str(binary), '--qubits', str(q), '--case-plan', str(path), '--stream-rows', '1',
                           '--results', str(Path(directory)/'raw')]
                for name in ('seed', 'repetitions', 'warmups', 'chunk_elements', 'validation', 'memory_reserve_bytes',
                             'workspace_limit_bytes', 'case_budget_ms'):
                    command += ['--'+name.replace('_', '-'), str(settings[name])]
                # Runtime driver also enforces a run budget between cases.
                if settings['run_budget_seconds']:
                    remaining_ms = max(1, int((settings['run_budget_seconds']-(time.monotonic()-started))*1000))
                    command += ['--run-budget-ms', str(remaining_ms)]
                print(f'q={q}: executing {len(runnable)} candidates', flush=True)
                with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
                    received = set()
                    for line in process.stdout:
                        if line.startswith('MEASUREMENT_ROW '):
                            row = json.loads(line[len('MEASUREMENT_ROW '):])
                            local = integer(row['case_id'], 'child case_id', 0, len(runnable)-1)
                            if local in received:
                                raise ValueError('duplicate child case')
                            received.add(local)
                            case = runnable[local]
                            base = base_row(case, settings, run_id, index[case['case_key']])
                            base.update(estimate(case, device, settings))
                            base.update({k: v for k, v in row.items() if not k.startswith('run_')})
                            base['preset'] = 'matrix'
                            if base['workspace_bytes'] is not None:
                                base['resource_feasibility'] = 'feasible' if base['status'] == 'success' else base['status']
                            if base['status'] == 'unsupported':
                                base['resource_rejection_reason'] = base['reason']
                            base['failure_category'] = ('unexpected_allocation_failure' if base['reason'].startswith('CUDA allocation:')
                                                        else 'resource_or_policy_rejection' if base['status']=='unsupported'
                                                        else 'execution_or_validation_failure' if base['status']=='failure' else None)
                            enrich(base, device)
                            record(case, base)
                        else:
                            print(line, end='', flush=True)
                    code = process.wait()
                export_results(prefix, header, rows)
                if code:
                    raise RuntimeError(f'benchmark child exited {code}; durable results retained; fix cause and resume')
                if len(received) != len(runnable):
                    raise RuntimeError('benchmark child omitted cases; resume to retry missing cases')
    # Include explicit pending rows in exports, without marking them complete in the journal.
    for case in cases:
        rows.setdefault(case['case_key'], base_row(case, settings, run_id, index[case['case_key']]))
    rows = {c['case_key']: rows[c['case_key']] for c in cases}
    export_results(prefix, header, rows)
    print(f'Results: {prefix}.json, {prefix}.csv; journal: {journal}')
    return int(any(r['status']=='failure' for r in rows.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-config')
    parser.add_argument('--grid-block-counts', help='comma-separated exact grid-x counts; excludes SM filters')
    parser.add_argument('--results', default='build/full')
    parser.add_argument('--binary', default='build/measurement_validate')
    for flag in ('dry-run', 'resume', 'recover'):
        parser.add_argument('--'+flag, action='store_true')
    for name in DEFAULTS:
        parser.add_argument('--'+name.replace('_', '-'), type=str if name=='validation' else int, default=None)
    args = parser.parse_args()
    try:
        # A prefix has exactly one writer; stale lock files are harmless.
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.results + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return run(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'failure: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
