#!/usr/bin/env python3
"""Discover Nsight metrics and profile one case separately from benchmark timing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', default='build/measurement_validate')
parser.add_argument('--case', type=int, default=6)
parser.add_argument('--output', default='build/profile')
parser.add_argument('benchmark_args', nargs=argparse.REMAINDER, help='After --: matching preset/input options')
args = parser.parse_args()
folder = Path(args.output)
folder.mkdir(parents=True, exist_ok=True)
manifest = {'status': 'unavailable', 'metrics': {}, 'headline_timing': False}

def save():
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

ncu = shutil.which('ncu')
if ncu is None:
    manifest['reason'] = 'Nsight Compute (ncu) unavailable'
    save()
    print(manifest['reason'])
    raise SystemExit(0)
version = subprocess.run([ncu, '--version'], capture_output=True, text=True)
manifest['version'] = version.stdout + version.stderr
query_command = [ncu, '--query-metrics', '--query-metrics-mode', 'all', '--csv']
query = subprocess.run(query_command, capture_output=True, text=True)
(folder / 'available-metrics.csv').write_text(query.stdout + query.stderr)
manifest['query_command'] = query_command
manifest['query_exit_code'] = query.returncode
if query.returncode:
    manifest['reason'] = 'metric discovery failed; see available-metrics.csv'
    save()
    raise SystemExit(0)
discovered = set(re.findall(r'\b(?:sm|smsp|dram|lts|l1tex|launch)__[A-Za-z0-9_.]+', query.stdout))
candidates = {
    'registers_per_thread': ['launch__registers_per_thread'],
    'static_shared_bytes_per_block': ['launch__shared_mem_per_block_static'],
    'dynamic_shared_bytes_per_block': ['launch__shared_mem_per_block_dynamic'],
    'achieved_occupancy': ['sm__warps_active.avg.pct_of_peak_sustained_active'],
    'dram_read_bytes': ['dram__bytes_read.sum'],
    'dram_write_bytes': ['dram__bytes_write.sum'],
    'local_load_sectors': ['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'],
    'local_store_sectors': ['l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum'],
}
selected = []
for name, choices in candidates.items():
    metric = next((m for m in choices if m in discovered), None)
    manifest['metrics'][name] = metric or 'unavailable in installed metric discovery'
    if metric:
        selected.append(metric)
if not selected:
    manifest['reason'] = 'no requested metrics available; discovery retained'
    save()
    raise SystemExit(0)
extra = args.benchmark_args
if extra[:1] == ['--']:
    extra = extra[1:]
if any(key in extra for key in ('--profile-case', '--validation', '--results')):
    parser.error('profile-case, validation and results are owned by this script')
binary = Path(args.binary).resolve()
manifest['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
command = [ncu, '--profile-from-start', 'off', '--metrics', ','.join(selected), '--csv', '--log-file', str(folder / 'metrics.csv'),
           '--export', str(folder / 'kernels'), '--force-overwrite', str(binary), *extra,
           '--profile-case', str(args.case), '--validation', 'skip', '--results', str(folder / 'run')]
manifest['command'] = command
result = subprocess.run(command, capture_output=True, text=True)
(folder / 'run.log').write_text(result.stdout + result.stderr)
manifest['exit_code'] = result.returncode
manifest['status'] = 'unavailable'
manifest['reason'] = 'profiling command failed or selected case did not execute'
try:
    run = json.loads((folder / 'run.json').read_text())
    selected_case = next(row for row in run['results'] if row['case_id'] == args.case)
    manifest['application_case_status'] = selected_case['status']
    manifest['application_case_reason'] = selected_case['reason']
    if result.returncode == 0 and selected_case['profiling_status'] == 'success':
        manifest['status'] = 'collected'
        manifest['reason'] = 'inspect per-kernel metrics.csv/ncu report; missing/denied counters are unavailable, never zero'
except (OSError, ValueError, KeyError, StopIteration) as error:
    manifest['application_result_error'] = str(error)
save()
print(f'Profiling artifacts: {folder}; headline CUDA-event timing disabled')
