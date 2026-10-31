#!/usr/bin/env python3
"""Capture reproducible evidence; this script does not certify register residency."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shlex
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arch', required=True, help='Explicit real architecture, e.g. sm_80')
parser.add_argument('--nvcc', default='nvcc')
parser.add_argument('--flags', default='-O2')
args = parser.parse_args()
if not re.fullmatch(r'sm_[0-9]+[af]?', args.arch):
    parser.error('--arch must be sm_<digits> with an optional a/f suffix')
for tool in (shlex.split(args.nvcc)[0], 'cuobjdump', 'make'):
    if not shutil.which(tool):
        parser.error(f'required tool unavailable: {tool}')
root = Path(__file__).resolve().parents[1]
folder = root / 'build' / 'evidence' / args.arch
folder.mkdir(parents=True, exist_ok=True)
keep = folder / 'intermediates'
keep.mkdir(exist_ok=True)
compute = args.arch.replace('sm_', 'compute_')
flags = (args.flags + f' --gpu-architecture={compute} --gpu-code={args.arch},{compute}'
         f' -lineinfo -Xptxas=-v,-warn-spills --keep --keep-dir={keep}')
command = ['make', '-B', 'cuda', f'BUILD={folder / "objects"}',
           f'NVCC={args.nvcc}', f'NVCCFLAGS={flags}']
manifest = {'architecture': args.arch, 'command': command, 'flags': flags,
            'residency': 'unverified: manual per-kernel PTX/SASS audit required',
            'cooperative_atomic_policy': 'unverified: trace grid.sync in every combined kernel; strict policy forbids internal atomic addition'}
paths = [root / 'Makefile', *root.glob('src/*'), *root.glob('include/measurement/*')]
manifest['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in paths if p.is_file()}

def capture(name, command):
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (folder / name).write_text(result.stdout)
    return result.returncode

capture('compiler.txt', shlex.split(args.nvcc) + ['--version'])
capture('cuobjdump-version.txt', ['cuobjdump', '--version'])
# Preserve the exact vendor synchronization implementation selected by this build.
# Preprocessing is evidence only; it cannot establish the executed call graph.
preprocess = (shlex.split(args.nvcc) + shlex.split(args.flags)
              + [f'--gpu-architecture={compute}', '-std=c++17', '-Iinclude',
                 '-E', 'src/global_reduction.cu'])
manifest['preprocess_command'] = preprocess
manifest['preprocess_exit_code'] = capture('cooperative-preprocessed.ii', preprocess)
status = capture('build.log', command)
manifest['build_exit_code'] = status
if status == 0:
    binary = folder / 'objects' / 'measurement_validate'
    manifest['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    status = status or manifest['preprocess_exit_code']
    for name, option in [('sass.txt', '--dump-sass'), ('ptx.txt', '--dump-ptx'),
                         ('resources.txt', '--dump-resource-usage')]:
        code = capture(name, ['cuobjdump', option, str(binary)])
        manifest[name + '_exit_code'] = code
        status = status or code
if shutil.which('nvidia-smi'):
    capture('gpu.txt', ['nvidia-smi', '-q'])
manifest['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in folder.glob('*') if p.is_file() and p.name != 'manifest.json'}
(folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Evidence: {folder}; residency remains unverified')
raise SystemExit(status)
