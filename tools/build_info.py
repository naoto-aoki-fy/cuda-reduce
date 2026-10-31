#!/usr/bin/env python3
"""Write stable, source/flags-bound build metadata for the Make-built driver."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
paths = sorted(p for folder in ('src', 'include', 'tools') for p in (root / folder).rglob('*') if p.is_file())
paths.append(root / 'Makefile')
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
compiler = os.environ.get('MEASUREMENT_NVCC', 'nvcc')
try:
    version = subprocess.run(shlex.split(compiler) + ['--version'], capture_output=True, text=True, check=True).stdout
except (OSError, subprocess.CalledProcessError) as error:
    version = f'unavailable: {error}'
revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
info = {'source_sha256': hashes, 'git_revision': revision, 'compiler': compiler,
        'compiler_version': version, 'nvcc_flags': os.environ.get('MEASUREMENT_NVCCFLAGS', ''),
        'cppflags': os.environ.get('MEASUREMENT_CPPFLAGS', ''),
        'target_architecture': 'see compiler flags; omitted architecture uses nvcc default'}
info['build_id'] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
# The macro is a C string containing a JSON object.
content = '#pragma once\n#define MEASUREMENT_BUILD_INFO ' + json.dumps(json.dumps(info, sort_keys=True)) + '\n'
path = Path(sys.argv[1])
if not path.exists() or path.read_text() != content:
    path.write_text(content)
