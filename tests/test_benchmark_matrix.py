import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matrix', ROOT/'tools/benchmark_matrix.py')
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)


class MatrixTests(unittest.TestCase):
    def config(self):
        return dict(version=1, q=[12,16], k=[1,3], candidates=dict(methods=['segmented']))

    def test_stage_counts(self):
        c = json.loads((ROOT/'configs/benchmark-full-pilot.json').read_text())
        _, w, cases = bm.plan(c)
        self.assertEqual(len(w), 48)
        self.assertEqual(len(cases), 48*121)
        self.assertEqual(sum(c['candidate']['method']=='segmented' for c in cases), 48)
        c = json.loads((ROOT/'configs/benchmark-full-coverage.json').read_text())
        _, w, cases = bm.plan(c)
        # q=12/16: k=q coincides three ways, k=q-1 two ways.
        self.assertEqual(len(w), (12+5*16)*3-6)
        self.assertEqual(len(cases), len(w)*3)

    def test_order_and_determinism(self):
        c = self.config()
        c.update(q=[12], k=[3], positions=dict(patterns=['explicit','random'], explicit=[[7,1,4]],
                                             reverse=True, seed=42, random_count=8))
        one = bm.plan(c)
        self.assertEqual(one, bm.plan(c))
        self.assertIn([7,1,4], [w['positions'] for w in one[1]])
        self.assertIn([4,1,7], [w['positions'] for w in one[1]])
        self.assertEqual(len({w['workload_id'] for w in one[1]}), len(one[1]))
        for w in one[1]:
            if w['position_pattern'].startswith('random'):
                self.assertGreater(max(w['positions'])-min(w['positions']), 2)
        for q in range(3,9):
            for k in range(2,q):
                for _, p in bm.positions(q,k,dict(patterns=['random'], random_count=10)):
                    self.assertGreater(max(p)-min(p), k-1)

    def test_invalid(self):
        for patch in [dict(q=[0]), dict(q=[60]), dict(q=[True]), dict(k=[0]), dict(k=[13]),
                      dict(positions=dict(patterns=['explicit'],explicit=[[1,1]])),
                      dict(positions=dict(patterns=['explicit'],explicit=[[99]])),
                      dict(positions=dict(patterns=['explicit'],explicit=[[7,1]])),
                      dict(version=2), dict(measurement=dict(repetitions=0)),
                      dict(candidates=dict(methods=['typo'])), dict(candidates=dict(block_sizes=[128.0])), dict(unknown=True)]:
            c = self.config()
            c.update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                bm.plan(c)

    def test_filters_and_ids(self):
        c = self.config()
        c['candidates'] = dict(methods=['custom','segmented'], block_sizes=[128], sm_multipliers=[2],
                               storage=['registers','shared'], layouts=['thread_major','bin_major'])
        _, w, cases = bm.plan(c)
        self.assertEqual(len(cases), len(w)*4)
        _, _, reordered = bm.plan(dict(c, q=list(reversed(c['q']))))
        self.assertEqual({x['case_key'] for x in cases}, {x['case_key'] for x in reordered})
        self.assertEqual(len({x['case_key'] for x in cases}), len(cases))

    def test_resource_limits(self):
        settings, _, cases = bm.plan(dict(self.config(), candidates=dict(methods=['custom'], storage=['global'],
                                                                         block_sizes=[128],sm_multipliers=[1])))
        d = dict(SM_count=108, shared_memory_optin_per_block=163840, initial_free_device_bytes=80*1024**3)
        c = cases[0]
        r = bm.estimate(c, d, settings)
        self.assertEqual(r['estimated_workspace_bytes'], 8*108*128*2+8*108*2+8*2+4)
        self.assertIsNone(r['resource_rejection_reason'])
        self.assertIn('workspace', bm.estimate(c,d,dict(settings,workspace_limit_bytes=1))['resource_rejection_reason'])
        segmented = copy.deepcopy(c)
        segmented['candidate'] = dict(method='segmented')
        self.assertIn('memory limit', bm.estimate(segmented,None,dict(settings,workspace_limit_bytes=0))['resource_rejection_reason'])
        self.assertIn('device memory', bm.estimate(c,dict(d,initial_free_device_bytes=1),settings)['resource_rejection_reason'])
        reg = copy.deepcopy(c)
        reg['candidate']['storage']='registers'
        self.assertIn('audit', bm.estimate(reg,d,settings)['resource_rejection_reason'])

    def test_grid_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.json'
            config.write_text(json.dumps(self.config()))
            command = [sys.executable, str(ROOT/'tools/benchmark_matrix.py'), '--matrix-config', str(config),
                       '--dry-run', '--results', str(Path(tmp)/'grid'), '--grid-block-counts', '32,64']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((Path(tmp)/'grid.plan.json').read_text())
            self.assertEqual(len(data['cases']), len(bm.plan(self.config())[2])) # Segmented stays deduplicated.
            config.write_text(json.dumps(dict(self.config(), candidates=dict(methods=['custom'],
                                  storage=['global'], layouts=['thread_major'], block_sizes=[256]))))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((Path(tmp)/'grid.plan.json').read_text())
            self.assertEqual({c['candidate']['grid_blocks'] for c in data['cases']}, {32,64})
            self.assertEqual({c['estimated_blocks'] for c in data['cases']}, {32,64})
            for counts in ('0', '32,32', '32,', '-1', 'bad'):
                self.assertNotEqual(subprocess.run(command[:-1]+[counts], capture_output=True).returncode, 0)
            config.write_text(json.dumps(dict(self.config(), candidates=dict(sm_multipliers=[1]))))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_exact_grid_counts(self):
        spec = dict(methods=['custom','cub','segmented'], storage=['global'], layouts=['thread_major'],
                    block_sizes=[128,256], grid_block_counts=[32,64,128,256,512,1024])
        settings, workloads, cases = bm.plan(dict(self.config(), candidates=spec))
        self.assertEqual(len(cases), len(workloads)*25)
        self.assertEqual(len({c['candidate_id'] for c in cases}), 25)
        device = dict(SM_count=108, shared_memory_optin_per_block=163840,
                      initial_free_device_bytes=80*1024**3)
        for case in cases:
            c, w = case['candidate'], case['workload']
            if c['method'] == 'segmented':
                self.assertEqual(c, dict(method='segmented'))
                continue
            self.assertNotIn('sm_multiplier', c)
            r = bm.estimate(case, device, settings)
            expected = 8*c['grid_blocks']*(c['block_size']+1)*w['bins']+8*w['bins']+4*w['k']
            self.assertEqual(r['estimated_workspace_bytes'], expected)
            self.assertEqual(r['estimated_blocks'], c['grid_blocks'])
            self.assertEqual(bm.estimate(case, None, settings)['estimated_workspace_bytes'], expected)
            self.assertEqual(bm.estimate(case, dict(device, SM_count=7), settings), r)
            self.assertIn('workspace', bm.estimate(case, device, dict(settings, workspace_limit_bytes=expected-1))['resource_rejection_reason'])
        entry = dict(method='custom', storage='global', layout='thread_major', block_size=256,
                     grid_blocks=32, strategy='two_kernel')
        self.assertEqual(bm.candidates(dict(include=[entry])), [entry])
        for bad in [dict(spec, sm_multipliers=[1]), dict(spec, grid_block_counts=[]),
                    *[dict(spec, grid_block_counts=v) for v in ([0],[-1],[True],[32.0],[32,32])],
                    dict(include=[dict(entry, sm_multiplier=1)])]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bm.candidates(bad)
        huge = copy.deepcopy(cases[0])
        huge['candidate']['grid_blocks'] = 1 << 31
        self.assertIn('grid', bm.estimate(huge, device, settings)['resource_rejection_reason'])

    def test_truncated_journal_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'r'
            journal=Path(str(prefix)+'.journal.jsonl')
            header=dict(type='header',metadata=dict(benchmark_run_id='test'),fingerprint='test')
            with journal.open('w') as f:
                bm.append_record(f, header)
                bm.append_record(f, dict(type='result',row=dict(case_key='one',status='success',positions=[7,1],raw_elapsed_ms=[1.2])))
                bm.append_record(f, dict(type='result',row=dict(case_key='one',status='success',positions=[7,1],raw_elapsed_ms=[1.3])))
                f.write('{"type":')
            h, rows, complete=bm.read_journal(journal)
            self.assertEqual(len(rows),1)
            self.assertLess(complete, journal.stat().st_size)
            bm.export_results(prefix,h,rows)
            self.assertEqual(len(json.loads(Path(str(prefix)+'.json').read_text())['results']),1)
            with Path(str(prefix)+'.csv').open() as f:
                values=list(csv.DictReader(f))
            self.assertEqual(json.loads(values[0]['positions']),[7,1])
            self.assertEqual(json.loads(values[0]['raw_elapsed_ms']),[1.3])
            journal.write_text(journal.read_text()+'}\n')
            with self.assertRaises(ValueError):
                bm.read_journal(journal)

    def test_resume_integration(self):
        # A host fake tests orchestration/protocol only; no GPU results claimed.
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            config=self.config()
            config.update(q=[12],k=[1],positions=dict(patterns=['low']),
                          candidates=dict(methods=['custom'],storage=['global'],layouts=['thread_major'],
                                          block_sizes=[256],grid_block_counts=[32,64]))
            conf=tmp/'plan.json'
            conf.write_text(json.dumps(config))
            binary=tmp/'fake'
            binary.write_text('''#!/usr/bin/env python3
import json,sys,pathlib,os
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
if '--device-info' in args:
 print(json.dumps(dict(SM_count=1,shared_memory_optin_per_block=100000,initial_free_device_bytes=1000000000,L2_bytes=1000,device_uuid=os.environ.get('FAKE_GPU','fake'),build='fake')))
else:
 with pathlib.Path(__file__).with_suffix('.calls').open('a') as f: f.write('call\\n')
 for i,line in enumerate(pathlib.Path(args['--case-plan']).read_text().splitlines()):
  fields=line.split()
  assert len(fields)==9 and fields[7]=='1' and int(fields[8]) in (32,64)
  print('MEASUREMENT_ROW '+json.dumps(dict(case_id=i,blocks=int(fields[8]),status='success',timing_status='stopped' if os.environ.get('FAKE_PARTIAL') else 'success',workspace_bytes=10,raw_elapsed_ms=[1.0],completed_repetitions=1)),flush=True)
  if os.environ.get('FAKE_CRASH'): sys.exit(9)
''')
            binary.chmod(0o755)
            command=[sys.executable,str(ROOT/'tools/benchmark_matrix.py'),'--matrix-config',str(conf),
                     '--binary',str(binary),'--results',str(tmp/'r')]
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            second=subprocess.run(command+['--resume'],capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertEqual(binary.with_suffix('.calls').read_text(),'call\n')
            # A torn last line is removed before append; exports remain unique.
            journal=tmp/'r.journal.jsonl'
            with journal.open('a') as f: f.write('{')
            self.assertEqual(subprocess.run(command+['--resume'],capture_output=True).returncode,0)
            rows=json.loads((tmp/'r.json').read_text())['results']
            self.assertEqual(len(rows),2)
            self.assertEqual({r['grid_blocks'] for r in rows}, {32,64})
            self.assertTrue(all(r['blocks']==r['grid_blocks'] and r['sm_multiplier'] is None for r in rows))
            with (tmp/'r.csv').open() as f:
                exported=list(csv.DictReader(f))
            self.assertEqual({json.loads(r['grid_blocks']) for r in exported},{32,64})
            altered=copy.deepcopy(config)
            altered['candidates']['grid_block_counts']=[32,128]
            conf.write_text(json.dumps(altered))
            changed=subprocess.run(command+['--resume'],capture_output=True,text=True)
            self.assertIn('incompatible resume',changed.stderr)
            conf.write_text(json.dumps(config))
            changed=subprocess.run(command+['--resume','--repetitions','2'],capture_output=True,text=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('incompatible resume',changed.stderr)
            changed=subprocess.run(command+['--resume'],capture_output=True,text=True,env=dict(os.environ,FAKE_GPU='other'))
            self.assertIn('incompatible resume',changed.stderr)
            # An interrupted child leaves its completed first case durable.
            conf.write_text(json.dumps(dict(config,positions=dict(patterns=['low','high']))))
            interrupted=command[:-1]+[str(tmp/'crash')]
            result=subprocess.run(interrupted,capture_output=True,text=True,env=dict(os.environ,FAKE_CRASH='1'))
            self.assertNotEqual(result.returncode,0)
            h, partial, _=bm.read_journal(tmp/'crash.journal.jsonl')
            self.assertEqual(len(partial),1)
            result=subprocess.run(interrupted+['--resume'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads((tmp/'crash.json').read_text())['results']
            self.assertEqual(len(data),4)
            self.assertEqual(len({r['case_key'] for r in data}),4)
            # Budget-stopped timings retain raw samples, then retry on resume.
            partial_cmd=command[:-1]+[str(tmp/'partial')]
            result=subprocess.run(partial_cmd,capture_output=True,text=True,env=dict(os.environ,FAKE_PARTIAL='1'))
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run(partial_cmd+['--resume'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            h, data, _=bm.read_journal(tmp/'partial.journal.jsonl')
            self.assertTrue(all(bm.completed(r) for r in data.values()))
            conf.write_text(json.dumps(config))
            binary.write_text(binary.read_text()+'\n# different build\n')
            changed=subprocess.run(command+['--resume'],capture_output=True,text=True)
            self.assertIn('incompatible resume',changed.stderr)


if __name__=='__main__':
    unittest.main()
