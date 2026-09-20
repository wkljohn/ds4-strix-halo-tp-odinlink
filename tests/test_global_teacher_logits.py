#!/usr/bin/env python3
"""Exercise global teacher comparison using complete synthetic process evidence."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'scripts/compare-teacher-logits.py'
fixtures = runpy.run_path(str(ROOT / 'tests/test_compare_teacher_logits.py'))
GLOBAL = 'DS4_ROCM_GLM5_Q4K_PREFILL_GLOBAL'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GlobalTeacher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'prompt.txt').write_text('context fixture')
        (self.root / 'target.txt').write_text('continuation fixture')
        self.fixture = self.root / 'fixture.tsv'
        self.fixture.write_text('case_000\tprompt.txt\ttarget.txt\n')
        self.dirs = [self.root / 'reference', self.root / 'candidate']
        self.meta = []
        for index, directory in enumerate(self.dirs):
            directory.mkdir()
            fixtures['dump'](directory / 'decode_000000.logits.json', [0., 1., 2., 3.],
                             source='ds4-score-official-frozen-teacher')
            fixtures['score_manifest'](directory / 'manifest', arm='kda-tp')
            meta = dict(line.split('=', 1) for line in (directory / 'manifest').read_text().splitlines())
            tag = directory.name
            run_id = f'12345678-1234-1234-1234-{index:012d}'
            env = dict(DS4_GLM5_KDA_TP='1', DS4_GLM5_KDA_OUTPUT_KSLICE='0',
                       DS4_GLM5_NEXT_PREFILL_BATCH='1024', DS4_GLM5_NATIVE_DRAFT='0',
                       DS4_ROCM_GLM5_Q4K_PREFILL_GROUPED='1',
                       DS4_ROCM_GLM5_Q4K_PREFILL_PARTITION='256',
                       DS4_TP_RDMA_LOGITS='1', DS4_GLM5_SPARSE_BATCH_BRIDGE='1',
                       DS4_GLM5_PREFILL_PROOF='1', **{GLOBAL: str(index)})
            encoded = ' '.join(f'{k}={v}' for k, v in env.items())
            meta.update(tag=tag, run_id=run_id, cases='1', teacher_positions='1',
                        context='9216', model_arch='glm5-next', inference_source_commit='a' * 40,
                        ds4_sha256='b' * 64, scorer_sha256='c' * 64,
                        quality_input_sha256=digest(self.fixture),
                        completion_schema='quality-terminal-v1',
                        worker_supervisor_sha256='d' * 64, quality_launcher_sha256='e' * 64,
                        extra_env=encoded)
            self.meta.append(meta)
            self.bind(index, 'scores', directory / (tag + '.tsv'),
                      'id\tprompt_tokens\ttarget_tokens\ncase_000\t2048\t1\n')
            for rank, rank_id in (('coordinator', 0), ('worker', 1)):
                meta[rank + '_env'] = encoded + ' DS4_BENCH_RUN_ID=' + run_id
                self.bind(index, rank + '_status', directory / f'{rank}-{tag}.status',
                          'exit_code=0\nsignal=0\n')
                lines = [f'ds4-tp: benchmark run_id={run_id}',
                         'ds4-tp: transport proof requested=rdma active=rdma payload_fallback_calls=0 failed=0',
                         'rdma GID index 3 (RoCE v2)', 'expanded_weight_cache_bytes=0',
                         f'GLM5 prefill execution rank={rank_id} start=0 prompt_tokens=2048 '
                         'requested_batch=1024 batched_tiles=2 batched_rows=2048 scalar_rows=0 min_tile=1024 max_tile=1024']
                lines.append(f'global expert domain engaged rank={rank_id} rows=1024 groups=1'
                             if index else 'grouped prefill engaged rows=1024 groups=4 physical_experts=288')
                self.bind(index, rank + '_log', directory / f'{rank}-{tag}.log', '\n'.join(lines) + '\n')
            self.inventory(index)

    def bind(self, index, name, path, data):
        path.write_text(data)
        self.meta[index][name + '_path'] = str(path)
        self.meta[index][name + '_sha256'] = digest(path)

    def inventory(self, index):
        directory = self.dirs[index]
        inventory = directory / 'files.sha256'
        inventory.write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(directory.glob('decode_*.logits.json'))))
        self.meta[index]['files_sha256'] = digest(inventory)

    def run_tool(self, mode='q4k-global', *extra):
        for directory, meta in zip(self.dirs, self.meta):
            (directory / 'manifest').write_text(''.join(f'{k}={v}\n' for k, v in meta.items()))
        return subprocess.run([sys.executable, str(TOOL), *map(str, self.dirs),
                               '--score-arm-mode', mode, '--score-fixture', str(self.fixture),
                               '--score-root', str(self.root), '--score-rdma-gid', '3',
                               *extra], text=True, capture_output=True)

    def test_complete_diagnostic_and_numeric_drift(self):
        fixtures['dump'](self.dirs[1] / 'decode_000000.logits.json', [0., 1., 4., 3.],
                         source='ds4-score-official-frozen-teacher')
        self.inventory(1)
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value['mode'], 'diagnostic')
        self.assertEqual(value['argmax_mismatches'], 1)
        self.assertFalse(value['capture_time_fixture_bound'])
        self.assertEqual(len(value['fixture_content_sha256_at_comparison']), 64)

    def test_repeat_same_setting_only(self):
        self.assertIn('same-mode repeat', self.run_tool('q4k-global-repeat').stderr)
        for field in ('extra_env', 'coordinator_env', 'worker_env'):
            self.meta[1][field] = self.meta[1][field].replace(GLOBAL + '=1', GLOBAL + '=0')
        for rank in ('coordinator', 'worker'):
            path = Path(self.meta[1][rank + '_log_path'])
            text = '\n'.join(line for line in path.read_text().splitlines() if 'global expert domain' not in line)
            self.bind(1, rank + '_log', path, text + '\ngrouped prefill engaged rows=1024 groups=4 physical_experts=288\n')
        result = self.run_tool('q4k-global-repeat')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('GLOBAL=0 versus GLOBAL=1', self.run_tool().stderr)

    def test_threshold_and_quality_bypass_refused(self):
        self.assertIn('diagnostic-only', self.run_tool('q4k-global', '--thresholds', '/absent').stderr)
        self.assertIn('forbids', self.run_tool('q4k-global', '--allow-quality-difference').stderr)

    def test_mutated_dump_or_terminal_refused(self):
        path = self.dirs[1] / 'decode_000000.logits.json'
        path.write_text(path.read_text() + ' ')
        self.assertIn('changed teacher dump', self.run_tool().stderr)
        self.inventory(1)
        self.bind(1, 'worker_status', Path(self.meta[1]['worker_status_path']), 'exit_code=7\nsignal=0\n')
        self.assertIn('unsuccessful', self.run_tool().stderr)

    def test_rank_environment_mismatch_refused(self):
        self.meta[1]['worker_env'] += ' UNRELATED_SETTING=1'
        self.assertIn('effective settings differ', self.run_tool().stderr)

    def test_unengaged_or_scalar_prefill_refused(self):
        path = Path(self.meta[1]['coordinator_log_path'])
        original = path.read_text()
        self.bind(1, 'coordinator_log', path, original.replace('global expert domain engaged', 'not engaged'))
        self.assertIn('engagement disagrees', self.run_tool().stderr)
        self.bind(1, 'coordinator_log', path, original.replace('batched_rows=2048 scalar_rows=0',
                                                             'batched_rows=1024 scalar_rows=1024'))
        self.assertIn('declared batched prefix', self.run_tool().stderr)

    def test_rebound_fallback_and_quality_mode_refused(self):
        path = Path(self.meta[1]['worker_log_path'])
        original = path.read_text()
        self.bind(1, 'worker_log', path, original.replace('payload_fallback_calls=0', 'payload_fallback_calls=1'))
        self.assertIn('RoCE/zero-fallback', self.run_tool().stderr)
        self.bind(1, 'worker_log', path, original)
        fixtures['dump'](self.dirs[1] / 'decode_000000.logits.json', [0., 1., 2., 3.],
                         quality=True, source='ds4-score-official-frozen-teacher')
        self.inventory(1)
        self.assertIn('ordinary production arithmetic', self.run_tool().stderr)

    def test_gid_and_worker_engagement_refused(self):
        path = Path(self.meta[1]['worker_log_path'])
        original = path.read_text()
        self.bind(1, 'worker_log', path, original.replace('GID index 3', 'GID index 4'))
        self.assertIn('RoCE/zero-fallback', self.run_tool().stderr)
        self.bind(1, 'worker_log', path, original.replace('global expert domain engaged', 'not engaged'))
        self.assertIn('engagement disagrees', self.run_tool().stderr)

    def test_bad_prerequisites_and_extra_case_refused(self):
        for before, after in (('GROUPED=1', 'GROUPED=0'), ('BRIDGE=1', 'BRIDGE=0'),
                              ('PROOF=1', 'PROOF=0'), ('KSLICE=0', 'KSLICE=1'),
                              ('BATCH=1024', 'BATCH=256'), ('DRAFT=0', 'DRAFT=6')):
            originals = [m['coordinator_env'] for m in self.meta]
            for m in self.meta:
                m['coordinator_env'] = m['coordinator_env'].replace(before, after)
            with self.subTest(before=before):
                self.assertIn('prerequisites', self.run_tool().stderr)
            for m, original in zip(self.meta, originals):
                m['coordinator_env'] = original
        self.meta[1]['cases'] = '2'
        self.assertNotEqual(self.run_tool().returncode, 0)

    def test_build_and_declared_selector_mismatch_refused(self):
        for field in ('source_commit', 'ds4_sha256'):
            old = self.meta[1][field]
            self.meta[1][field] = 'wrong'
            with self.subTest(field=field):
                self.assertIn('manifest ' + field, self.run_tool().stderr)
            self.meta[1][field] = old
        self.meta[1]['extra_env'] = self.meta[1]['extra_env'].replace(GLOBAL + '=1', '')
        self.assertIn('explicit declared GLOBAL', self.run_tool().stderr)

    def test_missing_control_and_coordinator_proofs_refused(self):
        path = Path(self.meta[0]['worker_log_path'])
        original = path.read_text()
        self.bind(0, 'worker_log', path, original.replace('grouped prefill engaged', 'not engaged'))
        self.assertIn('control lacks grouped', self.run_tool().stderr)
        self.bind(0, 'worker_log', path, original)
        path = Path(self.meta[1]['coordinator_log_path'])
        self.bind(1, 'coordinator_log', path, path.read_text().replace('GLM5 prefill execution', 'other record'))
        self.assertIn('prefill proof', self.run_tool().stderr)

    def test_fixture_content_rewrite_remains_explicitly_diagnostic(self):
        first = json.loads(self.run_tool().stdout)
        (self.root / 'prompt.txt').write_text('changed after capture: this is not capture-time binding')
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        second = json.loads(result.stdout)
        self.assertNotEqual(first['fixture_content_sha256_at_comparison'],
                            second['fixture_content_sha256_at_comparison'])
        self.assertFalse(second['capture_time_fixture_bound'])


if __name__ == '__main__':
    unittest.main()
