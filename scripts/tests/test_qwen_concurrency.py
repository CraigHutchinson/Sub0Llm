import importlib.util
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'compare_qwen_concurrency.py'
spec = importlib.util.spec_from_file_location('compare_qwen_concurrency', PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def manifest(port):
    return dict(engine_kind='llama-server', url=f'http://127.0.0.1:{port}', engine_id='engine', engine_sha256='a'*64,
                model_id='model', model_sha256='b'*64, server_build='build', server_model_path='/model.gguf',
                calibrator='/calibrator', server_pid=os.getpid(), cpus=[0], context_per_slot=128, server_cap=2, prompt_ids=[1, 2, 3],
                tokens_per_request=2, eos_token_ids=[43], seed=17, concurrency=[1, 2], rounds=3, timeout_seconds=2)


class Handler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Deliberately invalid streams are closed early by the client.

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path == '/props':
            value = dict(total_slots=2, build_info='build', model_path='/model.gguf')
        elif self.path == '/slots':
            value = [dict(id=i, n_ctx=128, is_processing=self.server.busy) for i in range(2)]
        else:
            self.send_error(404)
            return
        encoded = json.dumps(value).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.lock:
            self.server.payloads.append(body)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        progress = dict(content='', tokens=[0], stop=False, tokens_predicted=0,
                        prompt_progress=dict(total=3, cache=1 if self.server.mode == 'cached' else 0, processed=0, time_ms=0))
        if self.server.mode != 'missing_progress':
            self.wfile.write(('data: '+json.dumps(progress)+'\n\n').encode())
            self.wfile.flush()
        for i, token in enumerate((42, 43)):
            event = dict(content='x', tokens=[token], stop=False)
            self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
            self.wfile.flush()
        if self.server.mode != 'missing_final':
            final = dict(content='', tokens=[], stop=True, stop_type='limit', truncated=self.server.mode == 'truncated',
                         tokens_cached=5, id_slot=body['id_slot'], tokens_predicted=2, tokens_evaluated=3,
                         timings=dict(prompt_ms=1.0, predicted_ms=2.0))
            if self.server.mode == 'wrong_count':
                final['tokens_predicted'] = 1
            if self.server.mode == 'wrong_slot':
                final['id_slot'] = 99
            if self.server.mode == 'no_timings':
                del final['timings']
            self.wfile.write(('data: '+json.dumps(final)+'\n\n').encode())
            self.wfile.flush()


class ConcurrencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.mode = 'ok'
        cls.server.busy = False
        cls.server.lock = threading.Lock()
        cls.server.payloads = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.mode = 'ok'
        self.server.busy = False
        self.server.payloads.clear()
        self.spec = manifest(self.server.server_port)

    def test_worker_cpu_budget_and_memory(self):
        with patch.object(mod.Path, 'iterdir', return_value=iter([Path('1'), Path('2')])), \
             patch.object(mod.os, 'sched_getaffinity', side_effect=[{0}, {1}]):
            with self.assertRaisesRegex(ValueError, 'worker affinity'):
                mod.check_thread_affinities(os.getpid(), [0])
        values = mod.memory_snapshot(os.getpid())
        self.assertGreater(values['rss_kib'], 0)
        self.assertGreaterEqual(values['lifetime_peak_rss_kib'], values['rss_kib'])
        self.assertTrue(values['pss_kib'] is None or values['pss_kib'] > 0)

    def test_stream_and_two_slots(self):
        host, port = mod.validate(self.spec)
        slots, _ = mod.inspect_server(self.spec, host, port)
        result = mod.run_round(self.spec, host, port, slots, 2)
        self.assertEqual(len(result['requests']), 2)
        self.assertGreater(result['aggregate_tokens_s'], 0)
        self.assertEqual({p['id_slot'] for p in self.server.payloads}, {0, 1})
        for p in self.server.payloads:
            self.assertEqual(p['prompt'], [1, 2, 3])
            self.assertFalse(p['cache_prompt'])
            self.assertTrue(p['ignore_eos'])
            self.assertEqual(p['n_predict'], 2)
        self.assertEqual({p['seed'] for p in self.server.payloads}, {17, 18})
        for row in result['requests']:
            self.assertEqual(row['tokens'], 2)
            self.assertEqual(len(row['content_sha256']), 64)
            self.assertEqual(len(row['token_sha256']), 64)
            self.assertGreaterEqual(row['wall_s'], row['ttft_s'])

    def test_reject_incomplete_stream(self):
        for mode in ('missing_final', 'truncated', 'wrong_count', 'cached', 'missing_progress', 'wrong_slot', 'no_timings'):
            with self.subTest(mode=mode):
                self.server.mode = mode
                with self.assertRaises(ValueError):
                    mod.run_round(self.spec, '127.0.0.1', self.server.server_port, [0], 1)

    def test_reject_busy_and_capacity_mismatch(self):
        self.server.busy = True
        with self.assertRaises(ValueError):
            mod.inspect_server(self.spec, '127.0.0.1', self.server.server_port)
        self.server.busy = False
        self.spec['server_cap'] = 3
        with self.assertRaises(ValueError):
            mod.inspect_server(self.spec, '127.0.0.1', self.server.server_port)

    def test_invalid_manifest(self):
        changes = [dict(url='http://example.com:8080'), dict(engine_kind='sub0llm-native'),
                   dict(concurrency=[1, 4]), dict(rounds=2), dict(context_per_slot=5),
                   dict(engine_sha256='not-a-hash'), dict(prompt_ids=[True]), dict(server_pid=0), dict(cpus=[1, 1])]
        for change in changes:
            with self.subTest(change=change):
                bad = dict(self.spec, **change)
                with self.assertRaises(ValueError):
                    mod.validate(bad)

    def test_answer_prefix_ignores_only_post_eos_tokens(self):
        self.assertEqual(mod.answer_prefix([42, 43, 0], [43]), [42, 43])
        self.assertEqual(mod.answer_prefix([42, 43, 99], [43]), [42, 43])
        self.assertNotEqual(mod.answer_prefix([41, 43, 99], [43]), [42, 43])
        self.assertEqual(mod.answer_prefix([0, 42], [43]), [0, 42])

    def test_percentiles(self):
        self.assertEqual(mod.percentile([5, 2, 4, 1, 3], .95), 5)
        self.assertEqual(mod.percentile([5, 2, 4, 1, 3], .5), 3)

    def test_full_rounds_exclude_warmups(self):
        with tempfile.TemporaryDirectory() as tmp:
            calibrator = Path(tmp) / 'calibrator'
            calibrator.touch()
            self.spec['calibrator'] = str(calibrator)
            target = Path(tmp) / 'result.json'
            identity = dict(pid=self.spec['server_pid'], start_ticks=123, cpus=[0])
            state = SimpleNamespace(untrusted=False, source='linux', as_dict=lambda: {'state': 'ac'})
            with patch.object(mod, 'PerfLock', return_value=contextlib.nullcontext()), \
                 patch.object(mod, 'wait_for_quiet', return_value=(0.0, 0.0, True)), \
                 patch.object(mod, 'steady_calibration', return_value=100), \
                 patch.object(mod, 'server_identity', return_value=identity), \
                 patch.object(mod, 'process_stat', return_value={'start_ticks': 123, 'cpu_ticks': 0}), \
                 patch.object(mod.os, 'sched_getaffinity', return_value={0}), \
                 patch.object(mod, 'cpu_snapshot', return_value=(0, 0)), \
                 patch.object(mod, 'background_fraction', return_value=0), \
                 patch.object(mod, 'power_state', return_value=state):
                result = mod.run(self.spec, target)
            self.assertEqual(len(result['samples']), 8)
            self.assertEqual(sum(s['warmup'] for s in result['samples']), 2)
            self.assertEqual(set(result['summary']), {'1', '2'})
            self.assertEqual([s['concurrency'] for s in result['samples']], [1, 2, 1, 2, 2, 1, 1, 2])
            self.assertGreater(result['summary']['2']['latency_p50_s'], 0)
            self.assertEqual(json.loads(target.read_text())['schema'], 'qwen-concurrency-1')

    def test_identity_hash_and_affinity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model.gguf'
            model.write_bytes(b'model fixture')
            self.spec['server_model_path'] = str(model)
            self.spec['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
            self.spec['engine_sha256'] = mod.sha256_file(f'/proc/{os.getpid()}/exe')
            self.spec['cpus'] = sorted(os.sched_getaffinity(os.getpid()))
            self.assertEqual(mod.server_identity(self.spec)['pid'], os.getpid())
            self.spec['model_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                mod.server_identity(self.spec)
            self.spec['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
            self.spec['cpus'] = [max(self.spec['cpus']) + 1]
            with self.assertRaises(ValueError):
                mod.server_identity(self.spec)

    def test_three_tainted_attempts_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            calibrator = Path(tmp) / 'calibrator'
            calibrator.touch()
            self.spec['calibrator'] = str(calibrator)
            target = Path(tmp) / 'result.json'
            identity = dict(pid=self.spec['server_pid'], start_ticks=123, cpus=[0])
            state = SimpleNamespace(untrusted=False, source='linux', as_dict=lambda: {'state': 'ac'})
            with patch.object(mod, 'PerfLock', return_value=contextlib.nullcontext()), \
                 patch.object(mod, 'wait_for_quiet', return_value=(0.0, 0.0, True)), \
                 patch.object(mod, 'steady_calibration', return_value=100), \
                 patch.object(mod, 'server_identity', return_value=identity), \
                 patch.object(mod, 'process_stat', return_value={'start_ticks': 123, 'cpu_ticks': 0}), \
                 patch.object(mod.os, 'sched_getaffinity', return_value={0}), \
                 patch.object(mod, 'cpu_snapshot', return_value=(0, 0)), \
                 patch.object(mod, 'background_fraction', return_value=.06), \
                 patch.object(mod, 'power_state', return_value=state):
                with self.assertRaisesRegex(ValueError, 'three tainted'):
                    mod.run(self.spec, target)
            partial = json.loads(target.with_suffix('.partial.json').read_text())
            self.assertFalse(target.exists())
            self.assertEqual(len(partial['samples']), 3)
            self.assertTrue(all(s['tainted'] for s in partial['samples']))


if __name__ == '__main__':
    unittest.main()
