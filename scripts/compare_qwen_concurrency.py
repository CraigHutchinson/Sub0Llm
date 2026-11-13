#!/usr/bin/env python3
"""Measure simultaneous local llama-server requests; rates include prompt and decode wall time."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/perf'))
from sub0perf.common import atomic_write_text
from sub0perf.load import wait_for_quiet
from sub0perf.lock import PerfLock
from sub0perf.power import power_state
from sub0perf.runner import measure, steady_calibration


def validate(spec):
    if not isinstance(spec, dict) or spec.get('engine_kind') != 'llama-server':
        raise ValueError('only a declared llama-server is supported; native pooled Sub0Llm is not implemented')
    url = urlsplit(spec.get('url', ''))
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', '::1', 'localhost') or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
        raise ValueError('URL must be an unauthenticated HTTP loopback origin')
    try:
        port = url.port
    except ValueError as exc:
        raise ValueError('invalid server port') from exc
    if not port:
        raise ValueError('explicit server port required')
    for key in ('engine_id', 'engine_sha256', 'model_id', 'model_sha256', 'server_build', 'server_model_path', 'calibrator'):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError('missing identity: ' + key)
    for key in ('engine_sha256', 'model_sha256'):
        value = spec[key]
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid SHA-256: ' + key)
    for key in ('context_per_slot', 'server_cap', 'tokens_per_request', 'seed'):
        if type(spec.get(key)) is not int or spec[key] < (0 if key == 'seed' else 1):
            raise ValueError('invalid integer: ' + key)
    if spec['seed'] > 2**32 - 9 or spec['server_cap'] > 1024:
        raise ValueError('seed or server capacity out of range')
    if type(spec.get('server_pid')) is not int or spec['server_pid'] < 1:
        raise ValueError('a live server_pid is required')
    cpus = spec.get('cpus')
    if not isinstance(cpus, list) or not cpus or any(type(c) is not int or c < 0 for c in cpus) or len(cpus) != len(set(cpus)):
        raise ValueError('cpus must be distinct nonnegative CPU numbers')
    ids = spec.get('prompt_ids')
    if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < 2**31 for x in ids):
        raise ValueError('prompt_ids must be a nonempty array of token IDs')
    if len(ids) + spec['tokens_per_request'] >= spec['context_per_slot']:
        raise ValueError('prompt and output must fit inside the per-slot context without truncation')
    levels = spec.get('concurrency')
    if not isinstance(levels, list) or not levels or any(type(v) is not int or v not in (1, 2, 4, 8) or v > spec['server_cap'] for v in levels) or len(set(levels)) != len(levels):
        raise ValueError('concurrency must contain distinct values from 1,2,4,8 within server_cap')
    if type(spec.get('rounds')) is not int or spec['rounds'] < 3:
        raise ValueError('at least three measured rounds required')
    if type(spec.get('timeout_seconds')) not in (int, float) or not math.isfinite(spec['timeout_seconds']) or not 0 < spec['timeout_seconds'] <= 86400:
        raise ValueError('invalid timeout')
    return url.hostname, port


def process_stat(pid):
    # comm can contain spaces and parentheses; the last ')' ends field 2.
    text = Path(f'/proc/{pid}/stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    if len(fields) < 20 or fields[0] in ('Z', 'X'):
        raise ValueError('server process not alive')
    return {'start_ticks': int(fields[19]), 'cpu_ticks': int(fields[11]) + int(fields[12])}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def server_identity(spec):
    pid = spec['server_pid']
    before = process_stat(pid)
    exe = f'/proc/{pid}/exe'
    model_path = Path(spec['server_model_path'])
    if not model_path.is_absolute():
        model_path = Path(f'/proc/{pid}/cwd') / model_path
    affinity = sorted(os.sched_getaffinity(pid))
    if affinity != sorted(spec['cpus']):
        raise ValueError('server CPU affinity differs from declared CPUs')
    engine = sha256_file(exe)
    model = sha256_file(model_path)
    after = process_stat(pid)
    if before['start_ticks'] != after['start_ticks'] or engine != spec['engine_sha256'] or model != spec['model_sha256']:
        raise ValueError('server process or engine/model SHA-256 differs from manifest')
    return {'pid': pid, 'start_ticks': before['start_ticks'], 'engine_sha256': engine,
            'model_sha256': model, 'cpus': affinity, 'model_path': str(model_path)}


def memory_snapshot(pid):
    values = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            values[key] = int(value.split()[0])
    if set(values) != {'VmRSS', 'VmHWM'}:
        raise ValueError('server RSS accounting unavailable')
    pss = None
    try:
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            if line.startswith('Pss:'):
                pss = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return dict(rss_kib=values['VmRSS'], lifetime_peak_rss_kib=values['VmHWM'], pss_kib=pss)


def check_thread_affinities(pid, cpus):
    allowed = set(cpus)
    checked = 0
    for task in Path(f'/proc/{pid}/task').iterdir():
        try:
            actual = set(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue  # A transient HTTP worker may finish during the snapshot.
        if not actual or not actual <= allowed:
            raise ValueError('server worker affinity exceeds declared CPU budget')
        checked += 1
    if not checked:
        raise ValueError('server threads unavailable')
    return checked


def cpu_snapshot(cpus=None):
    wanted = {'cpu'} if cpus is None else {f'cpu{c}' for c in cpus}
    found = {}
    for line in Path('/proc/stat').read_text().splitlines():
        label, *numbers = line.split()
        if label in wanted:
            values = [int(v) for v in numbers[:8]]
            if len(values) < 5:
                raise ValueError('incomplete CPU accounting')
            total = sum(values)
            found[label] = (total - values[3] - values[4], total)
    if set(found) != wanted:
        raise ValueError('declared CPUs unavailable in /proc/stat')
    return tuple(sum(value[i] for value in found.values()) for i in (0, 1))


def background_fraction(before, after, server_ticks):
    busy, total = after[0] - before[0], after[1] - before[1]
    if total <= 0 or busy < 0 or server_ticks < 0:
        raise ValueError('invalid CPU accounting interval')
    return max(0.0, (busy - server_ticks) / total)


def request_json(host, port, path, timeout):
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f'{path}: HTTP {response.status}')
        return json.loads(response.read())
    finally:
        connection.close()


def inspect_server(spec, host, port):
    props = request_json(host, port, '/props', spec['timeout_seconds'])
    slots = request_json(host, port, '/slots', spec['timeout_seconds'])
    if not isinstance(props, dict) or type(props.get('total_slots')) is not int or props['total_slots'] != spec['server_cap'] or props.get('build_info') != spec['server_build'] or props.get('model_path') != spec['server_model_path']:
        raise ValueError('server identity or slot capacity differs from manifest')
    if not isinstance(slots, list) or len(slots) != spec['server_cap']:
        raise ValueError('server did not expose the declared slots')
    seen = set()
    for slot in slots:
        if not isinstance(slot, dict) or type(slot.get('id')) is not int or slot['id'] < 0 or slot['id'] in seen or slot.get('n_ctx') != spec['context_per_slot'] or slot.get('is_processing') is not False:
            raise ValueError('slot IDs, context or idle state differ from manifest')
        seen.add(slot['id'])
    return sorted(seen), props


def one_request(spec, host, port, slot, session, gate, released):
    payload = dict(prompt=spec['prompt_ids'], n_predict=spec['tokens_per_request'], stream=True,
                   cache_prompt=False, ignore_eos=True, return_tokens=True, return_progress=True, temperature=0,
                   seed=spec['seed'] + session, id_slot=slot)
    conn = http.client.HTTPConnection(host, port, timeout=spec['timeout_seconds'])
    try:
        gate.wait(timeout=spec['timeout_seconds'])
        start = time.monotonic()
        conn.request('POST', '/completion', body=json.dumps(payload), headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        response = conn.getresponse()
        if response.status != 200 or 'text/event-stream' not in response.getheader('Content-Type', '').lower():
            raise ValueError('server rejected streaming completion')
        pieces, ids, final = [], [], None
        first, prompt_progress = None, None
        for line in response:
            if time.monotonic() - start > spec['timeout_seconds']:
                raise ValueError('completion exceeded total timeout')
            line = line.decode('utf-8').strip()
            if not line or line.startswith(':'):
                continue
            if not line.startswith('data:'):
                raise ValueError('invalid SSE frame')
            data = line[5:].strip()
            if data == '[DONE]':
                break
            event = json.loads(data)
            if not isinstance(event, dict) or event.get('error') or type(event.get('stop')) is not bool:
                raise ValueError('invalid completion event')
            if 'prompt_progress' in event:
                progress = event['prompt_progress']
                if not isinstance(progress, dict) or type(progress.get('cache')) is not int or progress['cache'] != 0 or progress.get('total') != len(spec['prompt_ids']):
                    raise ValueError('prompt progress is missing or contains reused tokens')
                prompt_progress = progress
                # The pinned server emits a placeholder token [0] on progress-only frames.
                # It is not generated output; a real token 0 is still counted normally.
                if event.get('tokens_predicted') == 0:
                    continue
            text = event.get('content')
            tokens = event.get('tokens')
            if not isinstance(text, str) or not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError('invalid completion payload')
            if text and first is None:
                first = time.monotonic() - start
            pieces.append(text)
            ids.extend(tokens)
            if event['stop']:
                final = event
                break
        end = time.monotonic()
        if final is None or first is None or prompt_progress is None or final.get('stop_type') != 'limit' or final.get('truncated') is not False:
            raise ValueError('completion truncated, cached, empty, or stopped unexpectedly')
        if final.get('id_slot') != slot or final.get('tokens_predicted') != spec['tokens_per_request'] or final.get('tokens_evaluated') != len(spec['prompt_ids']) or len(ids) != spec['tokens_per_request']:
            raise ValueError('completion slot or token count differs from request')
        timings = final.get('timings')
        if not isinstance(timings, dict) or any(type(timings.get(k)) not in (int, float) or not math.isfinite(timings[k]) or timings[k] <= 0 for k in ('prompt_ms', 'predicted_ms')):
            raise ValueError('missing or invalid server timings')
        content = ''.join(pieces)
        return dict(slot=slot, session=session, wall_s=end-start, ttft_s=first, ended_after_release_s=end-released[0],
                    tokens=len(ids), content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    token_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
                    prompt_progress=prompt_progress, final_tokens_cached=final.get('tokens_cached'), timings=timings)
    finally:
        conn.close()


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(p * len(values)) - 1)]


def run_round(spec, host, port, slots, n):
    released = [0.0]
    gate = threading.Barrier(n, action=lambda: released.__setitem__(0, time.monotonic()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(one_request, spec, host, port, slots[i], i, gate, released) for i in range(n)]
        rows = [f.result() for f in futures]
    duration = max(r['ended_after_release_s'] for r in rows)
    if duration <= 0:
        raise ValueError('invalid round duration')
    return dict(concurrency=n, round_wall_s=duration, aggregate_tokens_s=sum(r['tokens'] for r in rows)/duration,
                ttft_p50_s=statistics.median(r['ttft_s'] for r in rows), ttft_p95_s=percentile([r['ttft_s'] for r in rows], .95),
                latency_p50_s=statistics.median(r['wall_s'] for r in rows), latency_p95_s=percentile([r['wall_s'] for r in rows], .95), requests=rows)


def run(spec, out):
    host, port = validate(spec)
    out = Path(out)
    partial = out.with_suffix('.partial.json')
    if out.exists() or partial.exists() or partial == out:
        raise ValueError('choose a new output file')
    calibrator = Path(spec['calibrator']).resolve(strict=True)
    calibration_identity = sha256_file(calibrator)
    source_identity = sha256_file(Path(__file__))
    with PerfLock(ROOT, 'concurrency throughput comparison', timeout=0):
        slots, props = inspect_server(spec, host, port)
        identity = server_identity(spec)
        power = power_state()
        if power.untrusted or power.source == 'fake':
            raise ValueError('trusted real power state required')
        evidence = dict(schema='qwen-concurrency-1', manifest=spec, server_props=props, samples=[],
                        identity=identity, host=platform.node(), platform=platform.platform(), power=power.as_dict(), power_source=power.source,
                        meaning='Aggregate generated tokens / round wall time including prompt processing; descriptive only.',
                        scope='One externally started server; PID/worker affinity and executable/model hashes checked before and after.',
                        source_sha256=source_identity, calibrator_sha256=calibration_identity,
                        memory_method='RSS/PSS snapshots; VmHWM is the process-lifetime high-water mark, not reset per sample.')
        def save_partial():
            atomic_write_text(partial, json.dumps(evidence, indent=2, allow_nan=False) + '\n')
        save_partial()
        with tempfile.TemporaryDirectory() as scratch:
            def calibration():
                return steady_calibration(lambda: measure(str(calibrator), ['--calibration'], scratch)['__calibration'])
            def sample_round(n, r):
                for attempt in range(3):
                    quiet = wait_for_quiet(.05, 10)
                    if not quiet[2]:
                        raise ValueError('host not quiet before round')
                    available, _ = inspect_server(spec, host, port)
                    if available != slots:
                        raise ValueError('server slots changed')
                    current = process_stat(spec['server_pid'])
                    if current['start_ticks'] != identity['start_ticks'] or sorted(os.sched_getaffinity(spec['server_pid'])) != identity['cpus'] or power_state().as_dict() != power.as_dict():
                        raise ValueError('server process, affinity or power state changed')
                    check_thread_affinities(spec['server_pid'], spec['cpus'])
                    memory_before = memory_snapshot(spec['server_pid'])
                    before = calibration()
                    cpu_global_before = cpu_snapshot()
                    cpu_selected_before = cpu_snapshot(spec['cpus'])
                    server_before = process_stat(spec['server_pid'])
                    sample = run_round(spec, host, port, slots, n)
                    server_after = process_stat(spec['server_pid'])
                    cpu_selected_after = cpu_snapshot(spec['cpus'])
                    cpu_global_after = cpu_snapshot()
                    after = calibration()
                    current_power = power_state()
                    if server_before['start_ticks'] != identity['start_ticks'] or server_after['start_ticks'] != identity['start_ticks'] or sorted(os.sched_getaffinity(spec['server_pid'])) != identity['cpus'] or current_power.untrusted or current_power.source == 'fake' or current_power.as_dict() != power.as_dict():
                        raise ValueError('server process, affinity or power changed during round')
                    check_thread_affinities(spec['server_pid'], spec['cpus'])
                    memory_after = memory_snapshot(spec['server_pid'])
                    server_ticks = server_after['cpu_ticks'] - server_before['cpu_ticks']
                    background = background_fraction(cpu_global_before, cpu_global_after, server_ticks)
                    selected_background = background_fraction(cpu_selected_before, cpu_selected_after, server_ticks)
                    valid_cal = math.isfinite(before) and math.isfinite(after) and before > 0 and after > 0 and abs(after / before - 1) <= .05
                    tainted = background > .05 or selected_background > .05 or not valid_cal
                    sample.update(round=r, attempt=attempt, warmup=r == -1, quiet=quiet,
                                  calibration_ns=[v if math.isfinite(v) else None for v in (before, after)],
                                  background_global=background, background_selected=selected_background,
                                  server_cpu_ticks=server_ticks, tainted=tainted,
                                  memory_before=memory_before, memory_after=memory_after)
                    evidence['samples'].append(sample)
                    save_partial()
                    if not tainted:
                        return
                raise ValueError('three tainted attempts; partial evidence preserved')
            for n in spec['concurrency']:
                sample_round(n, -1)
            for r in range(spec['rounds']):
                order = spec['concurrency'][r % len(spec['concurrency']):] + spec['concurrency'][:r % len(spec['concurrency'])]
                for n in order:
                    sample_round(n, r)
        if sha256_file(calibrator) != calibration_identity or sha256_file(Path(__file__)) != source_identity:
            raise ValueError('benchmark runner or calibrator changed during the run')
        final_slots, final_props = inspect_server(spec, host, port)
        if final_slots != slots or final_props.get('build_info') != props.get('build_info') or final_props.get('model_path') != props.get('model_path'):
            raise ValueError('server identity or slots changed during the run')
        final_power = power_state()
        if server_identity(spec) != identity or final_power.untrusted or final_power.source == 'fake' or final_power.as_dict() != power.as_dict():
            raise ValueError('server identity or power changed during the run')
        evidence['summary'] = {str(n): {'aggregate_tokens_s_median': statistics.median(s['aggregate_tokens_s'] for s in evidence['samples'] if s['concurrency'] == n and not s['warmup'] and not s['tainted']),
                                         'ttft_p50_s': statistics.median(v['ttft_s'] for s in evidence['samples'] if s['concurrency'] == n and not s['warmup'] and not s['tainted'] for v in s['requests']),
                                         'ttft_p95_s': percentile([v['ttft_s'] for s in evidence['samples'] if s['concurrency'] == n and not s['warmup'] and not s['tainted'] for v in s['requests']], .95),
                                         'latency_p50_s': statistics.median(v['wall_s'] for s in evidence['samples'] if s['concurrency'] == n and not s['warmup'] and not s['tainted'] for v in s['requests']),
                                         'latency_p95_s': percentile([v['wall_s'] for s in evidence['samples'] if s['concurrency'] == n and not s['warmup'] and not s['tainted'] for v in s['requests']], .95)} for n in spec['concurrency']}
        atomic_write_text(out, json.dumps(evidence, indent=2, allow_nan=False) + '\n', overwrite=False)
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run(json.loads(Path(args.manifest).read_text()), args.out)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
