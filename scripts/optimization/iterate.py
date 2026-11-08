#!/usr/bin/env python3
"""Focused, bounded propose/evaluate/recover loop; inference is a replaceable JSON subprocess."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/perf'))
from sub0perf.common import atomic_write_text
from sub0perf.lock import PerfLock

PROPOSAL_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'hypothesis': {'type': 'string'}, 'mechanism': {'type': 'string'},
        'edits': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
            'properties': {k: {'type': 'string'} for k in ('path', 'old', 'new')},
            'required': ['path', 'old', 'new']}},
        'no_change_reason': {'type': 'string'}},
    'required': ['hypothesis', 'mechanism', 'edits', 'no_change_reason']}
HINTS = [
    'Inspect serial accumulator dependencies; try independent accumulators and measure register pressure.',
    'Hoist repeated scale/index/broadcast work; check whether the compiler already does so.',
    'Consider decode/dot fusion, unsigned versus signed expansion and avoiding scratch traffic.',
    'Inspect generated assembly/vectorization and cache access before another source-level rewrite.',
    'Consider row blocking, batching and work distribution; preserve scope and mathematical invariants.',
    'Change the mechanism, not just variable names. Identify a concrete falsifiable cost reduction.']


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    atomic_write_text(Path(path), json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def relative(path):
    p = Path(path)
    if p.is_absolute() or not p.parts or any(x in ('.', '..', '.git') for x in p.parts) or '\\' in path:
        raise ValueError('unsafe relative path: ' + path)
    return p


def source(tree, name):
    p = tree / relative(name)
    if not p.resolve().is_relative_to(tree.resolve()) or p.is_symlink() or not p.is_file():
        raise ValueError('source must be a regular file inside checkout: ' + name)
    return p


def validate_item(item):
    if item.get('schema') != 'optimization-item-1' or not re.fullmatch(r'[a-z0-9-]{1,64}', item['id']):
        raise ValueError('invalid item contract/id')
    if not item.get('issue') or not item.get('goal') or not item.get('constraints'):
        raise ValueError('issue, focused goal and invariants required')
    files = item['editable_files']
    if not files or len(files) != len(set(files)):
        raise ValueError('nonempty unique editable scope required')
    for p in files:
        relative(p)
    if item['qualification_scope'] not in ('kernel-only', 'model'):
        raise ValueError('explicit qualification scope required')
    for n in ('max_attempts', 'stagnation_window', 'max_recoveries', 'wall_seconds', 'max_context_chars'):
        if type(item[n]) is not int or item[n] < 1:
            raise ValueError('positive budget required: ' + n)
    if not isinstance(item['provider_command'], list) or not item['provider_command']:
        raise ValueError('JSON inference command required')
    return item


def validate_proposal(p, item):
    if not isinstance(p, dict) or set(p) != set(PROPOSAL_SCHEMA['required']):
        raise ValueError('proposal fields mismatch')
    if any(not isinstance(p[k], str) for k in ('hypothesis', 'mechanism', 'no_change_reason')) or not isinstance(p['edits'], list):
        raise ValueError('proposal types mismatch')
    if len(p['edits']) > 12:
        raise ValueError('too many edits for a focused attempt')
    for e in p['edits']:
        if not isinstance(e, dict) or set(e) != {'path', 'old', 'new'} or any(not isinstance(v, str) for v in e.values()):
            raise ValueError('edit contract mismatch')
        if e['path'] not in item['editable_files'] or not e['old'] or e['old'] == e['new']:
            raise ValueError('out-of-scope or no-op edit')
        if len(e['old']) + len(e['new']) > 32000:
            raise ValueError('edit too large')
    if p['edits'] and (not p['hypothesis'].strip() or not p['mechanism'].strip()):
        raise ValueError('testable hypothesis and mechanism required')
    return p


def parse_reply(reply, item):
    if reply.get('schema') != 'optimization-inference-1' or reply.get('finish_reason') != 'stop':
        raise ValueError('inference failed/truncated; no candidate may be applied')
    return validate_proposal(json.loads(reply['content']), item)


class DuplicateProposal(ValueError):
    pass

def fingerprint(p):
    # Rationale/renaming the hypothesis cannot conceal an identical proposed patch.
    return hashlib.sha256(json.dumps(p['edits'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def apply(tree, proposal, item):
    pending = {}
    for e in proposal['edits']:
        p = source(tree, e['path'])
        text = pending.get(e['path'], p.read_text())
        if text.count(e['old']) != 1:
            raise ValueError('old text must match exactly once: ' + e['path'])
        pending[e['path']] = text.replace(e['old'], e['new'], 1)
    for name, text in pending.items():
        if len(text.encode()) > 262144:
            raise ValueError('candidate file too large')
    for name, text in pending.items():
        source(tree, name).write_text(text)


def command(argv, cwd, log, timeout, stdin=None):
    """No shell; reap the whole command group on interruption or timeout."""
    with Path(log).open('wb') as output:
        p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.communicate(stdin, timeout=timeout)
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait()
        if p.returncode:
            raise RuntimeError(f'command exit {p.returncode}; see {log}')


def render_context(tree, item):
    chunks = []
    for entry in item['context']:
        name = entry['path']; lines = source(tree, name).read_text().splitlines(keepends=True)
        start = entry.get('start_line', 1); stop = entry.get('end_line', len(lines))
        if 'start_marker' in entry:
            text=''.join(lines)
            if text.count(entry['start_marker'])!=1 or text.count(entry['end_marker'])!=1:
                raise ValueError('context markers must be unique')
            start=text[:text.index(entry['start_marker'])].count('\n')+1
            stop=text[:text.index(entry['end_marker'])].count('\n')+1
        if not 1 <= start <= stop <= len(lines):
            raise ValueError('invalid context slice')
        chunks.append({'path': name, 'sha256': sha(source(tree, name)), 'start_line': start,
                       'text': ''.join(lines[start-1:stop])})
    if len(json.dumps(chunks)) > item['max_context_chars']:
        raise ValueError('focused context exceeds declared budget; narrow the work item')
    return chunks


def judge(value, item, candidate, baseline):
    if value.get('schema') != 'optimization-evaluation-1' or value.get('qualification_scope') != item['qualification_scope']:
        raise ValueError('wrong evaluation contract/scope')
    expected = {n: sha(source(candidate, n)) for n in item['editable_files']}
    if value.get('candidate_files') != expected or value.get('baseline_commit') != baseline:
        raise ValueError('evaluation identifies a different candidate/baseline')
    if value.get('correctness') not in ('pass', 'fail') or value.get('memory') not in ('pass', 'fail', 'not_measured'):
        raise ValueError('explicit correctness and memory verdicts required')
    if value.get('performance') not in ('improved', 'unchanged', 'regressed', 'inconclusive'):
        raise ValueError('explicit performance verdict required')
    if not value.get('evidence') or any(not isinstance(p, str) for p in value['evidence']):
        raise ValueError('evaluation evidence required')
    return value['correctness'] == value['memory'] == 'pass' and value['performance'] == 'improved'


def _run(item_path, output, resume=False, proposals_only=False):
    item_path = Path(item_path).resolve(); item = validate_item(load(item_path)); output = Path(output).resolve()
    if git(ROOT, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('commit tracked changes before starting a reproducible experiment')
    output.mkdir(parents=True, exist_ok=True); state_path = output / 'state.json'
    provider = [x.replace('{root}', str(ROOT)) for x in item['provider_command']]
    identity = {str(Path(p).resolve()): sha(p) for p in item['identity_files']}
    contract = dict(item_sha256=sha(item_path), identities=identity, proposals_only=proposals_only)
    if resume:
        state = load(state_path)
        if state['contract'] != contract:
            raise ValueError('resume contract/artifacts changed')
        if state['status'] != 'running':
            raise ValueError('run is terminal; create a new run from its continuation packet')
    else:
        if state_path.exists():
            raise ValueError('output already exists; use --resume or a new path')
        state = dict(schema='optimization-run-1', contract=contract, item=item, base_commit=git(ROOT, 'rev-parse', 'HEAD'),
                     best_commit=git(ROOT, 'rev-parse', 'HEAD'), status='running', attempts=[], seen=[],
                     no_progress=0, recoveries=0, started=time.time(), next_hint=0)
        save(state_path, state)
    while len(state['attempts']) < item['max_attempts'] and time.time()-state['started'] < item['wall_seconds']:
        i = len(state['attempts'])+1; folder = output / f'attempt-{i:03d}'; folder.mkdir()
        tree = folder / 'candidate'; baseline = state['best_commit']
        git(ROOT, 'worktree', 'add', '--detach', str(tree), baseline)
        record = dict(attempt=i, baseline_commit=baseline, status='interrupted', folder=str(folder))
        state['attempts'].append(record); save(state_path, state)
        feedback = [dict(attempt=r['attempt'], status=r['status'], detail=r.get('detail', '')[-900:],
                         hypothesis=r.get('hypothesis', '')[:200]) for r in state['attempts'][-5:-1]]
        hint = (item.get('hints', [])+HINTS)[state['next_hint'] % len(item.get('hints', [])+HINTS)]
        request = dict(schema='optimization-request-1', work_item=item['id'], issue=item['issue'], goal=item['goal'],
                       constraints=item['constraints'], editable_files=item['editable_files'],
                       context=render_context(tree, item), artifact_dir=str(folder), feedback=feedback, hint=hint,
                       attempt=i, seed=item.get('seed', 42)+i, response_schema=PROPOSAL_SCHEMA)
        save(folder/'request.json', request)
        try:
            # Inference provider must terminate its model before it returns. Measurements happen later.
            with PerfLock(ROOT, 'focused optimization inference', timeout=0):
                command(provider, ROOT, folder/'provider.log', max(.1, min(item.get('inference_timeout', 600), item['wall_seconds']-(time.time()-state['started']))),
                        json.dumps(request).encode())
            reply = load(folder/'provider.log'); save(folder/'inference.json', reply)
            if identity != {str(Path(p).resolve()):sha(p) for p in item['identity_files']}:
                raise ValueError('trusted artifacts changed during inference')
            proposal = parse_reply(reply, item); save(folder/'proposal.json', proposal)
            record['hypothesis'] = proposal['hypothesis']
            if not proposal['edits']:
                record.update(status='no_proposal', detail=proposal['no_change_reason'])
            else:
                key = fingerprint(proposal)
                if key in state['seen']:
                    record.update(status='duplicate', detail='same edit set already tried; change mechanism')
                else:
                    state['seen'].append(key)
                    if proposals_only:
                        record.update(status='review_proposal', detail='not applied or measured')
                    else:
                        apply(tree, proposal, item)
                        diff = git(tree, 'diff', '--', *item['editable_files']); (folder/'candidate.diff').write_text(diff+'\n')
                        if not diff:
                            raise ValueError('no resulting diff')
                        tree_key=hashlib.sha256(json.dumps({n:sha(source(tree,n))for n in item['editable_files']},sort_keys=True).encode()).hexdigest()
                        if tree_key in state.setdefault('evaluated_trees',[]):
                            raise DuplicateProposal('identical resulting source tree already tried')
                        state['evaluated_trees'].append(tree_key)
                        evaluation = item.get('evaluation')
                        if not evaluation:
                            raise ValueError('trusted evaluation command required; use --proposals-only')
                        baseline_tree = folder/'baseline'; git(ROOT, 'worktree', 'add', '--detach', str(baseline_tree), baseline)
                        result = folder/'evaluation.json'
                        mapping = {'{root}': str(ROOT), '{candidate}': str(tree), '{baseline}': str(baseline_tree),
                                   '{result}': str(result), '{attempt}': str(folder)}
                        argv = [mapping.get(a, a.replace('{root}', str(ROOT))) for a in evaluation['command']]
                        command(argv, ROOT, folder/'evaluation.log', max(.1,min(evaluation['timeout'],item['wall_seconds']-(time.time()-state['started']))))
                        verdict = load(result)
                        # Pin every evaluator-produced file; model text cannot supply verdicts.
                        evidence = {}
                        for name in verdict.get('evidence', []):
                            p = Path(name).resolve()
                            if not p.is_relative_to(folder) or not p.is_file():
                                raise ValueError('evidence outside attempt or missing')
                            evidence[str(p)] = sha(p)
                        if git(tree,'rev-parse','HEAD')!=baseline or git(baseline_tree,'status','--porcelain','--untracked-files=no'):
                            raise ValueError('candidate HEAD or immutable baseline changed')
                        changed=set(git(tree, 'diff', 'HEAD', '--name-only').splitlines())
                        if not changed <= set(item['editable_files']):
                            raise ValueError('evaluator/candidate changed files outside allowed scope')
                        if identity != {str(Path(p).resolve()):sha(p) for p in item['identity_files']}:
                            raise ValueError('trusted artifacts changed during attempt')
                        accepted = judge(verdict, item, tree, baseline); record['evaluation_sha256']=sha(result); record['evaluation'] = verdict; record['evidence_sha256'] = evidence
                        if accepted:
                            git(tree, 'add', '--', *item['editable_files'])
                            git(tree, 'commit', '-m', f"Experiment {item['id']} attempt {i}: {proposal['hypothesis'][:100]}")
                            state['best_commit'] = git(tree, 'rev-parse', 'HEAD')
                            branch='optimization/'+item['id']+'/'+hashlib.sha256(str(output).encode()).hexdigest()[:12]
                            git(ROOT,'update-ref','refs/heads/'+branch,state['best_commit'])
                            state['review_branch']=branch
                            record.update(status='improved_for_review', candidate_commit=state['best_commit'])
                        else:
                            record.update(status='not_improved', detail=f"correctness={verdict['correctness']} memory={verdict['memory']} performance={verdict['performance']}; {verdict.get('error','')} {verdict.get('diagnostics','')}")
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            record.update(status='duplicate' if isinstance(error,DuplicateProposal) else 'failed', detail=str(error))
        if record['status'] == 'improved_for_review':
            state['no_progress'] = 0
        else:
            state['no_progress'] += 1
        if state['no_progress'] >= item['stagnation_window']:
            state['recoveries'] += 1; state['next_hint'] += 1; state['no_progress'] = 0
            record['recovery_hint'] = (item.get('hints', [])+HINTS)[state['next_hint'] % len(item.get('hints', [])+HINTS)]
        save(folder/'result.json', record); save(state_path, state)
        print(json.dumps(record), flush=True)
        if state['recoveries'] >= item['max_recoveries']:
            state['status'] = 'stalled'; break
    if state['status'] == 'running':
        state['status'] = 'budget_exhausted'
    state['continuation'] = dict(review_branch=state.get('review_branch'),best_commit=state['best_commit'], issue=item['issue'],
        review_required=True, qualification_scope=item['qualification_scope'],
        next_hint=(item.get('hints', [])+HINTS)[state['next_hint'] % len(item.get('hints', [])+HINTS)],
        novel_proposals=len(state['seen']), improved_attempts=sum(r['status']=='improved_for_review'for r in state['attempts']),
        action='Review preserved diffs/evidence; narrow or reprofile the work item. Never infer global optimality from a stalled run.')
    save(state_path, state); return state


def run(item_path, output, resume=False, proposals_only=False):
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.runner.lock').open('a')as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError('another runner owns this output directory')from e
        return _run(item_path,output,resume,proposals_only)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--item', required=True); p.add_argument('--out', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--proposals-only', action='store_true'); a = p.parse_args()
    try:
        s = run(a.item, a.out, a.resume, a.proposals_only); print(json.dumps(dict(status=s['status'], continuation=s['continuation'])))
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        print(json.dumps(dict(status='refused', error=str(e))), file=sys.stderr); return 2
    return 3 if s['status']=='stalled' else 0
if __name__ == '__main__':
    raise SystemExit(main())
