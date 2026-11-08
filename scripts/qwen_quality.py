#!/usr/bin/env python3
"""Resumable, artifact-bound 128-position quality capture and strict paired-NLL judgement."""
import array
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/perf'))
sys.path.insert(0,str(ROOT/'tools/reference'))
from sub0perf.common import atomic_write_text
from sub0perf.lock import PerfLock
from sub0perf.workload import digest,execute
from compare_sequence import compare_sequence


def load(path):return json.loads(Path(path).read_text())
def save(path,value):atomic_write_text(Path(path),json.dumps(value,indent=2,allow_nan=False)+'\n')
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_scores(scores,window):
    n=len(window['input_ids'])
    if scores.get('schema')!='qwen-nll-1' or scores.get('vocabulary')!=248320:raise ValueError('score schema/vocabulary mismatch')
    for field in ('input_ids','target_ids'):
        if any(type(t)is not int for t in scores.get(field,[])):raise ValueError('invalid token type')
        if scores.get(field)!=window[field]:raise ValueError('wrong '+field)
    if len(scores.get('nll',[]))!=n or len(scores.get('top_ids',[]))!=n:raise ValueError('incomplete score sequence')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in scores['nll']):raise ValueError('invalid NLL')
    if any(type(v) is not int or not 0<=v<248320 for v in scores['top_ids']):raise ValueError('invalid argmax')
    return scores


def validate_fixture(fixture):
    if fixture.get('schema')!='qwen-quality-fixture-1' or fixture.get('context')!=128 or fixture.get('tokens')!=4096:raise ValueError('wrong fixture contract')
    windows=fixture['windows'];domains={w['domain']for w in windows}
    if len(windows)!=32 or len({w['id']for w in windows})!=32 or domains!={'literature','science','civics','software'}:raise ValueError('wrong/duplicate windows')
    for domain in domains:
        rows=[w for w in windows if w['domain']==domain]
        if {w['id']for w in rows}!={f'{domain}-{n}'for n in range(8)}:raise ValueError('wrong domain window IDs')
        if len(rows)!=8 or sum(bool(w['capture_logits'])for w in rows)!=1 or sum(bool(w['check_original_baseline'])for w in rows)!=1:raise ValueError('incomplete domain')
    for w in windows:
        if type(w['capture_logits']) is not bool or w['capture_logits']!=w['check_original_baseline']:raise ValueError('invalid capture/baseline contract')
        if len(w['input_ids'])!=128 or len(w['target_ids'])!=128:raise ValueError('wrong context length')
        if any(type(t) is not int or not 0<=t<248320 for t in w['input_ids']+w['target_ids']):raise ValueError('bad fixture IDs')
    return fixture


def interval(values,critical=1.96):
    if len(values)<2 or any(not math.isfinite(v)for v in values):raise ValueError('insufficient/invalid paired data')
    mean=statistics.fmean(values);se=statistics.stdev(values)/math.sqrt(len(values))
    return dict(mean=mean,se=se,ci95=[mean-critical*se,mean+critical*se],count=len(values))


def quality_verdict(fixture,pairs):
    validate_fixture(fixture)
    if set(pairs)!={w['id']for w in fixture['windows']}:raise ValueError('missing/extra quality windows')
    all_diff=[];window_means=[];domains={};domain_means=[]
    for w in fixture['windows']:
        reference,candidate=pairs[w['id']]
        validate_scores(reference,w);validate_scores(candidate,w)
        delta=[b-a for a,b in zip(reference['nll'],candidate['nll'])]
        all_diff.extend(delta);window_means.append(statistics.fmean(delta));domains.setdefault(w['domain'],[]).append(delta)
    checks={'global_tokens':interval(all_diff),'global_windows':interval(window_means,2.039513)}
    for domain,groups in sorted(domains.items()):
        means=list(map(statistics.fmean,groups));domain_means.append(statistics.fmean(means))
        checks[domain+'_tokens']=interval([v for group in groups for v in group])
        checks[domain+'_windows']=interval(means,2.364624)
    checks['global_domains']=interval(domain_means,3.182446)
    verdict='pass'if all(v['ci95'][1]<=.03 for v in checks.values())else 'fail'if any(v['ci95'][0]>.03 for v in checks.values())else 'inconclusive'
    summary={}
    for name,index in (('reference',0),('candidate',1)):
        values=[v for w in fixture['windows']for v in pairs[w['id']][index]['nll']]
        mean=statistics.fmean(values);summary[name]=dict(mean_nll=mean,perplexity=math.exp(mean)if mean<700 else None)
    agreement=sum(a==b for w in fixture['windows']for a,b in zip(pairs[w['id']][0]['top_ids'],pairs[w['id']][1]['top_ids']))/len(all_diff)
    return dict(verdict=verdict,threshold_nats=.03,tokens=len(all_diff),checks=checks,arms=summary,top1_agreement=agreement)


def prepare(spec):
    if digest(spec['fixture'])!=spec['fixture_sha256']:raise ValueError('frozen fixture changed')
    fixture=validate_fixture(load(spec['fixture']))
    for domain,value in fixture['source_sha256'].items():
        if digest(Path(spec['fixture']).parent/(domain+'.txt'))!=value:raise ValueError('source text changed')
    if digest(spec['protocol'])!=spec['protocol_sha256']:raise ValueError('frozen protocol changed')
    model_hash=digest(spec['model'])
    if model_hash!=spec['model_sha256']:raise ValueError('wrong model hash')
    return fixture,model_hash


def expected_record(spec,provider,window,model_hash):
    arm=spec['providers'][provider]
    identity={name:digest(path)for name,path in arm['identity_files'].items()}
    capture=window['capture_logits']
    key=dict(schema='qwen-quality-cache-1',model_sha256=model_hash,provider=provider,
             identities=identity,command=arm['command'],capture_command=arm.get('capture_command',[]),
             window=window,capture_logits=capture,cpus=spec['cpus'],log_marker=arm.get('log_marker'))
    folder=Path(spec['cache'])/provider/signature(key)/window['id']
    return key,folder


def read_complete(folder,key,window):
    marker=folder/'complete.json'
    if not marker.exists():return None
    record=load(marker)
    required={'scores.json','run.log'}
    if window['capture_logits']:
        required.update('logits'+(f'.pos{p}'if p else '')+'.logits'for p in range(128))
    if set(record['files'])!=required:raise ValueError('incomplete cache file manifest')
    for name in required-{'scores.json','run.log'}:
        if (folder/name).stat().st_size!=248320*4:raise ValueError('truncated cached logits')
    if record['identity']!=key:raise ValueError('cache identity mismatch')
    for name,value in record['files'].items():
        if digest(folder/name)!=value:raise ValueError('cached output changed')
    validate_scores(load(folder/'scores.json'),window)
    if key.get('log_marker')and key['log_marker']not in (folder/'run.log').read_text():raise ValueError('cached configuration marker mismatch')
    peak=record['process']['peak_rss_mib']
    if type(peak)not in (int,float)or not math.isfinite(peak)or not 0<=peak<=8192:raise ValueError('cached memory budget exceeded/invalid')
    return record


def capture(spec_path,max_jobs=None):
    if max_jobs is not None and max_jobs<1:raise ValueError('max-jobs must be positive')
    spec=load(spec_path);fixture,model_hash=prepare(spec);jobs=0
    cpus=spec['cpus'];allowed=os.sched_getaffinity(0)
    if len(cpus)!=8 or len(set(cpus))!=8 or not set(cpus)<=allowed:raise ValueError('eight allowed CPUs required')
    env=dict(os.environ)
    if env.get('LD_LIBRARY_PATH') or env.get('LD_PRELOAD'):raise ValueError('remove dynamic-loader overrides')
    for k in ('KMP_AFFINITY','KMP_HW_SUBSET','KMP_ALL_THREADS','GOMP_CPU_AFFINITY','OMP_PLACES'):env.pop(k,None)
    env.update(OMP_NUM_THREADS='8',OMP_PROC_BIND='false',OMP_DYNAMIC='false',OMP_THREAD_LIMIT='8',OMP_MAX_ACTIVE_LEVELS='1')
    with PerfLock(ROOT,'dense quality qualification',timeout=0):
        try:
            os.sched_setaffinity(0,set(cpus))
            for window in fixture['windows']:
                for provider in ('reference','candidate','baseline'):
                    if provider=='baseline'and not window['check_original_baseline']:continue
                    key,folder=expected_record(spec,provider,window,model_hash)
                    if read_complete(folder,key,window) is not None:continue
                    if max_jobs is not None and jobs>=max_jobs:
                        save(Path(spec['cache'])/'state.json',dict(status='partial',completed_jobs=jobs));return
                    folder.mkdir(parents=True,exist_ok=True)
                    arm=spec['providers'][provider];mapping={'{model}':spec['model'],'{inputs}':','.join(map(str,window['input_ids'])),
                        '{targets}':','.join(map(str,window['target_ids'])),'{scores}':str(folder/'scores.json'),
                        '{dump}':str(folder/'logits')if window['capture_logits']else '-'}
                    argv=arm['command']+(arm.get('capture_command',[])if window['capture_logits']else [])
                    command=[mapping.get(v,v)for v in argv]
                    log=folder/'run.log';run_env=dict(env,QUALITY_LOG=str(log))
                    launcher=['/bin/sh','-c','exec "$@" > "$QUALITY_LOG" 2>&1','quality-exec',*command]
                    save(Path(spec['cache'])/'state.json',dict(status='running',driver_pid=os.getpid(),provider=provider,window=window['id'],folder=str(folder)))
                    print(json.dumps(dict(status='starting',provider=provider,window=window['id'],folder=str(folder))),flush=True)
                    process=execute(launcher,ROOT,run_env,3600)
                    process['launcher_output_sha256']=process.pop('output_sha256')
                    validate_scores(load(folder/'scores.json'),window)
                    expected_log=arm.get('log_marker')
                    if expected_log and expected_log not in log.read_text():raise ValueError('scorer configuration marker mismatch')
                    if process['peak_rss_mib']>8192:raise ValueError('memory budget exceeded')
                    if digest(spec['model'])!=model_hash or expected_record(spec,provider,window,model_hash)[0]!=key:raise ValueError('artifact changed during capture')
                    files={'scores.json':digest(folder/'scores.json'),'run.log':digest(log)}
                    if window['capture_logits']:
                        for position in range(128):
                            name='logits'+(f'.pos{position}'if position else '')+'.logits'
                            if (folder/name).stat().st_size!=248320*4:raise ValueError('incomplete logit vector')
                            files[name]=digest(folder/name)
                    record=dict(identity=key,command=command,process=process,files=files)
                    save(folder/'complete.json',record);jobs+=1
                    print(json.dumps(dict(status='complete',provider=provider,window=window['id'],**process)),flush=True)
            save(Path(spec['cache'])/'state.json',dict(status='captured',driver_pid=os.getpid()))
        except BaseException as error:
            save(Path(spec['cache'])/'state.json',dict(status='interrupted'if isinstance(error,KeyboardInterrupt)else 'failed',error=str(error)))
            raise
        finally:os.sched_setaffinity(0,allowed)


def judge(spec_path,out):
    spec=load(spec_path);fixture,model_hash=prepare(spec);pairs={};gates=[];records=[];nll_spot_checks=[]
    for window in fixture['windows']:
        scores={}
        for provider in ('reference','candidate','baseline'):
            if provider=='baseline'and not window['check_original_baseline']:continue
            key,folder=expected_record(spec,provider,window,model_hash);record=read_complete(folder,key,window)
            if record is None:raise ValueError('qualification incomplete: '+provider+'/'+window['id'])
            scores[provider]=load(folder/'scores.json')
            if window['capture_logits']:
                for position in (0,1,31,63,127):
                    values=array.array('f');values.frombytes((folder/('logits'+(f'.pos{position}'if position else '')+'.logits')).read_bytes())
                    if sys.byteorder!='little':values.byteswap()
                    maximum=max(values);loss=math.log(math.fsum(math.exp(float(v)-maximum)for v in values))+maximum-values[window['target_ids'][position]]
                    error=abs(loss-scores[provider]['nll'][position])
                    if error>1e-10:raise ValueError('independent NLL recomputation failed')
                    nll_spot_checks.append(dict(provider=provider,window=window['id'],position=position,absolute_error=error))
            records.append(dict(provider=provider,window=window['id'],folder=str(folder),process=record['process'],files=record['files'],scores=scores[provider]))
        pairs[window['id']]=(scores['reference'],scores['candidate'])
        if window['capture_logits']:
            _,ref=expected_record(spec,'reference',window,model_hash)
            for provider in ('candidate','baseline'):
                _,actual=expected_record(spec,provider,window,model_hash)
                result=compare_sequence(str(actual/'logits'),str(ref/'logits'),128,.001,248320)
                gates.append(dict(provider=provider,window=window['id'],**result))
    result=dict(schema='qwen-quality-result-1',manifest=spec,fixture_sha256=digest(spec['fixture']),quality=quality_verdict(fixture,pairs),numerical=gates,processes=records,nll_spot_checks=nll_spot_checks)
    result['passed']=result['quality']['verdict']=='pass'and all(r['passed']for r in gates)
    save(out,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('capture','judge'));p.add_argument('manifest');p.add_argument('--out');p.add_argument('--max-jobs',type=int);a=p.parse_args()
    try:
        if a.action=='capture':capture(a.manifest,a.max_jobs);return 0
        if not a.out:p.error('judge requires --out')
        return 0 if judge(a.manifest,a.out)['passed']else 2
    except (ValueError,OSError,RuntimeError,KeyError)as error:
        print(json.dumps(dict(status='refused',error=str(error))),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
