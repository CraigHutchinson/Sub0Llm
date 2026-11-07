#!/usr/bin/env python3
"""Interleaved native-phase comparison. Descriptive rates, never an automatic regression acceptance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/perf'))
from sub0perf.common import atomic_write_text
from sub0perf.gitstate import git_head, tree_clean
from sub0perf.load import system_load
from sub0perf.lock import PerfLock
from sub0perf.platform_proc import cpu_times, set_pin_cpu
from sub0perf.power import power_state
from sub0perf.runner import measure, steady_calibration
from sub0perf.workload import digest, execute


def phase_rates(data, prompt, decode):
    if data.get('schema')!='qwen-phase-1' or type(data.get('prompt_tokens')) is not int or data['prompt_tokens']!=prompt:
        raise ValueError('wrong phase schema/prompt denominator')
    if type(data.get('decode_evaluations')) is not int or data['decode_evaluations']!=decode:
        raise ValueError('wrong decode denominator')
    if type(data.get('batched_prefill')) is not bool or type(data.get('f32_attention')) is not bool:
        raise ValueError('missing phase method')
    expected_prefill=1 if data['batched_prefill'] else prompt
    for name,count in (('prefill_ms',expected_prefill),('decode_ms',decode)):
        values=data.get(name)
        if not isinstance(values,list) or len(values)!=count:
            raise ValueError('incomplete '+name)
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('invalid '+name)
    ids=data.get('top_ids')
    if not isinstance(ids,list) or len(ids)!=expected_prefill+decode or any(type(v) is not int or not 0<=v<248320 for v in ids):
        raise ValueError('incomplete/invalid predicted IDs')
    if type(data.get('setup_ms')) not in (int,float) or not math.isfinite(data['setup_ms']) or data['setup_ms']<0:
        raise ValueError('invalid setup duration')
    prefill=math.fsum(data['prefill_ms']);generation=math.fsum(data['decode_ms'])
    return {'prefill_tok_s':1000*prompt/prefill,'decode_tok_s':1000*decode/generation,
            'prefill_s':prefill/1000,'decode_s':generation/1000,'setup_s':data['setup_ms']/1000,
            'setup_plus_prefill_s':(data['setup_ms']+prefill)/1000,
            'generated_ids':ids[expected_prefill-1:]}


def quiet_window():
    if cpu_times() is None:raise RuntimeError('CPU load counters unavailable')
    for attempt in range(3):
        loads=[system_load(1.0) for _ in range(10)]
        if max(loads)<=.05:return {'samples':loads,'seconds':10}
        print('Waiting: background load exceeded 5% in the sustained window',flush=True)
    raise RuntimeError('host did not become quiet; no comparison verdict')


def cpu_snapshot(cpus=None):
    values=[]
    for line in Path('/proc/stat').read_text().splitlines():
        label,*numbers=line.split()
        wanted=label=='cpu' if cpus is None else label in {f'cpu{c}'for c in cpus}
        if wanted:
            numbers=list(map(int,numbers[:8]));values.append((sum(numbers)-numbers[3]-numbers[4],sum(numbers)))
    return tuple(sum(v[k]for v in values)for k in (0,1))


def background_fraction(before,after,child_cpu_s):
    busy=after[0]-before[0];total=after[1]-before[1]
    if total<=0:raise RuntimeError('invalid CPU accounting interval')
    return max(0.,(busy-child_cpu_s*os.sysconf('SC_CLK_TCK'))/total)


def summary(records):
    result={}
    for name in sorted({r['arm']for r in records}):
        rows=[r for r in records if r['arm']==name and not r['warmup'] and not r['tainted']]
        if not rows:continue
        result[name]={}
        for metric in ('prefill_tok_s','decode_tok_s','setup_s','setup_plus_prefill_s','process_wall_s','peak_rss_mib'):
            vals=[r['rates'][metric]for r in rows];median=statistics.median(vals)
            mad=statistics.median(abs(v-median)for v in vals)
            result[name][metric]={'samples':vals,'median':median,'min':min(vals),'max':max(vals),
                                  'robust_cv':1.4826*mad/median if median else 0}
    return result


def run(manifest,out,rounds=3):
    manifest=Path(manifest).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('choose a new evidence output path')
    if rounds<3:raise ValueError('at least three measured rounds required')
    spec=json.loads(manifest.read_text());cpus=spec['cpus'];prompt=len(spec['prompt_ids']);decode=spec['decode_evaluations']
    allowed=os.sched_getaffinity(0)
    if len(cpus)!=8 or len(set(cpus))!=8 or not set(cpus)<=allowed:raise ValueError('eight allowed distinct CPUs required')
    cores={(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text().strip(),
            Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text().strip())for c in cpus}
    if len(cores)!=8:raise ValueError('select eight physical cores, not SMT siblings')
    if not tree_clean(ROOT):raise ValueError('commit the staged experiment before timing')
    arms=spec['arms'];records=[]
    if not arms or any(arm['command'].count('{metrics}')!=1 or arm['command'].count('{tokens}')!=1 for arm in arms):
        raise ValueError('each arm requires exactly one metrics and token placeholder')
    if os.environ.get('LD_PRELOAD') or os.environ.get('LD_LIBRARY_PATH'):
        raise ValueError('remove ambient dynamic-loader overrides before comparison')
    partial=out.with_suffix('.partial.json')
    if partial.exists():raise ValueError('partial evidence already exists; choose a new path')
    original_env=dict(os.environ)
    env=dict(os.environ)
    for key in ('KMP_AFFINITY','KMP_HW_SUBSET','KMP_ALL_THREADS','GOMP_CPU_AFFINITY','OMP_PLACES'):env.pop(key,None)
    env.update({'OMP_NUM_THREADS':'8','OMP_PROC_BIND':'false','OMP_DYNAMIC':'false','OMP_THREAD_LIMIT':'8','OMP_MAX_ACTIVE_LEVELS':'1'})
    # execute() inherits these explicit thread settings; record them and restore the parent afterwards.
    state=None
    with PerfLock(ROOT,'Qwen phase comparison',timeout=0):
        files={**spec['identity_files'],**{('correctness_'+name):path for name,path in spec['correctness_records'].items()},'manifest':str(manifest)}
        identities={name:digest(path)for name,path in files.items()}
        if not all(json.loads(Path(path).read_text()).get('passed') is True for path in spec['correctness_records'].values()):
            raise ValueError('independent numerical gate did not pass')
        state=power_state()
        if state.untrusted or state.source=='fake':raise RuntimeError('trusted real power state required')
        evidence={'schema':'qwen-speed-comparison-1','source_commit':git_head(ROOT),'manifest':spec,
                  'identities':identities,'host':platform.node(),'system':platform.platform(),'power':state.as_dict(),
                  'cpus':cpus,'thread_environment':{k:env[k]for k in ('OMP_NUM_THREADS','OMP_PROC_BIND','OMP_DYNAMIC','OMP_THREAD_LIMIT','OMP_MAX_ACTIVE_LEVELS')},
                  'governors':{str(c):Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor').read_text().strip()for c in cpus},
                  'cache':'warm working weights; no eviction/pressure','samples':records,'timing_verdict':'descriptive comparison only',
                  'correctness':{name:json.loads(Path(path).read_text()) for name,path in spec['correctness_records'].items()}}
        try:
            os.sched_setaffinity(0,set(cpus));os.environ.clear();os.environ.update(env)
            def calibration():
                set_pin_cpu(cpus[0])
                try:
                    with tempfile.TemporaryDirectory()as scratch:
                        return steady_calibration(lambda:measure(spec['calibrator'],['--calibration'],scratch)['__calibration'])
                finally:set_pin_cpu(None)
            def one(arm,round_index,warmup):
                quiet=quiet_window()
                now=power_state()
                if now.untrusted or now.as_dict()!=state.as_dict():raise RuntimeError('power state changed')
                cal_before=calibration()
                with tempfile.TemporaryDirectory(prefix='qwen-speed-')as temporary:
                    metrics=str(Path(temporary)/'phases.json')
                    cmd=[metrics if v=='{metrics}' else ','.join(map(str,spec['prompt_ids'])) if v=='{tokens}' else v for v in arm['command']]
                    before=cpu_snapshot();before_selected=cpu_snapshot(cpus);times=os.times()
                    child=execute(cmd,ROOT,env,1200)
                    end_times=os.times();after=cpu_snapshot();after_selected=cpu_snapshot(cpus)
                    child_cpu=end_times.children_user+end_times.children_system-times.children_user-times.children_system
                    native=json.loads(Path(metrics).read_text())
                rates=phase_rates(native,prompt,decode)
                if native['workers']!=8:raise ValueError('wrong worker count')
                if native['context_capacity']<prompt+decode:raise ValueError('insufficient context')
                rates.update(process_wall_s=child['elapsed_s'],peak_rss_mib=child['peak_rss_mib'])
                foreign=background_fraction(before,after,child_cpu)
                foreign_selected=background_fraction(before_selected,after_selected,child_cpu)
                current=power_state()
                if current.untrusted or current.as_dict()!=state.as_dict():raise RuntimeError('power changed during sample')
                rec={'arm':arm['name'],'round':round_index,'warmup':warmup,'native':native,'rates':rates,
                     'quiet':quiet,'background_global':foreign,'background_selected':foreign_selected,
                     'tainted':foreign>.05 or foreign_selected>.05,'calibration_ns':[cal_before,calibration()],
                     'stdout_sha256':child['output_sha256'],'child_cpu_s':child_cpu,
                     'mean_busy_cores':child_cpu/child['elapsed_s']}
                records.append(rec)
                atomic_write_text(partial,json.dumps(evidence,indent=2,allow_nan=False)+'\n')
                print(json.dumps({'arm':arm['name'],'round':round_index,'warmup':warmup,'tainted':rec['tainted'],**rates}),flush=True)
                return rec
            # All warmups are measured for diagnostics and excluded from the summary.
            for arm in arms:one(arm,-1,True)
            for round_index in range(rounds):
                order=arms[round_index%len(arms):]+arms[:round_index%len(arms)]
                for arm in order:
                    for attempt in range(3):
                        if not one(arm,round_index,False)['tainted']:break
                    else:raise RuntimeError('persistent foreign load; preserve raw partial evidence, no verdict')
            if identities!={name:digest(path)for name,path in files.items()}:raise RuntimeError('binary/input/source identity changed')
            if git_head(ROOT)!=evidence['source_commit']or not tree_clean(ROOT):raise RuntimeError('source changed during experiment')
            evidence['summary']=summary(records)
            atomic_write_text(out,json.dumps(evidence,indent=2,allow_nan=False)+'\n',overwrite=False)
            return evidence
        finally:
            os.sched_setaffinity(0,allowed)
            os.environ.clear();os.environ.update(original_env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('--out',required=True);parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    try:run(args.manifest,args.out,args.rounds)
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print(json.dumps({'status':'refused','error':str(error)}));return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
