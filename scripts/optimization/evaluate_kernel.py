#!/usr/bin/env python3
"""Trusted kernel-only evaluator: frozen fixtures, staged builds, paired sub0perf measurements."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/perf'))
from sub0perf.lock import PerfLock
from sub0perf.load import wait_for_quiet
from sub0perf.platform_proc import set_pin_cpu
from sub0perf.power import power_state
from sub0perf.runner import measure,steady_calibration
from sub0perf.workload import digest,execute


def evaluate():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','result','compiler','benchmark','editable'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--rounds',type=int,default=5);p.add_argument('--min-gain',type=float,default=.02)
    p.add_argument('--max-rss-mib',type=float,default=256);p.add_argument('--cpu',type=int,default=0);a=p.parse_args()
    if a.rounds<3 or not 0<a.min_gain<1:raise ValueError('invalid predeclared rounds/gain')
    baseline=Path(a.baseline).resolve();candidate=Path(a.candidate).resolve();out=Path(a.result).resolve();folder=out.parent
    if not (candidate/a.editable).resolve().is_relative_to(candidate):raise ValueError('editable outside candidate')
    def interrupted(sig,frame):raise InterruptedError('evaluation interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    result=dict(schema='optimization-evaluation-1',qualification_scope='kernel-only',
        candidate_files={a.editable:digest(candidate/a.editable)},
        baseline_commit=subprocess.check_output(['git','-C',str(baseline),'rev-parse','HEAD'],text=True).strip(),
        correctness='fail',memory='not_measured',performance='inconclusive',evidence=[])
    records=[];binaries={};checks={};build_commands=[];env=dict(os.environ)
    if env.get('LD_LIBRARY_PATH')or env.get('LD_PRELOAD'):raise ValueError('remove loader overrides')
    try:
        # Compile the BASELINE's immutable benchmark/independent fixture sources against each header set.
        # Candidate scope cannot edit these fixtures or the evaluator itself.
        for label,tree,native in (('baseline',baseline,True),('candidate',candidate,True),('candidate-generic',candidate,False)):
            binary=folder/(label+'-bench');binaries[label]=str(binary)
            cmd=[a.compiler,'-std=c++23','-O3',*(['-march=native']if native else []),'-Wall','-Wextra','-Werror',
                 '-I'+str(tree/'include'),'-I'+str(baseline/'tools/perf'),'-isystem',str(baseline/'tests/vendor'),
                 str(baseline/'benchmarks/dense_quant_perf.cpp'),'-o',str(binary)]
            build_commands.append(cmd)
            log=folder/(label+'-build.log')
            result['evidence'].append(str(log))
            with log.open('wb')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
            checks[label]=execute([str(binary),'--list'],baseline,env,60)
        result['correctness']='pass'
        result['memory']='pass'if all(x['peak_rss_mib']<=a.max_rss_mib for x in checks.values())else 'fail'
        if result['memory']!='pass':raise ValueError('kernel RSS budget exceeded')
        identities={k:digest(v)for k,v in binaries.items()};identities['compiler']=digest(a.compiler)
        initial=power_state()
        governor_path=Path(f'/sys/devices/system/cpu/cpu{a.cpu}/cpufreq/scaling_governor')
        governor=governor_path.read_text().strip()if governor_path.exists()else 'unavailable'
        if initial.untrusted or initial.source=='fake' or initial.state!='ac' or initial.saver:raise ValueError('trusted mains power without saver required')
        set_pin_cpu(a.cpu)
        try:
            for r in range(a.rounds):
                quiet=wait_for_quiet(.05,10)
                # wait_for_quiet's return contract includes quietness; loaded rounds cannot pass.
                if not quiet[2]:raise ValueError('background load did not settle')
                order=['baseline','candidate'];random.Random(42+r).shuffle(order);pair={}
                with tempfile.TemporaryDirectory()as temporary:
                    for label in order:
                        cal=steady_calibration(lambda:measure(binaries[label],['--calibration'],temporary)['__calibration'])
                        value=measure(binaries[label],['--exact',a.benchmark],temporary)[a.benchmark]
                        end=measure(binaries[label],['--calibration'],temporary)['__calibration']
                        pair[label]=dict(ns=value,calibration_start_ns=cal,calibration_end_ns=end)
                if initial.as_dict()!=power_state().as_dict()or (governor_path.exists()and governor_path.read_text().strip()!=governor):raise ValueError('power/governor state changed')
                if any(abs(x['calibration_end_ns']/x['calibration_start_ns']-1)>.05 for x in pair.values()):raise ValueError('clock drift; inconclusive')
                pair['ratio']=(pair['candidate']['ns']/pair['candidate']['calibration_start_ns'])/(pair['baseline']['ns']/pair['baseline']['calibration_start_ns'])
                pair['round']=r;pair['quiet']=quiet;records.append(pair)
        finally:set_pin_cpu(None)
        if identities!={**{k:digest(v)for k,v in binaries.items()},'compiler':digest(a.compiler)}:raise ValueError('staged binary changed')
        ratios=[r['ratio']for r in records];median=statistics.median(ratios)
        noise=1.4826*statistics.median(abs(x-median)for x in ratios)/median
        limit=max(a.min_gain,3*noise)
        result['performance']='inconclusive'if noise>.02 else 'improved'if median<1-limit else 'regressed'if median>1+limit else 'unchanged'
        result.update(median_candidate_over_baseline=median,robust_cv=noise,decision_limit=limit,identities=identities)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError)as e:
        result['error']=str(e)
        result['diagnostics']='\n'.join(Path(p).read_text(errors='replace')[-2000:]for p in result['evidence']if p.endswith('-build.log'))
    evidence=folder/'kernel-evidence.json';evidence.write_text(json.dumps(dict(rounds=records,checks=checks,build_commands=build_commands,cpu=a.cpu,host=platform.node(),min_gain=a.min_gain,max_rss_mib=a.max_rss_mib,thread_environment={k:v for k,v in env.items()if k.startswith('OMP_')},governor=locals().get('governor','not measured'),power=locals().get('initial').as_dict()if 'initial'in locals()else None),indent=2)+'\n')
    result['evidence'].append(str(evidence));out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0
def main():
    with PerfLock(ROOT,'focused kernel build/evaluation',timeout=0):
        return evaluate()

if __name__=='__main__':raise SystemExit(main())
