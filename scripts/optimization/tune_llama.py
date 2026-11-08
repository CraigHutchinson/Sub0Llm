#!/usr/bin/env python3
"""Select the lowest projected request latency among explicitly tested llama.cpp profiles."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/perf'))
from sub0perf.lock import PerfLock
from sub0perf.load import wait_for_quiet
from sub0perf.power import power_state
from iterate import command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--configs',nargs='+',required=True)
    p.add_argument('--request',required=True);p.add_argument('--out',required=True)
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--target-new-tokens',type=int,default=512);a=p.parse_args()
    out=Path(a.out).resolve()
    if out.exists() or a.rounds<2 or a.target_new_tokens<1:raise ValueError('new output, >=2 rounds, positive target required')
    out.mkdir(parents=True);request=json.loads(Path(a.request).read_text());records=[];content=None;prompt=None
    initial=power_state()
    if initial.untrusted or initial.source=='fake' or initial.state!='ac' or initial.saver:raise ValueError('trusted mains required')
    with PerfLock(ROOT,'llama.cpp optimization profile tuning',timeout=0):
        for r in range(a.rounds):
            order=a.configs[r%len(a.configs):]+a.configs[:r%len(a.configs)]
            for config in order:
                if not wait_for_quiet(.05,10)[2]:raise ValueError('host not quiet')
                folder=out/f'{r}-{Path(config).stem}';folder.mkdir();request['artifact_dir']=str(folder)
                command([sys.executable,str(ROOT/'scripts/optimization/llama_provider.py'),'--config',str(Path(config).resolve())],
                    ROOT,folder/'reply.json',720,json.dumps(request).encode())
                reply=json.loads((folder/'reply.json').read_text());timing=reply['timings']
                if reply['finish_reason']!='stop':raise ValueError('truncated tuning output')
                if content is None:content=reply['content'];prompt=timing['prompt_n']
                if content!=reply['content'] or prompt!=timing['prompt_n']:raise ValueError('use a fixed enum response schema and identical prompt for tuning')
                projected=timing['prompt_ms']/1000+a.target_new_tokens/timing['predicted_per_second']
                row=dict(round=r,config=str(Path(config).resolve()),projected_s=projected,reply=reply)
                records.append(row);(out/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items()if k!='reply'}),flush=True)
                if initial.as_dict()!=power_state().as_dict():raise ValueError('power changed')
    scores={c:statistics.median(x['projected_s']for x in records if x['config']==c)for c in {x['config']for x in records}}
    winner=min(scores,key=scores.get)
    record=dict(schema='optimization-tuning-1',objective=f'prompt latency plus projected {a.target_new_tokens} generated tokens',
        scope='Best of measured profiles for this fixed request; not a global optimum or coding-quality comparison.',
        scores=scores,selected=winner,samples=records,power=initial.as_dict())
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(selected=winner,scores=scores)))
if __name__=='__main__':main()
