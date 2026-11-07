import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('speed',root/'scripts/compare_qwen_speed.py')
speed=importlib.util.module_from_spec(spec);spec.loader.exec_module(speed)
with tempfile.TemporaryDirectory()as directory:
    for mode in ('serial','batch'):
        out=Path(directory)/f'{mode}.json'
        subprocess.run([sys.argv[1],str(out),mode],check=True)
        rates=speed.phase_rates(json.loads(out.read_text()),13,8)
        assert rates['prefill_tok_s']==100.
        assert rates['decode_tok_s']==500.
        assert len(rates['generated_ids'])==9
    out=Path(directory)/'profile.json'
    subprocess.run([sys.argv[1],str(out),'profile'],check=True)
    profile=json.loads(out.read_text())
    assert profile['schema']=='qwen-projection-profile-1'
    entries={(e['phase'],e['type']):e for e in profile['entries']}
    assert len(entries)==3
    assert entries['prefill',19]==dict(phase='prefill',type=19,calls=2,failures=0,encoded_bytes=250,logical_flops=500,wall_ms=5.)
    assert entries['decode',19]['wall_ms']==4.
    assert entries['decode',19]['calls']==1
    assert entries['decode',8]['failures']==1
print('Native serial and batched phase denominators agree')
