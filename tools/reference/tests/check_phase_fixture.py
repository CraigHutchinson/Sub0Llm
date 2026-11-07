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
print('Native serial and batched phase denominators agree')
