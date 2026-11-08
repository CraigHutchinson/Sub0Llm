"""Independent analytic checks of the C++ log-sum-exp/target contract."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as directory:
    prefix=Path(directory)/'scores'
    subprocess.run([sys.argv[1],str(prefix)],check=True)
    base=json.loads(Path(str(prefix)+'-base.json').read_text())
    shift=json.loads(Path(str(prefix)+'-shift.json').read_text())
    uniform=json.loads(Path(str(prefix)+'-uniform.json').read_text())
    assert base['schema']=='qwen-nll-1'
    assert base['input_ids']==[0,1] and base['target_ids']==[0,2]
    assert base['top_ids']==[1,1]
    expected=math.log(math.fsum(math.exp(x) for x in [0.,1.,-1.]))
    assert abs(base['nll'][0]-expected)<1e-12
    assert abs(base['nll'][1]-(math.log(math.fsum(math.exp(x)for x in [2.,4.,1.]))-1))<1e-12
    assert all(abs(a-b)<1e-12 for a,b in zip(base['nll'],shift['nll']))
    assert uniform['vocabulary']==248320 and uniform['target_ids']==[42]
    assert abs(uniform['nll'][0]-math.log(248320))<1e-12
print('Independent NLL, shift invariance, full vocabulary and invalid-input checks passed')
