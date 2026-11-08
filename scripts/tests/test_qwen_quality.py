import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality',ROOT/'scripts/qwen_quality.py')
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)

class QualityTest(unittest.TestCase):
    def fixture(self):return json.loads((ROOT/'perf/quality/qwen27b-v1/fixture.json').read_text())
    def score(self,w,delta=0):
        return dict(schema='qwen-nll-1',vocabulary=248320,input_ids=w['input_ids'],target_ids=w['target_ids'],nll=[2.+delta]*128,top_ids=[0]*128)
    def pairs(self,f,delta):return {w['id']:(self.score(w),self.score(w,delta(w)))for w in f['windows']}
    def test_complete_pass_and_fail_retain_existing_threshold(self):
        f=self.fixture()
        for delta,expected in ((.01,'pass'),(.04,'fail')):
            r=quality.quality_verdict(f,self.pairs(f,lambda w:delta))
            self.assertEqual(r['verdict'],expected);self.assertEqual(r['tokens'],4096);self.assertEqual(r['threshold_nats'],.03)
    def test_cluster_uncertainty_can_refuse_a_token_level_pass(self):
        f=self.fixture();p=self.pairs(f,lambda w:-.06 if int(w['id'].split('-')[-1])%2==0 else .10)
        r=quality.quality_verdict(f,p)
        self.assertLess(r['checks']['global_tokens']['ci95'][1],.03)
        self.assertGreater(r['checks']['global_windows']['ci95'][1],.03)
        self.assertEqual(r['verdict'],'inconclusive')
    def test_missing_duplicate_and_incomplete_domains_refuse(self):
        f=self.fixture();p=self.pairs(f,lambda w:0);p.pop(next(iter(p)))
        with self.assertRaises(ValueError):quality.quality_verdict(f,p)
        bad=copy.deepcopy(f);bad['windows'][1]=bad['windows'][0]
        with self.assertRaises(ValueError):quality.validate_fixture(bad)
    def test_wrong_targets_nonfinite_and_missing_scores_refuse(self):
        w=self.fixture()['windows'][0]
        for change in ('targets','nan','missing'):
            s=copy.deepcopy(self.score(w))
            if change=='targets':s['target_ids'][0]+=1
            if change=='nan':s['nll'][0]=float('nan')
            if change=='missing':s['nll'].pop()
            with self.assertRaises(ValueError):quality.validate_scores(s,w)
    def test_cache_requires_completion_hash_and_finite_memory(self):
        w=copy.deepcopy(self.fixture()['windows'][1]);self.assertFalse(w['capture_logits'])
        with tempfile.TemporaryDirectory()as directory:
            folder=Path(directory);key={'version':1}
            self.assertIsNone(quality.read_complete(folder,key,w))
            quality.save(folder/'scores.json',self.score(w));(folder/'run.log').write_text('OK')
            record=dict(identity=key,files={n:quality.digest(folder/n)for n in ('scores.json','run.log')},process=dict(peak_rss_mib=6000.))
            quality.save(folder/'complete.json',record)
            self.assertIsNotNone(quality.read_complete(folder,key,w))
            record['process']['peak_rss_mib']=float('nan')
            # JSON NaN is accepted by the parser but must never count as a memory pass.
            (folder/'complete.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):quality.read_complete(folder,key,w)
            record['process']['peak_rss_mib']=6000.;quality.save(folder/'complete.json',record)
            quality.save(folder/'scores.json',self.score(w,.001))
            with self.assertRaises(ValueError):quality.read_complete(folder,key,w)
