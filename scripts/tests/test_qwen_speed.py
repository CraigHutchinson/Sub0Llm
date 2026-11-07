import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'compare_qwen_speed.py'
spec=importlib.util.spec_from_file_location('speed',path)
speed=importlib.util.module_from_spec(spec);spec.loader.exec_module(speed)


class PhaseAccountingTest(unittest.TestCase):
    def record(self,batched=False):
        return {'schema':'qwen-phase-1','prompt_tokens':13,'decode_evaluations':8,'batched_prefill':batched,
                'f32_attention':True,'setup_ms':20.,'prefill_ms':[130.] if batched else [10.]*13,
                'decode_ms':[2.]*8,'top_ids':[1]*((1 if batched else 13)+8)}

    def test_prompt_denominator_is_tokens_not_number_of_batches(self):
        for batch in (False,True):
            rate=speed.phase_rates(self.record(batch),13,8)
            self.assertEqual(rate['prefill_tok_s'],100.)
            self.assertEqual(rate['decode_tok_s'],500.)
            self.assertEqual(len(rate['generated_ids']),9)
            self.assertAlmostEqual(rate['setup_plus_prefill_s'],.150)

    def test_incomplete_or_wrong_counts_refuse(self):
        for key,value in (('prompt_tokens',12),('decode_evaluations',9),('decode_ms',[2.]*7),
                          ('prefill_ms',[10.]*12),('top_ids',[1]*20),('batched_prefill',None)):
            data=self.record();data[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):speed.phase_rates(data,13,8)

    def test_nonfinite_zero_or_boolean_durations_refuse(self):
        for invalid in (0,-1,float('nan'),float('inf'),True):
            data=self.record();data['decode_ms'][0]=invalid
            with self.subTest(value=invalid),self.assertRaises(ValueError):speed.phase_rates(data,13,8)

    def test_background_cpu_excludes_the_benchmark_child(self):
        hz=speed.os.sysconf('SC_CLK_TCK')
        self.assertAlmostEqual(speed.background_fraction((0,0),(8*hz,10*hz),8.),0.)
        self.assertAlmostEqual(speed.background_fraction((0,0),(9*hz,10*hz),8.),.1)

    def test_summary_keeps_raw_variation_and_excludes_only_flagged_samples(self):
        def row(value,warm=False,tainted=False):
            return {'arm':'A','warmup':warm,'tainted':tainted,'rates':{name:value for name in
                ('prefill_tok_s','decode_tok_s','setup_s','setup_plus_prefill_s','process_wall_s','peak_rss_mib')}}
        result=speed.summary([row(1),row(2),row(3),row(100,True),row(100,False,True)])['A']['decode_tok_s']
        self.assertEqual(result['samples'],[1,2,3]);self.assertEqual(result['median'],2)
