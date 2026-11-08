"""Control-loop failures must never become an optimization pass."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'optimization/iterate.py'
spec=importlib.util.spec_from_file_location('optimization_loop',path);loop=importlib.util.module_from_spec(spec);spec.loader.exec_module(loop)


def proposal(edits=None):
    return dict(hypothesis='reduce dependency latency',mechanism='independent sums',edits=edits or [],no_change_reason='none found')


class OptimizationTest(unittest.TestCase):
    def item(self):
        return dict(schema='optimization-item-1',id='test',issue='#68',goal='make x faster',constraints=['preserve output'],
                    editable_files=['x.hpp'],context=[dict(path='x.hpp')],qualification_scope='kernel-only',
                    max_attempts=4,stagnation_window=2,max_recoveries=2,wall_seconds=30,max_context_chars=2000,
                    provider_command=[sys.executable,'provider.py'],identity_files=[])

    def test_scope_and_atomic_edit_validation(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'x.hpp').write_text('x=1;\nx=1;\n')
            p=proposal([dict(path='x.hpp',old='x=1;',new='x=2;')])
            loop.validate_proposal(p,self.item())
            with self.assertRaises(ValueError):loop.apply(root,p,self.item())
            self.assertEqual((root/'x.hpp').read_text(),'x=1;\nx=1;\n')
            for name in ('../x.hpp','/tmp/x.hpp','.git/config'):
                with self.assertRaises(ValueError):loop.validate_proposal(proposal([dict(path=name,old='a',new='b')]),self.item())
            (root/'escape').symlink_to('/tmp')
            with self.assertRaises(ValueError):loop.source(root,'escape/x.hpp')

    def test_invalid_command_contract_is_refused(self):
        item=self.item();item['provider_command']=[42]
        with self.assertRaises(ValueError):loop.validate_item(item)
        item=self.item();item['evaluation']=dict(command=['tool'],timeout=-1)
        with self.assertRaises(ValueError):loop.validate_item(item)

    def test_compact_schema_and_edit_limit(self):
        item=self.item();item.update(max_edits=1,max_edit_chars=4)
        schema=loop.response_schema(item)
        self.assertEqual(schema['properties']['edits']['maxItems'],1)
        self.assertEqual(schema['properties']['edits']['items']['properties']['old']['maxLength'],4)
        with self.assertRaises(ValueError):loop.validate_proposal(proposal([dict(path='x.hpp',old='12345',new='0')]),item)

    def test_rationale_does_not_hide_duplicate_patch(self):
        a=proposal([dict(path='x.hpp',old='a',new='b')]);b={**a,'hypothesis':'a different story'}
        self.assertEqual(loop.fingerprint(a),loop.fingerprint(b))

    def test_evaluation_requires_scope_identity_correctness_memory_and_speed(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'x.hpp').write_text('x=2;')
            good=dict(schema='optimization-evaluation-1',qualification_scope='kernel-only',baseline_commit='abc',
                      candidate_files={'x.hpp':loop.sha(root/'x.hpp')},correctness='pass',memory='pass',performance='improved',evidence=['x.json'])
            self.assertTrue(loop.judge(good,self.item(),root,'abc'))
            for k,v in [('correctness','fail'),('memory','fail'),('memory','not_measured'),('performance','inconclusive'),('performance','regressed')]:
                self.assertFalse(loop.judge({**good,k:v},self.item(),root,'abc'))
            with self.assertRaises(ValueError):loop.judge({**good,'qualification_scope':'model'},self.item(),root,'abc')
            with self.assertRaises(ValueError):loop.judge({**good,'candidate_files':{}},self.item(),root,'abc')

    def test_empty_proposals_recover_continue_and_stall_with_resume_contract(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);subprocess.run(['git','init','-q',str(root)],check=True)
            for k,v in [('user.name','Test'),('user.email','test@example.invalid')]:subprocess.run(['git','-C',str(root),'config',k,v],check=True)
            (root/'x.hpp').write_text('x=1;\n');subprocess.run(['git','-C',str(root),'add','x.hpp'],check=True)
            subprocess.run(['git','-C',str(root),'commit','-qm','base'],check=True)
            item=self.item();item['provider_command']=[sys.executable,'provider.py'];(root/'item.json').write_text(json.dumps(item))
            fake=dict(schema='optimization-inference-1',content=json.dumps(proposal()),finish_reason='stop')
            (root/'provider.py').write_text('import json; print('+repr(json.dumps(fake))+')\n')
            with patch.object(loop,'ROOT',root):
                result=loop.run(root/'item.json',root/'out',proposals_only=True)
                self.assertEqual(result['status'],'stalled');self.assertEqual(len(result['attempts']),4)
                self.assertEqual(result['best_commit'],result['base_commit']);self.assertEqual(result['recoveries'],2)
                self.assertIn('recovery_hint',result['attempts'][1]);self.assertIn('recovery_hint',result['attempts'][3])
                self.assertNotEqual(result['attempts'][1]['recovery_hint'],result['attempts'][3]['recovery_hint'])
                with self.assertRaises(ValueError):loop.run(root/'item.json',root/'out',resume=True,proposals_only=True)

    def test_timeout_allows_provider_to_reap_separate_server_group(self):
        import os
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);script=root/'provider.py';pid=root/'child.pid'
            script.write_text("import subprocess,signal,sys,pathlib\nchild=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)\npathlib.Path(sys.argv[1]).write_text(str(child.pid))\ndef stop(s,f):\n raise SystemExit(2)\nsignal.signal(signal.SIGTERM,stop)\ntry:child.wait()\nfinally:child.terminate();child.wait()\n")
            with self.assertRaises(subprocess.TimeoutExpired):
                loop.command([sys.executable,str(script),str(pid)],root,root/'log',.3)
            with self.assertRaises(ProcessLookupError):os.kill(int(pid.read_text()),0)

    def test_measured_candidate_advances_incumbent_without_changing_root(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);subprocess.run(['git','init','-q',str(root)],check=True)
            for k,v in [('user.name','Test'),('user.email','test@example.invalid')]:subprocess.run(['git','-C',str(root),'config',k,v],check=True)
            (root/'x.hpp').write_text('x=1;\n');subprocess.run(['git','-C',str(root),'add','x.hpp'],check=True);subprocess.run(['git','-C',str(root),'commit','-qm','base'],check=True)
            response=dict(schema='optimization-inference-1',content=json.dumps(proposal([dict(path='x.hpp',old='x=1;',new='x=2;')])),finish_reason='stop')
            (root/'provider.py').write_text('print('+repr(json.dumps(response))+')\n')
            evaluator=root/'evaluate.py';evaluator.write_text("import json,pathlib,hashlib,subprocess,sys\nb,c,o=map(pathlib.Path,sys.argv[1:]);proof=o.parent/'proof.json';proof.write_text('{}')\no.write_text(json.dumps(dict(schema='optimization-evaluation-1',qualification_scope='kernel-only',baseline_commit=subprocess.check_output(['git','-C',str(b),'rev-parse','HEAD'],text=True).strip(),candidate_files={'x.hpp':hashlib.sha256((c/'x.hpp').read_bytes()).hexdigest()},correctness='pass',memory='pass',performance='improved',evidence=[str(proof)])))\n")
            item=self.item();item.update(max_attempts=3,max_recoveries=1,provider_command=[sys.executable,str(root/'provider.py')],evaluation=dict(command=[sys.executable,str(evaluator),'{baseline}','{candidate}','{result}'],timeout=5))
            (root/'item.json').write_text(json.dumps(item))
            with patch.object(loop,'ROOT',root):result=loop.run(root/'item.json',root/'out')
            self.assertNotEqual(result['best_commit'],result['base_commit']);self.assertEqual(result['attempts'][0]['status'],'improved_for_review')
            self.assertEqual(result['attempts'][1]['status'],'duplicate');self.assertEqual((root/'x.hpp').read_text(),'x=1;\n')
            item.update(max_attempts=1);(root/'next.json').write_text(json.dumps(item))
            with patch.object(loop,'ROOT',root):next_run=loop.run(root/'next.json',root/'next-out',base=result['best_commit'])
            self.assertEqual(next_run['base_commit'],result['best_commit']);self.assertEqual((root/'x.hpp').read_text(),'x=1;\n')

    def test_truncated_inference_never_applies_edits(self):
        p=proposal([dict(path='x.hpp',old='x=1;',new='x=2;')])
        reply=dict(schema='optimization-inference-1',content=json.dumps(p),finish_reason='length')
        with self.assertRaises(ValueError):loop.parse_reply(reply,self.item())
        self.assertEqual(loop.parse_reply({**reply,'finish_reason':'stop'},self.item()),p)
        for malformed in ([],None,dict(schema='optimization-inference-1',finish_reason='stop',content=[])):
            with self.assertRaises(ValueError):loop.parse_reply(malformed,self.item())
        with self.assertRaises(ValueError):loop.validate_proposal({**proposal(),'command':'rm'},self.item())

if __name__=='__main__':unittest.main()
