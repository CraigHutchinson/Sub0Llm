"""Conversation commit/rollback and completion reporting without model inference."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "qwen_chat_poc.py"
spec = importlib.util.spec_from_file_location("qwen_chat", path)
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


class ChatTest(unittest.TestCase):
    def session(self):
        return chat.Session("backend", "model", "tokenizer", 8, 1)

    def backend(self, args, **kwargs):
        if "--inspect-context" in args:
            return subprocess.CompletedProcess(args, 0, stdout="CONTEXT prompt=15 reserve=7 capacity=128\n", stderr="")
        Path(args[args.index("--response-file") + 1]).write_bytes(b"OK")
        return subprocess.CompletedProcess(args, 0, stdout=b"TEXT_RESULT generated=1 finish=eos\n", stderr=b"")

    def test_plain_non_thinking_template(self):
        self.assertEqual(chat.render([("user", " Hi ")]), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertIn("assistant\nHello<|im_end|>",chat.render([("user","Hi"),("assistant","Hello"),("user","Again")]))

    @patch.object(chat.subprocess, "run")
    def test_explicit_kernel_reaches_inspection_and_decode_while_default_omits_it(self, run):
        self.assertNotIn("--iq1s-kernel",self.session().base)
        run.side_effect=self.backend
        session=chat.Session("backend","model","tokenizer",8,1,"native")
        session.turn("Hi")
        for call in run.call_args_list:
            args=call.args[0]
            self.assertEqual(args[args.index("--iq1s-kernel")+1],"native")

    @patch.object(chat.subprocess, "run")
    def test_success_commits_reply_and_eos_reason_then_reset(self, run):
        run.side_effect = self.backend
        session=self.session()
        self.assertEqual(session.turn("Hi"), "OK")
        self.assertEqual(session.messages, [("user","Hi"),("assistant","OK")])
        self.assertEqual(session.last_result, {"generated_tokens":1,"finish_reason":"eos"})
        session.reset()
        self.assertEqual(session.messages, [])
        self.assertIsNone(session.last_context)
        self.assertIsNone(session.last_result)

    @patch.object(chat.subprocess, "run")
    def test_length_reason_is_not_reported_as_eos(self, run):
        def capped(args, **kwargs):
            result=self.backend(args, **kwargs)
            if "--inspect-context" not in args:result.stdout=b"TEXT_RESULT generated=8 finish=length\n"
            return result
        run.side_effect=capped
        session=self.session();session.turn("Hi")
        self.assertEqual(session.last_result["finish_reason"], "length")

    @patch.object(chat.subprocess, "run")
    def test_context_refusal_preserves_accepted_history(self, run):
        run.return_value=subprocess.CompletedProcess([],1,stdout="CONTEXT prompt=200 reserve=7 capacity=128\n",stderr="context window exceeded")
        session=self.session();session.messages=[("user","old"),("assistant","OK")]
        before=list(session.messages)
        with self.assertRaises(RuntimeError):session.turn("Too long")
        self.assertEqual(session.messages, before)
        self.assertEqual(run.call_count,1)

    @patch.object(chat.subprocess, "run")
    def test_missing_completion_or_invalid_utf8_rolls_back(self, run):
        for invalid_utf8 in (False,True):
            def broken(args, **kwargs):
                result=self.backend(args, **kwargs)
                if "--inspect-context" not in args:
                    if invalid_utf8:Path(args[args.index("--response-file")+1]).write_bytes(b"\xf0\x9f")
                    else:result.stdout=b""
                return result
            run.side_effect=broken
            session=self.session()
            with self.assertRaises(RuntimeError):session.turn("Hi")
            self.assertEqual(session.messages, [])

    @patch.object(chat.subprocess, "run")
    def test_only_final_completion_diagnostic_is_accepted(self, run):
        def with_quoted_diagnostic(args, **kwargs):
            result=self.backend(args, **kwargs)
            if "--inspect-context" not in args:
                result.stdout=b"TEXT_RESULT generated=100 finish=length\nquoted text\nTEXT_RESULT generated=1 finish=eos\n"
            return result
        run.side_effect=with_quoted_diagnostic
        session=self.session();session.turn("Hi")
        self.assertEqual(session.last_result, {"generated_tokens":1,"finish_reason":"eos"})

    def test_cli_can_use_local_model_default_for_context_inspection(self):
        import contextlib
        import io
        import sys
        output=io.StringIO()
        with patch.object(sys,"argv",[str(path),"--turn","/context"]),contextlib.redirect_stdout(output):
            self.assertEqual(chat.main(),0)
        self.assertIn('"messages": 0',output.getvalue())
