"""Dev-only Jinja cross-check of the PoC's supported text-only non-thinking subset."""
import argparse
import importlib.util
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("template", type=Path, help="chat_template.jinja exported from the actual GGUF")
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("qwen_chat",root/"scripts/qwen_chat_poc.py")
chat=importlib.util.module_from_spec(spec);spec.loader.exec_module(chat)
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
def reject(message):raise ValueError(message)
env.globals["raise_exception"]=reject
template=env.from_string(args.template.read_text())
cases=[[('user','Hi')],[('user','Hi'),('assistant','Hello'),('user','Again')],
       [('user','  Hello 🌍\n '),('assistant',' 你好 '),('user',' Next\nline ')]]
for messages in cases:
    rendered=template.render(messages=[{'role':role,'content':content} for role,content in messages],
                             add_generation_prompt=True,enable_thinking=False,preserve_thinking=False)
    if rendered!=chat.render(messages):raise SystemExit("embedded Jinja and supported PoC template differ")
    print(f"matched messages={len(messages)} utf8_bytes={len(rendered.encode('utf-8'))}")
