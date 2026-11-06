#!/usr/bin/env python3
"""scripts/models.py -- model weights for sub0llm profiles: where they are, fetch them, check the dims.

  models.py status                 every profile: is its model in the models directory?
  models.py path   <profile>       print the profile's model folder (for scripts: --gguf "$(models.py path X)")
  models.py check  [<profile>]     compare a profile's dims with its source-of-truth config.json on Hugging Face
  models.py fetch  <profile> [--yes]   list what would be downloaded; download only with --yes

The models directory is --models-dir or $SUB0_MODELS_DIR (no default: weights are large and machine-local).
Profiles are the *.profile files under profiles/ (format: include/sub0/profile.hpp). This script reads the
same files, so a new profile needs no change here. `check` and `fetch --yes` use the network; `status`/`path`
do not. Nothing is downloaded without --yes.
"""
import argparse, fnmatch, json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "sub0llm-models/1"}

# Hugging Face config.json key -> profile key. (hc-count, hc-lowrank and qsa-indexer-* are engine-side
# choices with no config.json counterpart, so they are not checked.)
HF_TO_PROFILE = {
    "hidden_size": "dmodel", "num_hidden_layers": "layers", "num_attention_heads": "heads",
    "num_key_value_heads": "kv-heads", "head_dim": "head-dim",
    "linear_num_key_heads": "gdn-key-heads", "linear_num_value_heads": "gdn-value-heads",
    "linear_key_head_dim": "gdn-key-head-dim", "linear_value_head_dim": "gdn-value-head-dim",
    "full_attention_interval": "gdn-full-attn-stride", "num_experts_per_tok": "experts-per-tok",
    "vocab_size": "vocab-exact",
}


def parse_profile(path: Path) -> dict:
    """Same format as include/sub0/profile.hpp: 'key = value', '#' comments after whitespace."""
    p = {"values": {}, "aliases": [], "name": path.stem, "file": path}
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw
        i = line.find("#")
        while i != -1 and i > 0 and not line[i - 1].isspace():
            i = line.find("#", i + 1)
        if i != -1:
            line = line[:i]
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            sys.exit(f"{path}:{n}: expected 'key = value'")
        k, v = (x.strip() for x in line.split("=", 1))
        k = k.lower()
        if k == "name": p["name"] = v
        elif k == "alias": p["aliases"] += [a.strip() for a in v.split(",") if a.strip()]
        elif k in ("model-source", "model-card", "model-dir", "model-include"): p[k.replace("-", "_")] = v
        else: p["values"][k] = v
    return p


def profiles() -> list:
    d = Path(os.environ.get("SUB0_PROFILE_DIR") or ROOT / "profiles")
    return [parse_profile(f) for f in sorted(d.glob("*.profile"))]


def find(name: str) -> dict:
    for p in profiles():
        if name.lower() in [p["name"].lower()] + [a.lower() for a in p["aliases"]]:
            return p
    sys.exit(f"unknown profile '{name}'; known: {', '.join(p['name'] for p in profiles())}")


def models_dir(arg) -> Path | None:
    v = arg or os.environ.get("SUB0_MODELS_DIR")
    return Path(v) if v else None


def model_path(p, mdir):
    return mdir / p["model_dir"] if mdir and p.get("model_dir") else None


def ggufs(path: Path) -> list:
    return sorted(path.rglob("*.gguf")) if path and path.is_dir() else []


def http_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as f:
        return json.load(f)


def cmd_status(a):
    mdir = models_dir(a.models_dir)
    print(f"models directory: {mdir or '(not set: --models-dir or $SUB0_MODELS_DIR)'}")
    for p in profiles():
        mp = model_path(p, mdir)
        if not p.get("model_dir"): state = "no model metadata"
        elif not mdir: state = "models directory not set"
        else:
            g = ggufs(mp)
            state = f"found, {len(g)} .gguf, {sum(x.stat().st_size for x in g) / 2**30:.1f} GiB" if g else f"NOT FOUND ({mp})"
        print(f"  {p['name']:<22} {state}")
    return 0


def cmd_path(a):
    mp = model_path(find(a.profile), models_dir(a.models_dir))
    if not mp: sys.exit("no model-dir in the profile, or no models directory set")
    print(mp)
    return 0


def expected_from_card(card: str) -> dict:
    c = http_json(f"https://huggingface.co/{card}/raw/main/config.json")
    t = c.get("text_config", c)
    exp = {HF_TO_PROFILE[k]: t[k] for k in HF_TO_PROFILE if k in t}
    # D_FF is the routed-expert width on MoE models (the real shared-expert width is equal), else the dense FFN width.
    if "moe_intermediate_size" in t:
        exp["d-ff"] = t["moe_intermediate_size"]
        if "shared_expert_intermediate_size" in t and t["shared_expert_intermediate_size"] != t["moe_intermediate_size"]:
            print("  note: shared_expert_intermediate_size differs from moe_intermediate_size; d-ff checked against the routed width")
    elif "intermediate_size" in t:
        exp["d-ff"] = t["intermediate_size"]
    exp["num-experts"] = t.get("num_experts", 0)
    rp = t.get("rope_parameters") or {}
    if "rope_theta" in rp or "rope_theta" in t: exp["rope-theta"] = rp.get("rope_theta", t.get("rope_theta"))
    prf = rp.get("partial_rotary_factor", t.get("partial_rotary_factor"))
    if prf and "head_dim" in t: exp["rotary-dim"] = int(prf * t["head_dim"])
    if "tie_word_embeddings" in t: exp["tie-embeddings"] = int(bool(t["tie_word_embeddings"]))
    return exp


def cmd_check(a):
    bad = 0
    for p in ([find(a.profile)] if a.profile else profiles()):
        if not p.get("model_card"):
            print(f"{p['name']}: no model-card in the profile -- nothing to check against")
            continue
        exp = expected_from_card(p["model_card"])
        diffs = []
        for k, want in sorted(exp.items()):
            got = p["values"].get(k)
            if got is None:
                if k in ("num-experts", "tie-embeddings"): continue   # configurator defaults cover absent keys
                diffs.append(f"{k}: profile does not set it; {p['model_card']} says {want}")
            elif float(got) != float(want):
                diffs.append(f"{k}: profile {got}, {p['model_card']} says {want}")
        print(f"{p['name']}: " + ("matches " + p["model_card"] if not diffs else f"{len(diffs)} MISMATCH(ES) vs {p['model_card']}"))
        for d in diffs: print("   " + d)
        bad += bool(diffs)
    return 1 if bad else 0


def cmd_fetch(a):
    p = find(a.profile)
    mdir = models_dir(a.models_dir)
    if not (p.get("model_source") and p.get("model_dir") and p.get("model_include")):
        sys.exit("the profile needs model-source, model-dir and model-include to fetch")
    if not mdir: sys.exit("set --models-dir or $SUB0_MODELS_DIR first")
    info = http_json(f"https://huggingface.co/api/models/{p['model_source']}?blobs=true")
    files = [s for s in info["siblings"] if fnmatch.fnmatch(s["rfilename"], p["model_include"])
             or fnmatch.fnmatch(os.path.basename(s["rfilename"]), p["model_include"])]
    total = sum(s.get("size") or 0 for s in files)
    dest = mdir / p["model_dir"]
    print(f"{p['model_source']} / {p['model_include']} -> {dest}: {len(files)} file(s), {total / 2**30:.1f} GiB")
    for s in files: print(f"   {(s.get('size') or 0) / 2**30:7.1f} GiB  {s['rfilename']}")
    if not files: sys.exit("nothing matches model-include")
    if not a.yes:
        print("dry run: nothing downloaded. Re-run with --yes to download.")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("downloading needs huggingface_hub:  python3 -m pip install huggingface_hub")
    snapshot_download(p["model_source"], allow_patterns=[p["model_include"]], local_dir=str(dest))
    print("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("path").add_argument("profile")
    sub.add_parser("check").add_argument("profile", nargs="?")
    f = sub.add_parser("fetch"); f.add_argument("profile"); f.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    return {"status": cmd_status, "path": cmd_path, "check": cmd_check, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
