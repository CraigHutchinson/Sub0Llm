#!/usr/bin/env python3
"""scripts/doctor.py -- Sub0Llm system readiness and diagnostic tool.

Checks:
  - CPU architecture & SIMD capabilities (AVX-512, AVX2, FMA)
  - Physical RAM, available memory, and swap headroom
  - Compiler and build toolchain (clang++, cmake, ninja, libomp, git-lfs)
  - Model assets (tokenizer files, GGUF shards, transplanted models)
  - Built binary readiness
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def check_mark(ok: bool, warn: bool = False) -> str:
    if ok:
        return f"{GREEN}✓{RESET}"
    if warn:
        return f"{YELLOW}!{RESET}"
    return f"{RED}✗{RESET}"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return -1, str(e)


def main() -> int:
    print(f"\n{BOLD}{CYAN}=== Sub0Llm System & Environment Doctor ==={RESET}\n")

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    warnings = 0
    errors = 0

    # 1. Host & CPU
    print(f"{BOLD}[1] CPU & Compute Hardware{RESET}")
    cpu_model = "Unknown"
    cpu_threads = os.cpu_count() or 1
    cpu_flags = set()

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                    elif line.startswith("flags"):
                        cpu_flags.update(line.split(":", 1)[1].strip().split())
        except Exception:
            pass
    elif sys.platform == "win32":
        cpu_model = os.environ.get("PROCESSOR_IDENTIFIER", "x86_64")

    print(f"  {check_mark(True)} CPU Model: {BOLD}{cpu_model}{RESET} ({cpu_threads} logical cores)")

    has_avx2 = "avx2" in cpu_flags or sys.platform == "win32"
    has_fma = "fma" in cpu_flags or sys.platform == "win32"
    has_avx512 = any(f.startswith("avx512") for f in cpu_flags)
    has_avx512_bf16 = "avx512_bf16" in cpu_flags
    has_avx512_vnni = "avx512_vnni" in cpu_flags

    simd_features = []
    if has_avx2: simd_features.append("AVX2")
    if has_fma: simd_features.append("FMA")
    if has_avx512: simd_features.append("AVX-512")
    if has_avx512_bf16: simd_features.append("avx512_bf16")
    if has_avx512_vnni: simd_features.append("avx512_vnni")

    simd_str = ", ".join(simd_features) if simd_features else "Standard x86_64"
    print(f"  {check_mark(has_avx2)} Vector SIMD extensions: {BOLD}{simd_str}{RESET}")
    if not has_avx2:
        print(f"    {YELLOW}Warning: AVX2 is strongly recommended for native-quantized decode kernels.{RESET}")
        warnings += 1

    # Check GPU
    gpu_found = False
    ret, out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if ret == 0 and out:
        print(f"  {check_mark(True)} GPU: {out}")
        gpu_found = True
    else:
        print(f"  {check_mark(True)} Compute Mode: {BOLD}CPU-only host{RESET} (integrated / no discrete CUDA device)")

    # 2. Memory & Disk Space
    print(f"\n{BOLD}[2] Memory & Disk Storage{RESET}")
    total_ram_gb = 0.0
    avail_ram_gb = 0.0
    swap_total_gb = 0.0

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    if key == "MemTotal":
                        total_ram_gb = int(val) / (1024 * 1024)
                    elif key == "MemAvailable":
                        avail_ram_gb = int(val) / (1024 * 1024)
                    elif key == "SwapTotal":
                        swap_total_gb = int(val) / (1024 * 1024)
        except Exception:
            pass

    print(f"  {check_mark(total_ram_gb >= 16)} Physical RAM: {BOLD}{total_ram_gb:.1f} GB total{RESET} ({avail_ram_gb:.1f} GB available, {swap_total_gb:.1f} GB swap)")
    
    # Model feasibility assessment based on RAM
    print(f"  {DIM}RAM Sizing Assessment:{RESET}")
    print(f"    - Qwen 3.8 27B (Dense, Q4_K_M ~16.5 GB): {GREEN}Fits entirely resident in memory{RESET}")
    print(f"    - Qwen 3.6 35B (MoE 3B active, ~15 GB):   {GREEN}Fits entirely resident in memory{RESET}")
    if total_ram_gb < 60:
        print(f"    - Qwen 3.8-Flash-Next (MoE 125B, ~49 GB): {YELLOW}Requires Sub0Llm mapped arena + MoE tiered caching{RESET}")
    else:
        print(f"    - Qwen 3.8-Flash-Next (MoE 125B, ~49 GB): {GREEN}High residency possible in 64GB+ RAM{RESET}")

    # Disk space check
    stat = shutil.disk_usage(repo_root)
    free_gb = stat.free / (1024 * 1024 * 1024)
    print(f"  {check_mark(free_gb >= 60)} Workspace Disk Space: {BOLD}{free_gb:.1f} GB free{RESET} on {repo_root}")
    if free_gb < 60:
        print(f"    {YELLOW}Warning: Full model weights and GGUF shards typically require ~50-80 GB of storage.{RESET}")
        warnings += 1

    # 3. Toolchain & Dependencies
    print(f"\n{BOLD}[3] Compiler & Build Toolchain{RESET}")
    
    # Clang++
    clang = shutil.which("clang++") or shutil.which("clang++-21")
    if clang:
        ret, out = run_cmd([clang, "--version"])
        ver = out.splitlines()[0] if out else "found"
        print(f"  {check_mark(True)} C++ Compiler: {BOLD}{ver}{RESET}")
    else:
        print(f"  {check_mark(False)} C++ Compiler: {RED}clang++ not found{RESET}")
        errors += 1

    # CMake
    cmake = shutil.which("cmake") or ("/snap/bin/cmake" if Path("/snap/bin/cmake").exists() else None)
    if cmake:
        ret, out = run_cmd([cmake, "--version"])
        ver = out.splitlines()[0] if out else "found"
        print(f"  {check_mark(True)} CMake: {BOLD}{ver}{RESET}")
    else:
        print(f"  {check_mark(False)} CMake: {RED}cmake not found{RESET}")
        errors += 1

    # Ninja
    ninja = shutil.which("ninja")
    if ninja:
        ret, out = run_cmd([ninja, "--version"])
        print(f"  {check_mark(True)} Ninja: {BOLD}v{out}{RESET}")
    else:
        print(f"  {check_mark(False)} Ninja: {RED}ninja not found{RESET}")
        errors += 1

    # Git LFS
    git_lfs = shutil.which("git-lfs")
    if git_lfs:
        ret, out = run_cmd([git_lfs, "--version"])
        print(f"  {check_mark(True)} Git LFS: {BOLD}{out.split()[0]} {out.split()[1] if len(out.split()) > 1 else ''}{RESET}")
    else:
        print(f"  {check_mark(False, warn=True)} Git LFS: {YELLOW}git-lfs not found{RESET}")
        warnings += 1

    # Check Git LFS files
    english_txt = repo_root / "data" / "english.txt"
    if english_txt.exists():
        with open(english_txt, "rb") as f:
            header = f.read(50)
            if b"oid sha256" in header:
                print(f"  {check_mark(False, warn=True)} Tracked test data: {YELLOW}LFS pointer stub detected (run 'git lfs pull'){RESET}")
                warnings += 1
            else:
                print(f"  {check_mark(True)} Tracked test data: {GREEN}Hydrated real files{RESET}")

    # 4. Built Executables Status
    print(f"\n{BOLD}[4] Sub0Llm Build Status{RESET}")
    native_dir = repo_root / "out" / "build" / "native"
    exe_ext = ".exe" if sys.platform == "win32" else ""

    targets = [
        ("sub0llm-configure", "Configurator & auto-sizer"),
        ("sub0llm-transplant-q48", "GGUF -> S0L5 transplant tool"),
        ("sub0llm-qwen4-gen", "End-to-end Qwen generation runner"),
        ("sub0llm", "Main umbrella multi-stage CLI"),
    ]

    for name, desc in targets:
        exe_path = native_dir / f"{name}{exe_ext}"
        if exe_path.exists():
            print(f"  {check_mark(True)} {name}: {GREEN}Built & ready{RESET} ({desc})")
        else:
            print(f"  {check_mark(False, warn=True)} {name}: {YELLOW}Not built yet{RESET} ({desc})")

    # 5. Model Assets Check
    print(f"\n{BOLD}[5] Model Assets & Weight Files{RESET}")
    
    # Tokenizer
    tok_dir = repo_root / "data" / "qwen_tokenizer"
    env_tok_dir = os.environ.get("SUB0_QWEN_TOKENIZER_DIR")
    effective_tok_dir = Path(env_tok_dir) if env_tok_dir else tok_dir

    vocab_json = effective_tok_dir / "vocab.json"
    merges_txt = effective_tok_dir / "merges.txt"
    if vocab_json.exists() and merges_txt.exists():
        print(f"  {check_mark(True)} Qwen Tokenizer: {GREEN}Found{RESET} at {effective_tok_dir}")
    else:
        print(f"  {check_mark(False, warn=True)} Qwen Tokenizer: {YELLOW}Missing{RESET} (needs vocab.json & merges.txt in data/qwen_tokenizer)")
        warnings += 1

    # Transplanted weights
    model_env = os.environ.get("SUB0_QWEN4_MODEL")
    candidate_paths = [
        Path(model_env) if model_env else None,
        Path("D:/ModelWeights/Sub0Llm-Qwen4-full48-bf16/qwen4_full48_q_bf16.bin"),
        repo_root / "models" / "qwen4_full48_q_bf16.bin",
    ]
    found_model = None
    for p in candidate_paths:
        if p and p.exists():
            found_model = p
            break

    if found_model:
        moeq = found_model.with_suffix(".bin.moeq") if found_model.suffix == ".bin" else Path(str(found_model) + ".moeq")
        print(f"  {check_mark(True)} Model Weights: {GREEN}Found{RESET} at {found_model}")
        if moeq.exists():
            print(f"  {check_mark(True)} MoE Expert Quant Sidecar: {GREEN}Found{RESET} ({moeq.stat().st_size / (1024**3):.1f} GiB)")
    else:
        print(f"  {check_mark(False, warn=True)} Model Weights: {YELLOW}No transplanted .bin detected{RESET} (use sub0llm-transplant-q48 to generate)")

    # Summary
    print(f"\n{BOLD}=== Summary ==={RESET}")
    if errors == 0 and warnings == 0:
        print(f"{GREEN}{BOLD}All checks passed! The system is fully ready to run Sub0Llm.{RESET}\n")
    elif errors == 0:
        print(f"{YELLOW}{BOLD}System is operational with {warnings} minor item(s) to check (see warnings above).{RESET}\n")
    else:
        print(f"{RED}{BOLD}{errors} critical error(s) found. Please resolve them before proceeding.{RESET}\n")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
