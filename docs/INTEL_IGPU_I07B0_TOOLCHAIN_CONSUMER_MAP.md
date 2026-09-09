# I07b.0 Windows toolchain and device-consumer map

Date: 2026-09-09. This is the Phase-1 executable-groundwork record required by
[the follow-up runbook](INTEL_IGPU_FOLLOWUP_RUNBOOK.md#phase-1--freeze-inputs-and-build-boundaries).
It is a source and installed-file audit. It adds no backend, selector, build target, ABI, or runtime
qualification.

The consumer census is against current local `main` at
`1aab572aef944449743decd620fa4e92b703c046`. The research worktree is
`research/intel-groundwork` at `6cefe894e7c74b9bf8a894e7cbc60f72857d9dce`; its merge base with `main` is
that same `1aab572` commit. The branch-only Intel probes and documents do not change the production
consumers below.

## Decision record

1. Keep generated `HAS_CUDA`, CMake `SUB0_BUILD_CUDA`, CUDA architecture/memory facts, and
   `ComputeBackend::Gpu` CUDA-specific. Intel must not make any of them true. They currently influence
   CUDA training defaults, worker counts, memory reporting, tuning, source selection, and tests; treating
   them as generic device facts would send an inference-only Intel build into CUDA/training behavior.
2. Reuse `SUB0_BUILD_DEVICE` as the generic, target-local fact that exactly one implementation of the
   neutral device seam is linked into that consumer. Today it is derived in
   [`device_backend.hpp`](../include/sub0/device_backend.hpp#L23-L28), and no code reads it. There is no
   competing semantic to preserve. Define it on `sub0_gen` and relevant tests when CUDA or Intel is
   linked; do not define it on `sub0_train` for the first Intel build.
3. The smallest eventual Intel selection is `SUB0_DEVICE=INTEL`, beside `NONE` and `CUDA`, resolved at
   CMake configure time. Preserve today's default behavior by resolving `AUTO` to CUDA only through the
   existing detection path; do not auto-select Intel in the first delivery. The selected Intel target
   receives a target-local `SUB0_DEVICE_INTEL=1` implementation selector plus the generic
   `SUB0_BUILD_DEVICE=1`. `SUB0_COMPUTE` remains `CPU` and the generated `COMPUTE_MODE` remains
   `ComputeBackend::Cpu`, so training, tuning, and CPU fallback keep their current semantics.
4. Do not emit a speculative `HAS_INTEL` analogue. The fixed backend selection is a build fact, while
   artifact/operator support is read once from the POD ABI/capability handshake. The Intel backend must
   be compiled against the same generated configuration and report its ABI size/version,
   `MODEL_ARCH_ID`, `PARAM_FLOATS`, `VOCAB`, and `SEQ_LEN`; `sub0_gen` compares them before upload.
   Backend choice is deliberately variable between a CPU training build and an inference build and does
   not belong in the checkpoint architecture fingerprint.

The ABI handshake must not return the current `Sub0DeviceCaps::name` pointer across the DLL boundary.
Its native wire record uses only explicitly sized integers and an inline fixed-capacity provider name;
it is standard-layout and trivially copyable, with compile-time size/offset assertions on both sides.
The first export is a non-throwing query of the form `status query_abi(out_record, out_record_bytes)` so
the host can reject version, size and generated-config mismatches before any allocation or queue
creation. The existing ergonomic `Sub0DeviceCaps` remains a host-side view constructed from that
validated wire record. Any diagnostic text uses a caller-owned bounded output buffer or an equally
fixed inline record, never a backend-owned pointer whose lifetime could end at unload.

`SUB0_DEVICE=INTEL` is a proposal for I07b/I11, not a Phase-1 option. Per
[`AGENTS.md`](../AGENTS.md#8-only-add-the-surface-area-actually-consumed), it must land only with the
real Intel target and the consumers described here.

## Complete build/config consumer census

| Symbol | Consumer | Present meaning and consequence |
|---|---|---|
| `SUB0_HAS_CUDA` | [`cmake/Backends.cmake`](../cmake/Backends.cmake#L42-L83) | Detects a toolkit plus a visible `nvidia-smi` device, then records CUDA arch, dedicated VRAM, and Windows shared memory. |
| `SUB0_BUILD_CUDA` | [`cmake/Backends.cmake`](../cmake/Backends.cmake#L85-L118) | Derived from `SUB0_COMPUTE_RESOLVED`; CPU makes it false, GPU requires detected CUDA and makes it true, HYBRID is rejected. |
| `SUB0_BUILD_CUDA` | [`cmake/Backends.cmake`](../cmake/Backends.cmake#L137-L153) and [`sub0_build_facts.hpp.in`](../cmake/sub0_build_facts.hpp.in#L8-L13) | Becomes integer `SUB0_BUILD_CUDA_INT`, which is baked into the configurator as `build_facts::HAS_CUDA`; the non-CUDA branch also zeros CUDA arch/memory. |
| `SUB0_BUILD_CUDA` | [root CMake](../CMakeLists.txt#L156-L168) | Enables the CUDA language only for the CUDA build. |
| `SUB0_BUILD_CUDA` | [root CMake](../CMakeLists.txt#L337-L371) | Creates `sub0_backend_cuda`, compiles `backend.cu` as CUDA C++20 for the detected architecture, links `CUDA::cudart` and `CUDA::cublas`, and creates `sub0-cuda-selftest`. |
| `SUB0_BUILD_CUDA` | [root CMake](../CMakeLists.txt#L398-L423) | Privately links the CUDA DLL and defines the macro on both `sub0_train` and `sub0_gen`. These are the two production CUDA import edges. |
| `SUB0_BUILD_CUDA` | [`tests/CMakeLists.txt`](../tests/CMakeLists.txt#L214-L230) | Adds CUDA tests to `sub0_tests`, links the CUDA DLL, and selects the header bridge. Lines 247-252 add the toolkit DLL directory during CTest discovery/run. |
| `SUB0_BUILD_CUDA` | [`device_backend.hpp`](../include/sub0/device_backend.hpp#L23-L28) | Currently aliases `SUB0_BUILD_DEVICE`; lines 122-226 select inline neutral-to-CUDA forwards. |
| `SUB0_BUILD_CUDA` | [`train_stage.cpp`](../src/train_stage.cpp#L1001-L1083) | Compiles the CUDA training-session enable body. The same guard wraps device step, backward, synchronization, shutdown, and CUDA tuning at lines 1100, 1144, 1173, 1186, 3328, and 3516. These must stay CUDA-only for the first Intel DLL. |
| `HAS_CUDA` | [`configurator.cpp`](../tools/configurator.cpp#L683-L690) | Initializes the user-overridable CUDA/build facts. Lines 875-923 parse `--has-cuda`, default `--compute` to GPU only for CUDA, and force an unavailable GPU/HYBRID request back to CPU. |
| `HAS_CUDA` | [`configurator.cpp`](../tools/configurator.cpp#L1591-L1600) | Enables the CUDA VRAM-based training-batch estimate. |
| `HAS_CUDA` / `ComputeBackend` | [`configurator.cpp`](../tools/configurator.cpp#L1767-L1778) | Emits the generated CUDA fact and `ComputeBackend { Cpu, Gpu, Hybrid }`/`COMPUTE_MODE`. |
| `HAS_CUDA` | [`cli_stages.hpp`](../include/sub0/cli_stages.hpp#L149-L154) | Chooses CUDA's default training batch instead of the CPU threads-times-windows default. |
| `HAS_CUDA` | [`decode.hpp`](../include/sub0/decode.hpp#L44-L55) | Rejects device decode before consulting capabilities or initializing. This is the production guard that must use the generic linked fact/capability path for Intel. |
| `HAS_CUDA` | [`train_stage.cpp`](../src/train_stage.cpp#L1001-L1015) | Further gates CUDA `GpuTrainer::enable`. Lines 3191, 3219, and 3329 use it for CUDA tune selection/provenance/entry. Keep all of these CUDA-specific. |
| `HAS_CUDA` | [`backend.cpp`](../src/backends/cpu/backend.cpp#L1881-L1898) | Prints CUDA availability and CUDA memory in the CPU engine's configuration banner. Do not make this claim for Intel. |
| `HAS_CUDA` | [`layout_tests.cpp`](../tests/layout_tests.cpp#L198-L214) | Enforces that a CUDA build has CUDA arch/VRAM and a non-CUDA build has zero CUDA arch. |
| `ComputeBackend` | [`backend.cpp`](../src/backends/cpu/backend.cpp#L1836-L1863) | A `Gpu` build reports one host worker; CPU/HYBRID report `DEFAULT_THREADS`. Lines 1881-1898 repeat the distinction in the normal config banner. |
| `ComputeBackend` | [`train_stage.cpp`](../src/train_stage.cpp#L3803-L3835) | Labels device memory planning inapplicable when compute mode is CPU. This report is CUDA-training-oriented today and must not be repurposed as Intel inference capacity. |
| `ComputeBackend` | [`layout_tests.cpp`](../tests/layout_tests.cpp#L198-L204) | Enforces ternary builds as CPU compute. |
| `SUB0_BUILD_DEVICE` | [`device_backend.hpp`](../include/sub0/device_backend.hpp#L23-L28) | Definition only. Repo-wide source search finds no preprocessor or expression consumer after this alias. It is therefore available for the generic linked-device meaning above. |

Comments in `cuda_tests.cpp`, `engine_tests.cpp`, and `mock_device_backend.cpp` mention these symbols but
do not add further behavior. `backend.cu` includes generated CUDA facts through `layout.hpp`; its real
compile-time compatibility guards are vendor-specific and remain in that target.

## Neutral seam consumers by responsibility

[`device_backend.hpp`](../include/sub0/device_backend.hpp#L30-L45) defines the current eight-field POD
capability record. Its mock branch is lines 66-120, CUDA declarations/inline forwards are lines 122-226,
and no-device fail-fast stubs are lines 228-253. No production source currently exports native
`sub0_dev_*` symbols: the header inlines them to `sub0_cuda_*`. I07a's native-symbol cleanup is therefore
a prerequisite for a second implementation with the same symbol names.

| Class | Consumer and exact calls | Backend-selection implication |
|---|---|---|
| Generation | [`decode.hpp`](../include/sub0/decode.hpp#L44-L60): `caps`, `init`, `set_tf32`, parameter upload, shutdown. Lines 124-160 reset KV state and call `forward_one` for prompt and generated tokens. [`gen_stage.cpp`](../src/gen_stage.cpp#L277-L297) creates this session in production `sub0_gen_stage`. | Convert the CUDA existence check at decode line 45 and target link at root CMake lines 420-423 to the generic selected-device path. Rename `gpu_*`/`use_gpu` wording when the implementation lands; capability and return codes decide fallback. |
| Generation callers outside `sub0_gen_stage` | [`train_stage.cpp`](../src/train_stage.cpp#L940-L948) uses a decode session for the final training preview; additional sessions are at lines 3678 (autotemp statistics), 3796 (autotemp sample), and 4121 (report samples). | These live in `sub0_train`, which must not link the first Intel DLL. They continue to see the no-device stubs unless CUDA is built. This prevents Intel from entering train/diagnostic paths accidentally. |
| Evaluation | [`eval.hpp`](../include/sub0/eval.hpp#L130-L147) calls `forward_loss`; lines 166-190 query `supports_eval`, initialize, upload parameters, and shut down. | First Intel reports eval false, so the CPU route remains. The test-only mock continues to prove capability-based fallback independently of CUDA/Intel. |
| Training | [`train_stage.cpp`](../src/train_stage.cpp#L1001-L1083) queries train/optimizer capabilities, initializes, uploads params and moments, reserves scratch, and queries free memory. Lines 1095-1189 call train step, backward, downloads, and shutdown. | CUDA-only compile guards remain. Do not replace them with `SUB0_BUILD_DEVICE`; doing so would compile the Intel selection into training. |
| Hybrid training | [`train_stage.cpp`](../src/train_stage.cpp#L1909-L1933) caches binding-compose capability. Lines 2118-2124 describe the POD binding arrays; calls at 2578 and 2607 install/clear them, and line 2649 re-uploads host-updated parameters. | CUDA training behavior; unavailable to first Intel build. |
| Tune/memory diagnostic | [`train_stage.cpp`](../src/train_stage.cpp#L3328-L3507) initializes the CUDA training backend, checks footprint, shuts down/restarts, sets TF32, reads free memory, and times device train steps. | Keep behind `SUB0_BUILD_CUDA` and `HAS_CUDA`. It is not an Intel inference tuner. |
| Report diagnostic | [`train_stage.cpp`](../src/train_stage.cpp#L3977-L3994) uses `eval::Session` and reads the capability name for provenance; lines 4111-4127 run the decode sample battery. | In the first Intel build this binary is not linked to Intel. A later report consumer needs an explicit scope decision and real eval/decode support. |
| Mock seam test | [`eval_seam_tests.cpp`](../tests/eval_seam_tests.cpp#L120-L130) directly checks `forward_loss`; line 208 resets via shutdown. [`mock_device_backend.cpp`](../tests/mock_device_backend.cpp#L1-L18) implements the mock exports. | Preserve the separate executable and `SUB0_BUILD_MOCK_DEVICE` priority at [`tests/CMakeLists.txt`](../tests/CMakeLists.txt#L189-L212); it is the CPU-only capability-routing gate. |
| CUDA diagnostics | [`cuda_tests.cpp`](../tests/cuda_tests.cpp#L960-L980) and lines 2275-2294 read neutral caps, while the rest of that file and [`cuda_selftest.cpp`](../tools/cuda_selftest.cpp#L27-L33) call backend-specific `sub0_cuda_*` diagnostics. | Diagnostic symbols remain vendor-specific. Each Intel kernel package supplies its own parity hooks/tests; they do not enter the neutral production ABI. |

The standalone [`sub0llm-gen.cpp`](../tools/sub0llm-gen.cpp#L1-L4) calls `run_gen`, which crosses the C
stage ABI declared at [`cli_stages.hpp`](../include/sub0/cli_stages.hpp#L20-L39) into `sub0_gen_stage`.
Root CMake lines 412-423 build that stage DLL; lines 441-450 link it into `sub0llm-gen`; lines 425-433
also link `sub0_gen` into the umbrella `sub0llm`. Those are the real production load paths. By contrast,
`sub0llm-qwen4-gen` links `sub0_core` directly at
[`CMakeLists.txt`](../CMakeLists.txt#L468-L506) and never exercises the stage/device seam.

## Current CUDA link inventory

The complete CUDA-specific link graph on current `main` is:

```text
CUDA::cudart + CUDA::cublas
             -> sub0_backend_cuda.dll
                -> sub0-cuda-selftest.exe              diagnostic
                -> sub0_train.dll -> sub0llm{-train,-tune}.exe / sub0llm.exe
                -> sub0_gen.dll   -> sub0llm-gen.exe / sub0llm.exe
                -> sub0_tests.exe                         diagnostic
```

The defining lines are root CMake 340-368, 383-423, 425-450 and tests CMake 214-252. No other CMake
target links `CUDA::`, `cudart`, `cublas`, or `sub0_backend_cuda`. The Qwen forward/generation harnesses
link only `sub0_core` (and frontend/CLI/`psapi` as applicable), so they remain CPU oracles rather than
production device consumers.

## Locally observable Windows toolchain candidate

These are facts observed on 2026-09-09, not a passed cross-DLL gate. Archived manifests preserve their
own older commit bases; this table does not rewrite that provenance.

| Layer | Observed candidate/evidence | Status for I07b.0 |
|---|---|---|
| Host compiler | `C:\Program Files\LLVM\bin\clang++.exe`, clang 22.1.6, target `x86_64-pc-windows-msvc`. The existing `d196check` CMake cache selects it and `lld-link.exe`; its `sub0_gen` compile command uses `-std=gnu++26`, `-D_DLL -D_MT`, and `--dependent-lib=msvcrt`. | Candidate pinned; not rebuilt from this worktree by this audit. |
| Generator | CMake 4.2.3 with Ninja 1.13.2. [`CMakePresets.json`](../CMakePresets.json#L5-L15) and [`workflow.ps1`](../scripts/workflow.ps1#L134-L142) select Ninja plus clang/clang++. | Candidate pinned. |
| Host STL | Visual Studio 18 Community, VC tools `14.51.36231`; preprocessor reports `_MSVC_STL_VERSION=145`, `_MSVC_STL_UPDATE=202604L`, `_MSC_VER=1951`. `yvals_core.h` SHA-256 `10B59D42BDE2105E76CF16735E34DD0F1D45A943CB1B789C4A0406C67C2BD684`. | Candidate pinned. No STL type may cross the Intel seam. |
| Host CRT | Windows SDK/UCRT `10.0.26100.0`; `corecrt.h` SHA-256 `822E503B81DD7B3D7DF93CA22FCED3672A5154484FD42054D5941E619BCF6CBC`. `sub0_gen.dll` imports `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, and API-set UCRT DLLs. | Dynamic CRT candidate pinned. |
| DPC++ compiler | `C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\icx-cl.exe`, Intel oneAPI DPC++/C++ 2025.3.3 (`2025.3.3.20260319`), target `x86_64-pc-windows-msvc`, SHA-256 `B56A6CAE493169514D2FD14B6781A5DD9EBBEDA2FA0524D72CAA9594EB67E7A0`. [`compile-gates.json`](intel-groundwork/2026-09-09/compile-gates.json) records successful C++20 SYCL compile/link. | Compiler pinned for the smoke candidate. |
| DPC++ STL/CRT | The runners import the same VS 18 `vcvars64.bat`; the compiled probe imports `MSVCP140.dll`, `VCRUNTIME140.dll`, and UCRT API sets. It also imports `sycl8.dll`. | Same MSVC ABI/dynamic CRT family is observed, but allocation/exception safety remains unproved until the smoke gate. |
| SYCL runtime | `sycl8.dll` version `2025.3.0.0`, SHA-256 `C9CDCC84C80D0B6904D4E5429E3B3DE43201BE7425251156D6D67408F00535B8`; `ur_loader.dll` version `0.12.0`, SHA-256 `B5D895938F78EBD9AA1250FDB595678C0193A3B9F7BDD0B157C61099D31008CE`. | Files pinned; full clean-shell redistribution closure is I23 work. |
| oneDNN | Installed package root `C:\Program Files (x86)\Intel\oneAPI\dnnl\2025.3`; header macros and executed controls report 3.9.1. `dnnl.dll` version `3.9.1 (78e781f6...)`, SHA-256 `544133A6BD576CEF441A341650BFEE16C5C452AC9E0CDFFB42A7D02515EA09D9`; `dnnl.lib` SHA-256 `40878A2512B7B4E2BED49F601758858AA663BC8FC8B3237D955C598AA4446331`. | Binary package pinned for the SYCL smoke candidate; exact upstream source/tag and redistribution remain S0a/I23. |
| Level Zero runtime | `C:\Windows\System32\ze_loader.dll` exists, version 1.32.0, SHA-256 `F5CB847750766CAEEF0ED02A1027F3661B04EA96C4911CD492F852CB8F9E640B`. Intel's installed `ur_adapter_level_zero.dll` is 2025.3.0.0, SHA-256 `65A8C022CCABE908455B283C2F7DF7646B8111CE7D0B0E88B53810D3F688F42E`. | Presence on disk is not proof of the loaded path/API. No `ze_api.h` or `ze_loader.lib` exists under the searched oneAPI root, and none was supplied to the compile gate. Native ZE remains unavailable/unpinned for this phase. |
| Generated header dialect | [`CMakeLists.txt`](../CMakeLists.txt#L19-L43) builds host C++ as C++26 but requires every header reachable from CUDA to remain C++20-compatible. Generated `sub0_config.hpp` is a small umbrella over constexpr-only `sub0_corpus.hpp` and `sub0_system.hpp`; DPC++ probes currently compile `/std:c++20 /EHsc -fsycl`. | Smoke DLL must compile the exact generated headers as C++20; successful standalone probes do not prove that compilation yet. |
| Export convention | Host `SUB0_API` is `__declspec(dllexport)` on Windows at [`core.hpp`](../include/sub0/core.hpp#L28-L34). CUDA uses `extern "C" __declspec(dllexport)` at [`backend.cu`](../src/backends/cuda/backend.cu#L38-L45). | Intel must use the same unmangled C/POD convention, with an import/export define rather than exporting inline bridges. |
| Development/runtime search path | Intel probe runners import VS `vcvars64`, prepend `compiler\2025.3\bin` and `compiler\2025.3\bin\compiler`, add `compiler\2025.3\lib`, and add `dnnl\2025.3\bin` when oneDNN is used; see [`run-groundwork.ps1`](../scripts/intel/inventory/run-groundwork.ps1#L17-L48). Production DLLs currently co-locate beside executables by target output; Windows has no RPATH. | Smoke manifest must record the exact effective `PATH` entries and loaded module paths. I23 later proves a clean-shell redistributable layout. |

## Required POD smoke-DLL gate through production generation

The remaining I07b.0 proof must use `sub0llm-gen -> sub0_gen_stage -> DecodeSession -> sub0_dev_*`, not
a standalone probe and not `sub0llm-qwen4-gen`.

This work is deliberately split. **I07b.1** proves generated-header ordering, the wire ABI and one
linked deterministic call through production generation. **I07b.2/I23** proves friendly dynamic-load
diagnostics, clean-shell dependency closure, redistribution and unload behavior. I07b.1 may use the
ordinary eager import; its missing-DLL control must exit nonzero and must never fall back to CPU, but a
pre-`main` Windows loader message is not represented as a backend diagnostic success.

1. Build the ordinary host/configurator with clang C++26 and the Intel backend child with the pinned
   DPC++ C++20 compiler. Build the Intel child only after `sub0llm-configure` has produced the exact
   `sub0_config.hpp`; compile against that file and `layout.hpp`, then return ABI version/struct size,
   `MODEL_ARCH_ID`, `PARAM_FLOATS`, `VOCAB`, and `SEQ_LEN` through fixed-width POD fields.
2. Export native unmangled `sub0_dev_*` entry points from one `sub0_backend_intel.dll`. The host passes
   only fixed-width scalars, borrowed pointers plus explicit element counts, and caller-owned output
   buffers. No `std::string`, `std::vector`, exception object, `sycl::queue`, oneDNN object, or allocator
   ownership crosses the boundary.
3. Use a smoke implementation that truthfully implements a deterministic decode-shaped operation:
   `init`, parameter-upload validation/copy, `kv_reset`, and `forward_one` enqueue a bounded kernel that
   fills the caller's `[VOCAB]` logits with finite deterministic values and waits before return. Run a
   generated tiny dense model through `sub0llm-gen --n 1` and assert the selected token/logit checksum,
   init/upload/reset/forward/shutdown call counts, and that the CPU fallback was not used. This tests the
   real production source at [`gen_stage.cpp`](../src/gen_stage.cpp#L277-L297).
4. Every exported function is non-throwing at the ABI. Catch synchronous exceptions and
   `wait_and_throw` failures inside the Intel DLL, translate them to stable integer status plus a bounded
   POD error record, and prove an injected init and queued-kernel exception returns a nonzero stage exit
   with the process still alive. No C++ exception may reach clang-built code.
5. Host allocations are freed by the host; Intel-DLL allocations are freed by the Intel DLL. Upload
   either completes a copy before returning or stores only DLL-owned storage. Poison/check the host input
   after upload, exercise repeated init/shutdown, and run the test under the available Windows heap
   verifier so cross-CRT frees, leaks, and use-after-free fail the gate.
6. `sub0_dev_shutdown` is idempotent and does not return until all submitted work has completed or a
   recorded unrecoverable failure has transferred ownership to process teardown. It releases queues,
   events, oneDNN objects, and USM in dependency order. No backend unload/reload is allowed while work or
   a borrowed host output remains live.
7. I07b.1 proves that a missing backend DLL or transitive runtime DLL exits nonzero and cannot silently
   generate on CPU. I07b.2/I23 owns a clear stage-level diagnostic. Because normal Windows import loading
   can fail before `main`, that later work may use a compile-time-fixed loader shim or a tested delay-load
   handler for the single selected DLL. It must not grow into a runtime backend registry.

The smoke backend is test evidence, not a capability claim for Qwen. The first real Intel inference DLL
reports the following values only after the corresponding I11 entry points are implemented and fixture
gated:

```text
name                         = "intel"
supports_train               = 0
supports_eval                = 0
supports_decode              = 1
supports_interception        = 0
supports_binding_compose     = 0
supports_opt_state           = 0
supports_tf32                = 0
```

Before real decode exists, `supports_decode` is also `0`; a smoke-only deterministic logits provider must
identify itself as `intel-smoke` and cannot ship as the production selection. `supports_eval` remains
false even if decode works: the current eval contract is batched forward-loss, a separate operation.

## Acceptance commands for the implementation phase

Target names below are the contract this record proposes. They become runnable only when I07b/I11 adds
the real targets and selector.

```powershell
# Consumer census: only the documented implementation, production, mock, and diagnostic sites remain.
git grep -n -E 'HAS_CUDA|SUB0_BUILD_CUDA|SUB0_BUILD_DEVICE|ComputeBackend|sub0_dev_[A-Za-z0-9_]+' -- `
  '*.cpp' '*.hpp' '*.cu' '*.cmake' 'CMakeLists.txt' '*.in'

# CPU default: no Intel compiler/library discovery or dependency.
cmake -S . -B out/build/i07b0-cpu -G Ninja -DSUB0_COMPUTE=CPU -DSUB0_DEVICE=NONE
cmake --build out/build/i07b0-cpu --target sub0llm-configure sub0_tests sub0_frontend_tests sub0_eval_seam_tests
ctest --test-dir out/build/i07b0-cpu --output-on-failure

# Intel smoke selection: host remains CPU; only generation receives the selected device seam.
cmake -S . -B out/build/i07b0-intel-smoke -G Ninja -DSUB0_COMPUTE=CPU `
  -DSUB0_DEVICE=INTEL -DSUB0_INTEL_ABI_SMOKE=ON
cmake --build out/build/i07b0-intel-smoke --target sub0llm-configure
out/build/i07b0-intel-smoke/sub0llm-configure.exe --corpus data/english.txt `
  --dmodel 96 --layers 2 --heads 2 --kv-heads 2
cmake --build out/build/i07b0-intel-smoke --target sub0llm-train sub0llm-gen sub0_intel_abi_smoke_tests
out/build/i07b0-intel-smoke/sub0llm-train.exe --corpus data/english.txt `
  --model out/build/i07b0-intel-smoke/smoke-model.bin --steps 1 --batch 1 --fresh
out/build/i07b0-intel-smoke/sub0llm-gen.exe --model out/build/i07b0-intel-smoke/smoke-model.bin `
  --n 1 --temp 0 --topk 1 --seed 1 "smoke"
ctest --test-dir out/build/i07b0-intel-smoke -R 'intel_abi_smoke' --output-on-failure

# Symbol/import proof. Neutral production exports must be unmangled; no sub0_cuda_* import enters Intel gen.
llvm-readobj --coff-exports out/build/i07b0-intel-smoke/bin/sub0_backend_intel.dll
llvm-readobj --coff-imports out/build/i07b0-intel-smoke/bin/sub0_gen.dll
```

The I07b.1 test suite must additionally execute injected synchronous/async exceptions, ABI-size/version
and config mismatches, missing backend/transitive DLLs, and repeated init/shutdown. I07b.2/I23 adds
unload-after-drain and friendly load-failure diagnostics. The default CPU assertion counts and the full
unfiltered CUDA suite must match their pre-change baselines; the Intel selection must leave `sub0_train`
without an Intel import and with `HAS_CUDA=false` and `COMPUTE_MODE=Cpu` in its generated header.

## Unknowns and stop conditions

- The host/DPC++ exception and allocation contract is unproved until the production-stage smoke
  above passes. Matching MSVC ABI/CRT imports is necessary evidence, not proof.
- The current superbuild has only the host child
  ([`SuperBuild.cmake`](../cmake/SuperBuild.cmake#L51-L70)); generated-header ordering, DPC++ child
  isolation, import-library handoff, and shared output staging remain implementation work.
- Direct Level Zero development headers/import library and the actually loaded loader/API version are
  not pinned. Stop native-ZE work; continue only the already qualified SYCL control until S0a supplies a
  compatible approved tuple.
- The complete clean-shell DLL dependency/redistribution/license closure is not established. Do not call
  the candidate package deployable before I23. Friendly missing-DLL diagnostics and backend unload are
  I07b.2/I23 gates; the first eager-import smoke proves only nonzero failure without CPU fallback.
- The legacy f32 whole-arena upload cannot admit the real encoded Qwen artifact. I07b/I11 must add the
  versioned, size-tagged preparation metadata and reject unsupported architecture, packing, operation,
  state, or context combinations before allocation/execution.
- Stop production interface work at any ABI/config mismatch, cross-boundary exception, cross-allocator
  ownership, ambiguous DLL load, unload with queued work, or Intel import in `sub0_train`. Record the
  failure rather than falling through to a success claim.

This audit used local source/file inspection only. No compiler, device, or inference workload was run.
