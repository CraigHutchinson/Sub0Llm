<#
.SYNOPSIS
  End-to-end configure -> build -> (tune) -> train -> gen helper for a single corpus.

.DESCRIPTION
  Wraps the manual sequence in docs/WORKFLOW_ARCHITECTURE.md's "Target workflow" section into one
  command: create/reuse a CMake build dir, build the sub0llm-configure EXECUTABLE, run it directly
  (`sub0llm-configure.exe --corpus <c>`, not a CMake target -- there is no CMake-orchestrated config
  step; the build never regenerates config behind your back), build the stage tools (not the test
  suites), optionally tune, train, then run gen once as a sanity check.

  This is a convenience wrapper, not a new code path: every step just shells out to the same
  cmake / sub0llm-* invocations you'd type by hand, kept here so the sequence + the CUDA toolchain
  setup (see Import-VcVars below) don't have to be re-derived each time.

.PARAMETER BuildDir
  Subdirectory under out/build/. Reused across runs (idempotent reconfigure) unless -Clean is passed.

.PARAMETER Corpus
  Path to the training corpus (repo-relative or absolute). Passed to sub0llm-configure's --corpus,
  never to CMake -- the corpus is a tool argument, not a build-config knob.

.PARAMETER Compute
  AUTO / CPU / GPU / HYBRID -- see cmake/Backends.cmake. AUTO (default) uses the GPU if one is found.

.PARAMETER Steps
  Training steps; 0 (default) lets sub0llm-train auto-size to the corpus and stop on plateau.
  (An explicit -Steps N always runs the full N -- plateau-stop is auto-mode only.)

.PARAMETER OpMix
  0 (default) = off. >0 blends the op-delegation curriculum (teach the model to ROUTE arithmetic to the
  exact `math` node -- `[op math]`, which gen dispatches -- instead of computing it, including multi-step
  collapse chains). Combine with -Corpus data/gsm8k.txt for the GSM8K math workflow. See
  docs/DETERMINISTIC_MECHANISMS.md. (The proven arch stack -- gated FFN / tied embeddings / QK-norm / Muon --
  is now configure/train default-ON, so no extra flag is needed for it.)

.PARAMETER Tune
  Run sub0llm-tune before training and rebuild so the baked DEFAULT_THREADS picks up its result.
  Off by default: training runs fine on the un-tuned defaults, and tuning adds its own search time.

.PARAMETER SkipConfigure
  Skip the build+run of sub0llm-configure entirely and reuse whatever config header the build dir
  already has (e.g. a prior run against the same corpus). Useful for iterating on the stage tools /
  training loop without re-deriving the vocabulary each time, especially on a large corpus where that
  can take minutes. The generated header must already exist -- the stage tools fail to compile if not.

.EXAMPLE
  scripts/workflow.ps1 -BuildDir smoke -Corpus data/tinystories.txt -Steps 500
  Quick end-to-end smoke run: configure+build for tinystories, train 500 steps, generate a sample.

.EXAMPLE
  scripts/workflow.ps1 -BuildDir prod -Corpus data/fineweb_edu.txt -Tune
  Full run: tune first, then train to the auto-sized step/plateau budget.

.EXAMPLE
  python scripts/get_gsm8k.py                       # -> data/gsm8k.txt
  scripts/workflow.ps1 -BuildDir gsm8k -Corpus data/gsm8k.txt -OpMix 0.5 -Gsm8k data/gsm8k.txt -Steps 4000
  GSM8K math workflow: base corpus = GSM8K (reasoning fluency); -Gsm8k converts GSM8K's OWN <<expr=result>>
  annotations to delegated [op math] frames as the op source, blended at -OpMix 0.5 -- so the model routes
  GSM8K's arithmetic to the exact node instead of learning it. (Omit -Gsm8k for the synthetic curriculum.)
  Score with `out/build/gsm8k/sub0llm.exe eval <model>/model.bin`.
#>
[CmdletBinding()]
param(
    [string]$BuildDir  = "workflow",
    [string]$Corpus    = "data/tinystories.txt",
    [ValidateSet("AUTO", "CPU", "GPU", "HYBRID")]
    [string]$Compute   = "AUTO",
    [int]$Steps        = 0,
    [double]$OpMix     = 0,
    [string]$Gsm8k     = "",
    [switch]$Tune,
    [switch]$Clean,
    [switch]$SkipConfigure,
    [string]$ModelPath = "",
    [string]$Prompt    = "Once upon a time",
    [int]$GenTokens    = 200,
    [double]$Temp      = 0.8,
    [int]$TopK         = 40
)

$ErrorActionPreference = "Stop"
$RepoRoot   = Split-Path -Parent $PSScriptRoot
$CorpusPath = (Resolve-Path (Join-Path $RepoRoot $Corpus)).Path
$Bin        = Join-Path $RepoRoot "out\build\$BuildDir"

if (-not $ModelPath) {
    # An explicit path (not sub0llm-train's own auto-naming) so this script always knows where to
    # point gen afterward without having to parse train's stdout for its final "-> <path>" line.
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($CorpusPath)
    $ModelPath = Join-Path $RepoRoot "models\${stem}_$BuildDir\model.bin"
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$ExeArgs)
    Write-Host ">> $Exe $($ExeArgs -join ' ')" -ForegroundColor Cyan
    & $Exe @ExeArgs
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited with code $LASTEXITCODE" }
}

function Import-VcVars {
    # nvcc needs cl.exe on PATH to compile backend_cuda.cu even though CMake's own CUDA detection
    # (find_package(CUDAToolkit), used for the AUTO compute resolution) does not require it -- so a
    # plain PowerShell session builds fine right up until the first .cu file, then fails opaquely
    # ("nvcc fatal: Cannot find compiler 'cl.exe'"). Importing vcvars64's environment up front avoids
    # that. Harmless to skip/no-op for a CPU-only build.
    $vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $vcvars) {
        Write-Warning "vcvars64.bat not found -- a GPU/AUTO build will fail at the CUDA compile step. Install the VS Build Tools, or pass -Compute CPU."
        return
    }
    (cmd /c "`"$($vcvars.FullName)`" && set") | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
    }
}

if ($Clean -and (Test-Path $Bin)) {
    Write-Host "=== removing $Bin ===" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $Bin
}

Import-VcVars

Write-Host "=== configure: $BuildDir <- $CorpusPath (compute=$Compute) ===" -ForegroundColor Yellow
Invoke-Checked cmake @(
    "-S", $RepoRoot, "-B", $Bin, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release", "-DSUB0_NATIVE=ON",
    "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++",
    "-DSUB0_COMPUTE=$Compute"
)

if (-not $SkipConfigure) {
  Write-Host "=== build: configurator ===" -ForegroundColor Yellow
  Invoke-Checked cmake @("--build", $Bin, "--target", "sub0llm-configure")

  Write-Host "=== configure: deriving vocab from $CorpusPath (this can take a while on a large corpus) ===" -ForegroundColor Yellow
  Invoke-Checked (Join-Path $Bin "sub0llm-configure.exe") @("--corpus", $CorpusPath)
}

Write-Host "=== build: stage tools ===" -ForegroundColor Yellow
Invoke-Checked cmake @("--build", $Bin, "--target", "sub0llm-train", "sub0llm-gen", "sub0llm-tune")

if ($Tune) {
    Write-Host "=== tune ===" -ForegroundColor Yellow
    Invoke-Checked (Join-Path $Bin "sub0llm-tune.exe") @()
    Invoke-Checked cmake @("--build", $Bin, "--target", "sub0llm-train", "sub0llm-gen")  # bake the tuned knobs
}

Write-Host "=== train -> $ModelPath ===" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path (Split-Path $ModelPath) | Out-Null
$trainArgs = @()
if ($Steps -gt 0) { $trainArgs += @("--steps", "$Steps") }
if ($OpMix -gt 0) { $trainArgs += @("--op-mix", "$OpMix") }   # blend the op-delegation (math routing) curriculum
if ($Gsm8k)       { $trainArgs += @("--gsm8k", (Resolve-Path (Join-Path $RepoRoot $Gsm8k)).Path) }  # real GSM8K as the op source
$trainArgs += @($ModelPath, $CorpusPath)
Invoke-Checked (Join-Path $Bin "sub0llm-train.exe") $trainArgs

Write-Host "=== gen sanity check ===" -ForegroundColor Yellow
Invoke-Checked (Join-Path $Bin "sub0llm-gen.exe") @($ModelPath, $Prompt, "--n", "$GenTokens", "--temp", "$Temp", "--topk", "$TopK")

Write-Host "workflow complete: $ModelPath" -ForegroundColor Green
