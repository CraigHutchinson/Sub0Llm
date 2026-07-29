<#
.SYNOPSIS
  LoopSplit A/B on Cosmopedia: does weight-shared looping recover the capacity of real layers?

.DESCRIPTION
  Three arms at MATCHED EXECUTIONS, which is what makes the result interpretable. LoopSplit moves
  compute and parameters in OPPOSITE directions, so a single comparison cannot be read cleanly:

    A  deep baseline   L=16, no loop      -> 16 executions, 16 layers of parameters
    B  looped          L=10, middle 6 x2  -> 16 executions, 10 layers of parameters
    C  shallow control L=10, no loop      -> 10 executions, 10 layers of parameters

  A vs B holds COMPUTE constant and cuts parameters by ~37%. If B ~= A, looping recovered the
  capacity of six real layers for free -- the actual claim. C exists because without it a B-vs-A
  result is unattributable: it separates "the loop did it" from "10 layers was already enough".

  The prior TinyStories A/B measured only the parameter-matched form (same L, loop on/off), where
  the loop arm simply spends more compute. Its negative therefore said "extra compute did not help
  on a corpus with a 1.5k vocabulary", which is not evidence about parameter efficiency.

  Deliberate choices, each fixing a specific confound found while planning this:
    * Cosmopedia, not TinyStories -- looping refines ABSTRACTIONS, and TinyStories has none to
      refine (see the scope caveat on the earlier negative result).
    * vocab 16384, not Cosmopedia's natural ~49k -- at 49k a small model is >50% embedding TABLE,
      and looping repeats the BODY. Testing a depth mechanism on a model that is mostly lookup
      would produce another uninterpretable negative.
    * SUB0_FIXED_SEQ=1 -- the default varies window width uniformly over [SEQ_LEN/8, SEQ_LEN], so
      position 500 of 512 is trained on 2.7% of steps and position 511 on 0.2%. A mechanism for
      building abstractions ACROSS context cannot be tested on a model whose far context is
      untrained.
    * Runs are STRICTLY SEQUENTIAL. Concurrent trainers share one GPU and halve each other's
      throughput, making every performance number meaningless. sub0llm-train enforces this itself;
      this script never passes --allow-concurrent.

.PARAMETER Probe
  Measure throughput only (60 steps per arm), print the projected epoch/run/total time, and STOP.
  Run this first. The full sweep is many GPU-hours and the projection is what tells you whether the
  budget below is affordable before you commit to it.

.PARAMETER Seeds
  Seeds per arm. GQA's own A/B found seed noise 1.73x the signal with rankings FLIPPING between
  seeds, so a 1-seed result here is not evidence. 3 is the minimum that can show a consistent
  ordering; use 1 only for a first-look smoke.

.PARAMETER CorpusFraction
  Fraction of Cosmopedia per epoch. 1.0 is the full corpus. Lower it to fit the time budget the
  probe reports -- the subset is distributed and document-aligned, and NESTED across fractions, so
  0.25 is a strict subset of 0.5.

.EXAMPLE
  pwsh scripts/loopsplit_ab.ps1 -Probe
  pwsh scripts/loopsplit_ab.ps1 -Seeds 3 -CorpusFraction 0.5
#>
param(
    [switch] $Probe,
    [int]    $Seeds = 3,
    [double] $CorpusFraction = 1.0,
    [string] $Corpus = "data/cosmopedia.txt",
    [int]    $Vocab = 16384,
    [int]    $DModel = 192,
    [int]    $Heads = 6,
    [int]    $SeqLen = 512,
    # Pinned identically across every arm and seed -- see the call site for why leaving these to the
    # per-arm auto tuner silently broke the first sweep's token matching. The defaults are what the
    # most expensive arm (16 executions) fits on an 8 GB card.
    [int]    $Batch = 448,
    [double] $Lr = 0.00748332,
    [int]    $Steps = 8586,
    [string] $Clang = "C:/Program Files/LLVM/bin/clang++.exe",
    [string] $ClangC = "C:/Program Files/LLVM/bin/clang.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# name, N_LAYERS, LOOP_MIDDLE_LAYERS, LOOP_REPEATS
$Arms = @(
    @{ Name = "A_deep16";   Layers = 16; Middle = 0; Repeats = 1 },
    @{ Name = "B_loop10x6"; Layers = 10; Middle = 6; Repeats = 2 },
    @{ Name = "C_shallow10";Layers = 10; Middle = 0; Repeats = 1 }
)

function Write-Head($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

# --- pre-flight -------------------------------------------------------------------------------
Write-Head "pre-flight"
if (-not (Test-Path $Corpus)) { throw "corpus not found: $Corpus (run scripts/get_cosmopedia.py)" }
$bytes = (Get-Item $Corpus).Length
Write-Host ("corpus: {0} ({1:N2} GB)" -f $Corpus, ($bytes / 1GB))

# The EOT convention is mandatory: the document scan is <|endoftext|>-only (the blank-line fallback
# was removed as unsound), so a corpus without markers is rejected at configure time. Check the head
# rather than the whole file -- this is a cheap sanity check, not a validation pass.
# Stream the first 4MB. ReadAllBytes() would pull the WHOLE corpus into memory and throws outright
# past 2GB -- a check that cannot run on the files it is meant to guard is worse than no check.
$fs   = [System.IO.File]::OpenRead((Resolve-Path $Corpus))
try {
    $n    = [int][Math]::Min([long]4000000, [long]$bytes)
    $head = [byte[]]::new($n)
    $null = $fs.Read($head, 0, $n)
} finally { $fs.Dispose() }
if (-not ([System.Text.Encoding]::UTF8.GetString($head).Contains("<|endoftext|>"))) {
    throw "no <|endoftext|> in the first 4MB of $Corpus -- wrong extraction; see scripts/get_cosmopedia.py"
}
Write-Host "<|endoftext|> markers: present"

foreach ($a in $Arms) {
    # layout.hpp: static_assert((N_LAYERS - LOOP_MIDDLE_LAYERS) % 2 == 0) -- symmetric head/tail.
    # Checked HERE so a bad arm fails in a second, not 20 minutes into a CUDA build.
    if (($a.Layers - $a.Middle) % 2 -ne 0) {
        throw "$($a.Name): (layers - middle) must be even for a symmetric head/tail; got $($a.Layers)-$($a.Middle)"
    }
    $a.Exec = $a.Layers + $a.Middle * ($a.Repeats - 1)
}
$execs = ($Arms | ForEach-Object { $_.Exec }) | Select-Object -Unique
Write-Host ("executions per arm: " + (($Arms | ForEach-Object { "$($_.Name)=$($_.Exec)" }) -join "  "))
if ($execs.Count -eq $Arms.Count) {
    Write-Warning "no two arms share an execution count -- A vs B is only interpretable when they MATCH"
}

if (Get-Process sub0llm-train -ErrorAction SilentlyContinue) {
    throw "a sub0llm-train is already running -- stop it first (concurrent runs invalidate every timing)"
}
$free = (Get-PSDrive ($Root[0])).Free
Write-Host ("free disk: {0:N0} GB" -f ($free / 1GB))
if ($free -lt 60GB) { Write-Warning "under 60GB free; each arm keeps checkpoints (~250MB each)" }

# --- build + configure each arm --------------------------------------------------------------
foreach ($a in $Arms) {
    $bin = "out/build/ls_$($a.Name)"
    Write-Head "arm $($a.Name): configure ($($a.Layers) layers, middle $($a.Middle) x$($a.Repeats), $($a.Exec) executions)"
    if (-not (Test-Path "$bin/build.ninja")) {
        # CMAKE_BUILD_TYPE must be EXPLICIT: without it CMP0141 selects MSVC's ProgramDatabase debug
        # format, which clang rejects ("value 'ProgramDatabase' not known for this CXX compiler").
        # Existing build dirs have it cached and never hit this; a fresh one does.
        cmake -S . -B $bin -G Ninja -DCMAKE_BUILD_TYPE=Release `
              -DCMAKE_CXX_COMPILER="$Clang" -DCMAKE_C_COMPILER="$ClangC" `
              -DSUB0_COMPUTE=GPU -DSUB0_NATIVE=ON -DSUB0_BUILD_TESTS=OFF
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $($a.Name)" }
    }
    cmake --build $bin --target sub0llm-configure -j 8
    if ($LASTEXITCODE -ne 0) { throw "configure-tool build failed for $($a.Name)" }

    & "$bin/sub0llm-configure.exe" --corpus $Corpus --vocab $Vocab --dmodel $DModel `
        --layers $a.Layers --heads $Heads --seq $SeqLen `
        --loop-middle-layers $a.Middle --loop-repeats $a.Repeats
    if ($LASTEXITCODE -ne 0) { throw "sub0llm-configure failed for $($a.Name)" }

    cmake --build $bin -j 8
    if ($LASTEXITCODE -ne 0) { throw "engine build failed for $($a.Name)" }
}

# --- probe: measure, do not guess -------------------------------------------------------------
# Every projection below comes from a real 60-step run of THIS build on THIS corpus. Estimating from
# an older config was wrong by 32% once already this project (a d256/seq256/vocab16k baseline does
# not transfer to seq512/vocab49k), so the sweep is never sized from anything but a measurement.
$env:SUB0_FIXED_SEQ = "1"
Write-Head "throughput probe (SUB0_FIXED_SEQ=1)"
$proj = @()
foreach ($a in $Arms) {
    $bin = "out/build/ls_$($a.Name)"
    Remove-Item -Recurse -Force "models/_probe_$($a.Name)" -ErrorAction SilentlyContinue
    # Say what is happening. Each probe is minutes of silence otherwise -- CUDA init, mmap of a
    # multi-GB corpus.tok, 60 steps, then a model save -- and a bare header with no output for ten
    # minutes reads as a hang, not as work in progress.
    Write-Host ("  probing {0} ({1} executions) -- CUDA init + corpus mmap + 60 steps, ~2-5 min..." -f `
        $a.Name, $a.Exec) -ForegroundColor DarkGray
    & "$bin/sub0llm-train.exe" --steps 60 --fresh --model "models/_probe_$($a.Name)/model.bin" `
        --corpus-fraction $CorpusFraction --subset-seed 1 2>&1 | Out-Null
    # Take the ENGINE's own steady-state numbers, never (60 / wall-clock). A 60-step probe is
    # dominated by fixed startup -- CUDA init, mmap of a multi-GB corpus.tok, model init, final save --
    # so wall-clock/steps over-reports epoch time by ~3.6x (measured: 13.84h projected vs 6.8h real).
    # The log's tok/s differences tokens_seen and its "next ep in Xh" derives from that, so both are
    # steady-state by construction.
    $log   = Get-ChildItem "models/_probe_$($a.Name)/train.log" -ErrorAction SilentlyContinue
    $sched = 0; $toks = 0; $eph = 0
    if ($log) {
        $txt = Get-Content $log.FullName -Raw
        if ($txt -match 'schedule: (\d+) steps/epoch')  { $sched = [int]$Matches[1] }
        # last occurrence wins: later lines are warmer and more representative
        $m = [regex]::Matches($txt, '\((\d+) tok/s\)');       if ($m.Count) { $toks = [int]$m[$m.Count-1].Groups[1].Value }
        $m = [regex]::Matches($txt, 'next ep in ([0-9.]+)h');  if ($m.Count) { $eph  = [double]$m[$m.Count-1].Groups[1].Value }
    }
    $proj += [pscustomobject]@{ Arm = $a.Name; StepsPerEpoch = $sched; TokensPerSec = $toks; HoursPerEpoch = [math]::Round($eph,2) }
    Write-Host ("    -> {0:N0} tok/s, {1} steps/epoch, {2}h/epoch" -f $toks, $sched, [math]::Round($eph,2)) -ForegroundColor DarkGray
    Remove-Item -Recurse -Force "models/_probe_$($a.Name)" -ErrorAction SilentlyContinue
}
$proj | Format-Table -AutoSize
$worst = ($proj | Measure-Object HoursPerEpoch -Maximum).Maximum
# 3 epochs is the planning figure, not a promise: runs are plateau-stopped and may finish sooner.
Write-Host ("projected: {0} arms x {1} seeds x ~3 epochs x <={2}h = ~{3:N1} GPU-hours" -f `
    $Arms.Count, $Seeds, $worst, ($Arms.Count * $Seeds * 3 * $worst)) -ForegroundColor Yellow

if ($Probe) {
    Write-Host "`n-Probe set: stopping before the sweep. Re-run without -Probe (and tune -CorpusFraction) to commit." -ForegroundColor Green
    exit 0
}

# --- the sweep --------------------------------------------------------------------------------
Write-Head "sweep: $($Arms.Count) arms x $Seeds seeds, sequential"
foreach ($seed in 1..$Seeds) {
    foreach ($a in $Arms) {
        $bin = "out/build/ls_$($a.Name)"
        $tag = "models/ls_$($a.Name)_s$seed"
        if (Test-Path "$tag/state.json") { Write-Host "skip $tag (already exists)"; continue }
        Write-Host "`n--- $($a.Name) seed $seed ---" -ForegroundColor Cyan
        # --seed varies init/shuffle; --subset-seed is PINNED so every arm and seed sees the SAME
        # documents. Varying both would confound architecture with data.
        # --batch/--lr/--steps are PINNED and identical across arms. Leaving them to the per-arm auto
        # tuner is what confounded the first sweep: it sizes batch from free VRAM, so the cheapest arm
        # (fewest executions) got batch 512 where the others got 448 -- and since tokens_per_step is
        # batch*SEQ_LEN, "matched steps" silently meant 14.3% MORE TOKENS for that arm, plus a
        # different lr. The batch must be what the MOST EXPENSIVE arm fits; the headroom the cheaper
        # arms give up is the price of a controlled comparison. See loopsplit-3arm-batch-confound.
        & "$bin/sub0llm-train.exe" --fresh --model "$tag/model.bin" `
            --batch $Batch --lr $Lr --steps $Steps `
            --seed $seed --subset-seed 1 --corpus-fraction $CorpusFraction
        if ($LASTEXITCODE -ne 0) { Write-Warning "$($a.Name) seed $seed exited $LASTEXITCODE" }
    }
}

Write-Head "results"
# Compare on val_nelbo at matched TOKENS, never at matched steps: the arms differ in compute per
# step, so equal steps is not equal work and equal wall-clock is not equal data.
Get-ChildItem "models/ls_*/state.json" | ForEach-Object {
    $s = Get-Content $_.FullName -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Run = $_.Directory.Name; Status = $s.status; Steps = $s.steps
        Epochs = [math]::Round($s.epochs,2); TokensSeen = $s.tokens_seen; BestValNelbo = $s.best_val_nelbo
    }
} | Sort-Object Run | Format-Table -AutoSize
Write-Host "`nReport all three pillars (correctness / throughput / memory), never quality alone." -ForegroundColor Yellow
