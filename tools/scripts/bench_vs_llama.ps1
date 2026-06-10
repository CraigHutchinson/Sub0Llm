#!/usr/bin/env pwsh
# bench_vs_llama.ps1
#
# Interleaved, same-session control baseline: sub0llm-gemma vs llama.cpp, run back-to-back so
# thermal/driver/clock state is shared. This is the ONLY valid way to cite a llama performance
# reference — never compare our fresh numbers against a llama number from a prior session.
#
# Two MODES (the engines, binaries and swept dimension differ; ALL logging / metric parsing /
# medians / smoke-check / CSV / thermal-balanced schedule are shared):
#   -Mode threads : CPU. Sweep thread count TMin..TMax. ours=build-native, llama=...b9334/cpu,
#                   both vary -t. The swept variable is the thread count.
#   -Mode hybrid  : GPU/CPU hybrid. ours=build-cuda-native (--mode hybrid --gpu-layers V),
#                   llama=...b9334/cuda (-ngl Ngl [-fa on] [-ctk/-ctv q8_0]). The swept variable is
#                   OUR gpu-layers (-GpuLayers list); llama runs at a fixed config as the reference.
#
# Schedule: cyclic-rotation Latin square over the swept variants × Samples passes — every variant
# occupies every time-slot once (first-order thermal drift cancels); ours/llama order alternates
# per pass to cancel within-slot bias.
#
# Apples-to-apples: identical prompt text to both; both greedy (--temp 0). TG is timed after full
# prefill (equivalent starting state). PP: ours is token-by-token (TTFT); llama is batched — an
# architectural difference shown in the PP column (until we add batched GPU prefill).
#
# Usage:
#   .\tools\scripts\bench_vs_llama.ps1 -Mode hybrid -GpuLayers 24 -Ngl 28          # control baseline
#   .\tools\scripts\bench_vs_llama.ps1 -Mode hybrid -GpuLayers 16,20,24 -Ctx 4096  # sweep ours' layers
#   .\tools\scripts\bench_vs_llama.ps1 -Mode threads -TMin 8 -TMax 20              # CPU thread sweep

param(
    [ValidateSet("threads","hybrid")]
    [string]$Mode    = "hybrid",
    [string]$Model   = "models/gemma-4-12b-it-Q8_0.gguf",
    [int]   $N       = 64,
    [int]   $Samples = 6,
    # threads mode
    [int]   $TMin    = 1,
    [int]   $TMax    = -1,         # -1 = auto-detect logical core count
    # hybrid mode
    [int[]] $GpuLayers = @(24),    # ours: first V layers on GPU (the swept variants)
    [int]   $Ngl       = 28,       # llama: -ngl (fixed reference)
    [int]   $Ctx       = 0,        # KV capacity (0 = engine default)
    [int]   $Threads   = 0,        # CPU threads for both (0 = auto)
    [switch]$KvQ8,                 # q8_0 KV on both (ours --q8-kv, llama -ctk/-ctv q8_0)
    [switch]$NoFa,                 # disable llama flash-attn (default -fa on)
    [string]$Prompt  = "The transformer architecture, introduced in the paper Attention Is All You Need, revolutionised natural language processing by replacing recurrent networks with self-attention mechanisms. Each token attends to every other token in the sequence, allowing the model to capture long-range dependencies in a single forward pass.",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── mode config: binaries, swept variants, column label, smoke strictness ──────
if ($Mode -eq "threads") {
    if ($TMax -le 0) { $TMax = [Environment]::ProcessorCount }
    $OursBin    = "./build-native/bin/sub0llm-gemma.exe"
    $LlamaDir   = "D:/tools/llamacpp/vendor/llama.cpp-prebuilt/b9334/cpu"
    $Variants   = [int[]]($TMin..$TMax)
    $VarLabel   = "t"
    $StrictSmoke = $true            # CPU: greedy must match exactly (abort otherwise)
} else {
    $cudaBin = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"
    if (Test-Path $cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }   # cudart for our binary
    $OursBin    = "./build-cuda-native/bin/sub0llm-gemma.exe"
    $LlamaDir   = "D:/tools/llamacpp/vendor/llama.cpp-prebuilt/b9334/cuda"
    $Variants   = [int[]]$GpuLayers
    $VarLabel   = "gl"
    $StrictSmoke = $false           # GPU kernels may round-diverge late — warn, don't abort
}
$LlamaCompletion = Join-Path $LlamaDir "llama-completion.exe"
$LlamaVersion    = "$(Split-Path (Split-Path $LlamaDir -Parent) -Leaf)-$(Split-Path $LlamaDir -Leaf)"
$K               = $Variants.Count
$TotalRounds     = $Samples * $K

# ── per-mode argument builders (given the swept variant value) ─────────────────
function OursArgs([int]$v, [int]$n) {
    $a = @('--model',$Model,'--text',$Prompt,'-n',"$n")
    if ($Mode -eq "threads") { $a += @('--mode','greedy','-t',"$v") }
    else {
        $a += @('--mode','hybrid','--gpu-layers',"$v")
        if ($Ctx -gt 0)     { $a += @('--ctx',"$Ctx") }
        if ($Threads -gt 0) { $a += @('-t',"$Threads") }
        if ($KvQ8)          { $a += '--q8-kv' }
    }
    return $a
}
function LlamaArgs([int]$v, [int]$n) {
    $a = @('-m',$Model,'--prompt',$Prompt,'-n',"$n",'-no-cnv','-s','42','--temp','0')
    if ($Mode -eq "threads") { $a += @('-t',"$v") }
    else {
        $a += @('-ngl',"$Ngl")
        if (-not $NoFa)     { $a += @('-fa','on') }
        if ($Ctx -gt 0)     { $a += @('-c',"$Ctx") }
        if ($Threads -gt 0) { $a += @('-t',"$Threads") }
        if ($KvQ8)          { $a += @('-ctk','q8_0','-ctv','q8_0') }
    }
    return $a
}

# ── shared metric / text helpers (identical stderr formats across modes) ───────
function Get-OursMetrics([int]$v) {
    if ($DryRun) { return @{ pp = 10.0 + $v*0.1; tg = [math]::Max(0.1, 1+$v*0.4-$v*$v*0.008); wallS = 20.0 } }
    $pp = $null; $tg = $null; $t0 = Get-Date
    try { foreach ($line in (& $OursBin @(OursArgs $v $N) 2>&1)) {
        if ($line -match 'prompt forward.*\(([0-9.]+) tok/s\)') { $pp = [double]$Matches[1] }
        if ($line -match 'generated.*\(([0-9.]+) tok/s\)')      { $tg = [double]$Matches[1] }
    } } catch { }
    return @{ pp = $pp; tg = $tg; wallS = [math]::Round(((Get-Date)-$t0).TotalSeconds,1) }
}
function Get-LlamaMetrics([int]$v) {
    if ($DryRun) { return @{ pp = 120.0 + $v; tg = [math]::Max(0.1, 1.1+$v*0.42-$v*$v*0.0085); wallS = 14.0 } }
    $pp = $null; $tg = $null; $t0 = Get-Date
    try { foreach ($line in (& $LlamaCompletion @(LlamaArgs $v $N) 2>&1)) {
        if ($line -match 'prompt eval time' -and $line -match '([0-9]+\.[0-9]+) tokens per second') { $pp = [double]$Matches[1] }
        if ($line -match 'common_perf_print:\s+eval time' -and $line -match '([0-9]+\.[0-9]+) tokens per second') { $tg = [double]$Matches[1] }
    } } catch { }
    return @{ pp = $pp; tg = $tg; wallS = [math]::Round(((Get-Date)-$t0).TotalSeconds,1) }
}
function Get-OursGenText([int]$v, [int]$n) {
    try { foreach ($line in (& $OursBin @(OursArgs $v $n) 2>$null)) {
        if ($line -match '^gen_text:\s*(.*)') { return $Matches[1].Trim() } } } catch { }
    return $null
}
function Get-LlamaGenText([int]$v, [int]$n) {
    try {
        $j = ((& $LlamaCompletion @(LlamaArgs $v $n) 2>$null) -join ' ').Trim()
        if ($j.StartsWith($Prompt)) { $j = $j.Substring($Prompt.Length).Trim() }
        return $j
    } catch { }
    return $null
}
function Get-Median([double[]]$vals) {
    if (-not $vals -or $vals.Count -eq 0) { return $null }
    $s = $vals | Sort-Object; $n = $s.Count
    if ($n % 2 -eq 1) { return $s[($n-1)/2] } else { return ($s[$n/2-1]+$s[$n/2])/2.0 }
}
function Fmt($v, [string]$f="F2") { if ($null -eq $v) { return "N/A" } return ([double]$v).ToString($f) }

# ── accumulators ────────────────────────────────────────────────────────────────
$oursTG=@{}; $oursPP=@{}; $oursWall=@{}; $llamaTG=@{}; $llamaPP=@{}; $llamaWall=@{}
foreach ($v in $Variants) {
    $oursTG[$v]=[Collections.Generic.List[double]]::new(); $oursPP[$v]=[Collections.Generic.List[double]]::new(); $oursWall[$v]=[Collections.Generic.List[double]]::new()
    $llamaTG[$v]=[Collections.Generic.List[double]]::new(); $llamaPP[$v]=[Collections.Generic.List[double]]::new(); $llamaWall[$v]=[Collections.Generic.List[double]]::new()
}

# ── prompt token length (no model load in threads mode; quick tokenize) ────────
$PromptLen = if ($DryRun) { 42 } else {
    try { @((& $OursBin --model $Model --mode tokenize --text $Prompt 2>$null) -split '\s+' | Where-Object { $_ -match '^\d+$' }).Count } catch { 0 }
}

# ── header ────────────────────────────────────────────────────────────────────
$gitHash   = try { git rev-parse --short HEAD 2>$null } catch { "unknown" }
$modelName = [System.IO.Path]::GetFileNameWithoutExtension($Model)
$kvTag     = if ($KvQ8) { "q8" } else { "std" }
Write-Host ""
Write-Host "INTERLEAVED bench (control baseline)  mode=$Mode  model=$modelName  n=$N  samples=$Samples"
Write-Host "Engines: sub0llm @ $gitHash  vs  llama @ $LlamaVersion"
if ($Mode -eq "threads") { Write-Host "Sweep: threads ${TMin}..${TMax}  (K=$K x $Samples passes = $TotalRounds rounds)" }
else { Write-Host "Sweep: ours gpu-layers = $($Variants -join ',')  vs  llama ngl=$Ngl fa=$(if($NoFa){'off'}else{'on'}) KV=$kvTag ctx=$(if($Ctx){$Ctx}else{'auto'})  (K=$K x $Samples = $TotalRounds rounds)" }
if ($DryRun) { Write-Host "*** DRY-RUN — synthetic data ***" }
Write-Host "Prompt ($PromptLen tokens): '$($Prompt.Substring(0,[math]::Min(60,$Prompt.Length)))...'"
Write-Host "NOTE: ours PP = token-by-token; llama PP = batched (architectural — see PP column)"
Write-Host ""

# ── smoke-check (greedy text equality) ─────────────────────────────────────────
if (-not $DryRun) {
    $sv = $Variants[[math]::Min($Variants.Count-1, [int]([math]::Floor($Variants.Count*0.75)))]
    Write-Host "Smoke-check ($VarLabel=$sv, n=8 greedy text) ..."
    $ot = Get-OursGenText $sv 8; $lt = Get-LlamaGenText $sv 8
    if ($null -eq $ot -or $null -eq $lt) {
        Write-Warning "  could not capture text from one engine"
        if ($StrictSmoke) { Write-Error "Smoke-check failed — aborting."; exit 1 }
    } elseif ($ot -ne $lt) {
        Write-Warning "  greedy text DIFFERS:`n    ours : '$ot'`n    llama: '$lt'"
        if ($StrictSmoke) { Write-Error "Engines diverge — aborting (threads mode requires parity)."; exit 1 }
    } else { Write-Host "  greedy match: '$ot'" }
    Write-Host ""
}

# ── main interleaved loop ───────────────────────────────────────────────────────
$sep = "+-------+------+------+-------+----------+--------+----------+----------+"
Write-Host $sep
Write-Host "| round | pass | $($VarLabel.PadLeft(4)) |   min | engine   | wall_s |  PP tok/s|  TG tok/s|"
Write-Host $sep
$start = Get-Date; $round = 0
for ($p = 0; $p -lt $Samples; $p++) {
    for ($i = 0; $i -lt $K; $i++) {
        $v = $Variants[($p + $i) % $K]; $round++
        $elapsed = [math]::Round(((Get-Date)-$start).TotalMinutes,1)
        if ($p % 2 -eq 0) { $om = Get-OursMetrics $v; $lm = Get-LlamaMetrics $v }
        else              { $lm = Get-LlamaMetrics $v; $om = Get-OursMetrics $v }
        if ($null -ne $om.tg) { $oursTG[$v].Add($om.tg) };  if ($null -ne $om.pp) { $oursPP[$v].Add($om.pp) };  if ($null -ne $om.wallS) { $oursWall[$v].Add($om.wallS) }
        if ($null -ne $lm.tg) { $llamaTG[$v].Add($lm.tg) }; if ($null -ne $lm.pp) { $llamaPP[$v].Add($lm.pp) }; if ($null -ne $lm.wallS) { $llamaWall[$v].Add($lm.wallS) }
        $pfx = "| {0,5} | {1,4} | {2,4} | {3,5} |" -f $round, ($p+1), $v, $elapsed
        Write-Host ("$pfx {0,-8} | {1,6}s| {2,9}| {3,9}|" -f 'sub0llm',(Fmt $om.wallS 'F1'),(Fmt $om.pp),(Fmt $om.tg))
        Write-Host ("$pfx {0,-8} | {1,6}s| {2,9}| {3,9}|" -f 'llama',  (Fmt $lm.wallS 'F1'),(Fmt $lm.pp),(Fmt $lm.tg))
        Write-Host $sep
    }
}

# ── summary ────────────────────────────────────────────────────────────────────
$csv = [Collections.Generic.List[string]]::new()
$csv.Add("$VarLabel,engine,wall_med,pp_med,pp_max,tg_med,tg_max,pp_ratio_pct,tg_ratio_pct,pp_samples,tg_samples")
Write-Host ""
Write-Host "=== Summary — median (and max) across $Samples samples per $VarLabel ==="
Write-Host $sep
Write-Host "| stat  | pass | $($VarLabel.PadLeft(4)) |   min | engine   | wall_s |  PP tok/s|  TG tok/s|"
Write-Host $sep
$bestOursTG=0.0; $bestOursTGv=$null; $bestLlamaTG=0.0; $bestLlamaTGv=$null
foreach ($v in $Variants) {
    $otg=$oursTG[$v].ToArray(); $opp=$oursPP[$v].ToArray(); $ows=$oursWall[$v].ToArray()
    $ltg=$llamaTG[$v].ToArray(); $lpp=$llamaPP[$v].ToArray(); $lws=$llamaWall[$v].ToArray()
    $otgMed=Get-Median $otg; $oppMed=Get-Median $opp; $owsMed=Get-Median $ows
    $ltgMed=Get-Median $ltg; $lppMed=Get-Median $lpp; $lwsMed=Get-Median $lws
    $otgMax= if($otg.Count){($otg|Measure-Object -Maximum).Maximum}else{$null}
    $ltgMax= if($ltg.Count){($ltg|Measure-Object -Maximum).Maximum}else{$null}
    $oppMax= if($opp.Count){($opp|Measure-Object -Maximum).Maximum}else{$null}
    $lppMax= if($lpp.Count){($lpp|Measure-Object -Maximum).Maximum}else{$null}
    $tgRatio= if($otgMed -and $ltgMed){100.0*$otgMed/$ltgMed}else{$null}
    $ppRatio= if($oppMed -and $lppMed){100.0*$oppMed/$lppMed}else{$null}
    $pfx="| {0,5} | {1,4} | {2,4} | {3,5} |" -f 'med','-',$v,'-'
    Write-Host ("$pfx {0,-8} | {1,6}s| {2,9}| {3,9}|" -f 'sub0llm',(Fmt $owsMed 'F1'),(Fmt $oppMed),(Fmt $otgMed))
    Write-Host ("$pfx {0,-8} | {1,6}s| {2,9}| {3,9}|" -f 'llama',  (Fmt $lwsMed 'F1'),(Fmt $lppMed),(Fmt $ltgMed))
    Write-Host ("|       |      |      |       | ratio o/l|        | {0,8}%| {1,8}%|" -f (Fmt $ppRatio 'F1'),(Fmt $tgRatio 'F1'))
    Write-Host $sep
    $oppRaw=($opp|ForEach-Object{$_.ToString('F2')}) -join ';'; $otgRaw=($otg|ForEach-Object{$_.ToString('F2')}) -join ';'
    $lppRaw=($lpp|ForEach-Object{$_.ToString('F2')}) -join ';'; $ltgRaw=($ltg|ForEach-Object{$_.ToString('F2')}) -join ';'
    $csv.Add("$v,sub0llm,$(Fmt $owsMed 'F1'),$(Fmt $oppMed),$(Fmt $oppMax),$(Fmt $otgMed),$(Fmt $otgMax),$(Fmt $ppRatio 'F1'),$(Fmt $tgRatio 'F1'),$oppRaw,$otgRaw")
    $csv.Add("$v,llama,$(Fmt $lwsMed 'F1'),$(Fmt $lppMed),$(Fmt $lppMax),$(Fmt $ltgMed),$(Fmt $ltgMax),-,-,$lppRaw,$ltgRaw")
    if ($null -ne $otgMed -and $otgMed -gt $bestOursTG)  { $bestOursTG=$otgMed; $bestOursTGv=$v }
    if ($null -ne $ltgMed -and $ltgMed -gt $bestLlamaTG) { $bestLlamaTG=$ltgMed; $bestLlamaTGv=$v }
}
Write-Host ("BEST TG:  sub0llm {0} @ {1}={2}   llama {3} @ {4}={5}" -f (Fmt $bestOursTG),$VarLabel,$bestOursTGv,(Fmt $bestLlamaTG),$VarLabel,$bestLlamaTGv)
Write-Host ""

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not (Test-Path data)) { New-Item -ItemType Directory data | Out-Null }
$out = "data/bench_${Mode}@${gitHash}_vs_llama_${modelName}_kv${kvTag}_n${N}_${stamp}.csv"
$csv | Set-Content $out -Encoding UTF8
Write-Host "Saved: $out   (total $([math]::Round(((Get-Date)-$start).TotalMinutes,1)) min)"
