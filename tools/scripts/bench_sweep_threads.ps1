#!/usr/bin/env pwsh
# bench_sweep_threads.ps1
#
# Full thread-count sweep (default 1..N_logical_cores) for both sub0llm-gemma and
# llama.cpp, measuring BOTH prompt-processing (PP tok/s) and token-generation (TG tok/s).
#
# Schedule: cyclic-rotation Latin square to balance thermal drift:
#   Pass p runs threads in order: t[(p+0)%K], t[(p+1)%K], ..., t[(p+K-1)%K]
# This guarantees every thread count occupies every time-slot position exactly
# once (when Samples is a multiple of K), so first-order thermal drift cancels.
# Within each slot, ours/llama order alternates per pass to cancel within-slot bias.
#
# Apples-to-apples design:
#   Both engines receive the identical prompt text via --text / --prompt.
#   llama-cli is used (not llama-bench which feeds random synthetic tokens regardless
#   of -p), so KV-cache content and decoding context are truly equivalent.
#   TG: timed after full prompt prefill — identical starting state for both engines.
#   PP: our PP is token-by-token forward_one (TTFT-representative); llama's is batched
#       prefill — a valid architectural difference captured in the PP column.
#
# Usage:
#   .\tools\scripts\bench_sweep_threads.ps1
#   .\tools\scripts\bench_sweep_threads.ps1 -Model models/gemma-4-12b-it-Q8_0.gguf -N 64 -Samples 6
#   .\tools\scripts\bench_sweep_threads.ps1 -TMin 8 -TMax 20   # narrow sweep
#   .\tools\scripts\bench_sweep_threads.ps1 -Prompt "Your custom text here"

param(
    [string]$Model   = "models/gemma-4-12b-it-Q8_0.gguf",
    [int]   $N       = 64,
    [int]   $Samples = 6,
    [int]   $TMin    = 1,
    [int]   $TMax    = -1,  # -1 = auto-detect logical core count
    [string]$Prompt  = "The transformer architecture, introduced in the paper Attention Is All You Need, revolutionised natural language processing by replacing recurrent networks with self-attention mechanisms. Each token attends to every other token in the sequence, allowing the model to capture long-range dependencies in a single forward pass."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($TMax -le 0) { $TMax = [Environment]::ProcessorCount }

$LlamaDir     = "D:/tools/llamacpp/vendor/llama.cpp-prebuilt/b9334/cpu"
$LlamaCli     = Join-Path $LlamaDir "llama-cli.exe"
$LlamaVersion = Split-Path (Split-Path $LlamaDir -Parent) -Leaf
$OursBin      = "./build-native/bin/sub0llm-gemma.exe"
$Threads      = [int[]]($TMin..$TMax)
$K            = $Threads.Count
$TotalRounds  = $Samples * $K

# ── get prompt token length (no model load) ───────────────────────────────────

$PromptLen = try {
    $toks = & $OursBin --model $Model --mode tokenize --text $Prompt 2>$null
    @($toks -split '\s+' | Where-Object { $_ -match '^\d+$' }).Count
} catch { 0 }
if ($PromptLen -lt 1) {
    Write-Warning "Could not tokenize prompt; prompt length will show as 0 in header"
    $PromptLen = 0
}

# ── helpers ───────────────────────────────────────────────────────────────────

# Returns {tg, pp} from a single run of our binary (both from one model load).
# main.cpp greedy mode emits on stderr:
#   [gemma] prompt forward N tok in Xs (PP tok/s)
#   [gemma] generated N tok in Xs (TG tok/s)
function Get-OursMetrics([int]$t) {
    $tg = $null; $pp = $null
    try {
        $out = & $OursBin --model $Model --mode greedy --text $Prompt -n $N -t $t 2>&1
        foreach ($line in $out) {
            if ($line -match 'generated.*\(([0-9.]+) tok/s\)')      { $tg = [double]$Matches[1] }
            if ($line -match 'prompt forward.*\(([0-9.]+) tok/s\)')  { $pp = [double]$Matches[1] }
        }
    } catch { }
    return @{ tg = $tg; pp = $pp }
}

# Returns {tg, pp} from a single llama-cli run with the real prompt text.
# llama-cli prints llama_print_timings to stderr, e.g.:
#   llama_print_timings: prompt eval time = ... (X.XX tokens per second)
#   llama_print_timings:        eval time = ... (X.XX tokens per second)
function Get-LlamaMetrics([int]$t) {
    $tg = $null; $pp = $null
    try {
        $out = & $LlamaCli -m $Model --prompt $Prompt -n $N -t $t `
                           --no-display-prompt -s 42 --log-disable 2>&1
        foreach ($line in $out) {
            if ($line -match 'prompt eval time' -and
                $line -match '([0-9]+\.[0-9]+) tokens per second') { $pp = [double]$Matches[1] }
            if ($line -match 'llama_print_timings:\s+eval time' -and
                $line -match '([0-9]+\.[0-9]+) tokens per second') { $tg = [double]$Matches[1] }
        }
    } catch { }
    return @{ tg = $tg; pp = $pp }
}

function Get-Median([double[]]$vals) {
    if (-not $vals -or $vals.Count -eq 0) { return $null }
    $s = $vals | Sort-Object
    $n = $s.Count
    if ($n % 2 -eq 1) { return $s[($n - 1) / 2] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

function Fmt([object]$v, [string]$fmt = "F2") {
    if ($null -eq $v) { return "N/A" }
    return ([double]$v).ToString($fmt)
}

# ── per-thread accumulators ───────────────────────────────────────────────────

$oursTG  = @{}; $oursPP  = @{}
$llamaTG = @{}; $llamaPP = @{}
foreach ($t in $Threads) {
    $oursTG[$t]  = [System.Collections.Generic.List[double]]::new()
    $oursPP[$t]  = [System.Collections.Generic.List[double]]::new()
    $llamaTG[$t] = [System.Collections.Generic.List[double]]::new()
    $llamaPP[$t] = [System.Collections.Generic.List[double]]::new()
}

# ── header ────────────────────────────────────────────────────────────────────

$gitHash   = try { git rev-parse --short HEAD 2>$null } catch { "unknown" }
$modelName = [System.IO.Path]::GetFileNameWithoutExtension($Model)

Write-Host ""
Write-Host "Thread sweep  model=$modelName  n=$N  t=${TMin}..${TMax}  samples=$Samples"
Write-Host "Schedule: cyclic-rotation Latin square (K=$K threads x $Samples passes = $TotalRounds rounds)"
Write-Host "Engines: sub0llm @ $gitHash  vs  llama.cpp @ $LlamaVersion"
Write-Host "Prompt ($PromptLen tokens): '$Prompt'"
Write-Host "NOTE: llama-cli used (not llama-bench) so both engines receive identical prompt text"
Write-Host "NOTE: our PP = sequential token-by-token prefill; llama PP = batched prefill"
Write-Host ""
Write-Host "Commands (t = thread count for each round):"
Write-Host "  ours : $OursBin --model $Model --mode greedy --text '<prompt>' -n $N -t <t>"
Write-Host "  llama: $LlamaCli -m $Model --prompt '<prompt>' -n $N -t <t> --no-display-prompt -s 42 --log-disable"
Write-Host ""

$startTime = Get-Date
$round = 0

# ── main loop ─────────────────────────────────────────────────────────────────
# One continuous table; two rows (one per engine) per round.
# Columns: round | pass | threads | elapsed | engine | PP tok/s | TG tok/s
#
$tblSep = "+-------+------+------+-------+----------+----------+----------+"
$tblHdr = "| round | pass |    t |   min | engine   |  PP tok/s|  TG tok/s|"
Write-Host $tblSep
Write-Host $tblHdr
Write-Host $tblSep

for ($p = 0; $p -lt $Samples; $p++) {

    for ($i = 0; $i -lt $K; $i++) {
        $t = $Threads[($p + $i) % $K]
        $round++
        $elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)

        # Alternate ours/llama order each pass to cancel within-slot bias
        if ($p % 2 -eq 0) {
            $om = Get-OursMetrics  $t
            $lm = Get-LlamaMetrics $t
        } else {
            $lm = Get-LlamaMetrics $t
            $om = Get-OursMetrics  $t
        }

        if ($null -ne $om.tg) { $oursTG[$t].Add($om.tg) }
        if ($null -ne $om.pp) { $oursPP[$t].Add($om.pp) }
        if ($null -ne $lm.tg) { $llamaTG[$t].Add($lm.tg) }
        if ($null -ne $lm.pp) { $llamaPP[$t].Add($lm.pp) }

        $prefix = "| {0,5} | {1,4} | {2,4} | {3,5} |" -f $round, ($p+1), $t, $elapsed
        Write-Host ("$prefix {0,-8} | {1,9}| {2,9}|" -f 'sub0llm', (Fmt $om.pp), (Fmt $om.tg))
        Write-Host ("$prefix {0,-8} | {1,9}| {2,9}|" -f 'llama',   (Fmt $lm.pp), (Fmt $lm.tg))
        # Separator between rounds; thicker (===) between passes
        if ($i -eq $K - 1) {
            Write-Host $tblSep
        } else {
            Write-Host "+-------+------+------+-------+----------+----------+----------+"
        }
    }
}

# ── summary table ─────────────────────────────────────────────────────────────

$hdr = "{0,7}  {1,8} {2,8}  {3,8} {4,8}  {5,8} {6,8}  {7,8} {8,8}  {9,7}" -f `
    "threads",
    "o_tg_med", "o_tg_max",
    "l_tg_med", "l_tg_max",
    "o_pp_med", "o_pp_max",
    "l_pp_med", "l_pp_max",
    "tg_ratio"
$sep = "-" * ($hdr.Length)

Write-Host "=== Summary (TG = token generation, PP = prompt processing, tok/s) ==="
Write-Host $hdr
Write-Host $sep

$csvLines = [System.Collections.Generic.List[string]]::new()
$csvLines.Add("threads,ours_tg_med,ours_tg_max,llama_tg_med,llama_tg_max,ours_pp_med,ours_pp_max,llama_pp_med,llama_pp_max,tg_ratio_pct")

$bestOursTGt  = $null; $bestOursTGMed  = 0.0
$bestLlamaTGt = $null; $bestLlamaTGMed = 0.0
$bestOursPPt  = $null; $bestOursPPMed  = 0.0
$bestLlamaPPt = $null; $bestLlamaPPMed = 0.0

foreach ($t in $Threads) {
    $otg = $oursTG[$t].ToArray();  $opp = $oursPP[$t].ToArray()
    $ltg = $llamaTG[$t].ToArray(); $lpp = $llamaPP[$t].ToArray()

    $otgMed = Get-Median $otg; $otgMax = if ($otg.Count) { ($otg | Measure-Object -Maximum).Maximum } else { $null }
    $ltgMed = Get-Median $ltg; $ltgMax = if ($ltg.Count) { ($ltg | Measure-Object -Maximum).Maximum } else { $null }
    $oppMed = Get-Median $opp; $oppMax = if ($opp.Count) { ($opp | Measure-Object -Maximum).Maximum } else { $null }
    $lppMed = Get-Median $lpp; $lppMax = if ($lpp.Count) { ($lpp | Measure-Object -Maximum).Maximum } else { $null }
    $tgRatio = if ($null -ne $otgMed -and $null -ne $ltgMed -and $ltgMed -gt 0) { 100.0 * $otgMed / $ltgMed } else { $null }

    Write-Host ("{0,7}  {1,8} {2,8}  {3,8} {4,8}  {5,8} {6,8}  {7,8} {8,8}  {9,7}" -f `
        $t,
        (Fmt $otgMed), (Fmt $otgMax),
        (Fmt $ltgMed), (Fmt $ltgMax),
        (Fmt $oppMed), (Fmt $oppMax),
        (Fmt $lppMed), (Fmt $lppMax),
        (Fmt $tgRatio "F1"))

    $csvLines.Add("$t,$(Fmt $otgMed),$(Fmt $otgMax),$(Fmt $ltgMed),$(Fmt $ltgMax),$(Fmt $oppMed),$(Fmt $oppMax),$(Fmt $lppMed),$(Fmt $lppMax),$(Fmt $tgRatio 'F1')")

    if ($null -ne $otgMed -and $otgMed -gt $bestOursTGMed)  { $bestOursTGMed  = $otgMed; $bestOursTGt  = $t }
    if ($null -ne $ltgMed -and $ltgMed -gt $bestLlamaTGMed) { $bestLlamaTGMed = $ltgMed; $bestLlamaTGt = $t }
    if ($null -ne $oppMed -and $oppMed -gt $bestOursPPMed)  { $bestOursPPMed  = $oppMed; $bestOursPPt  = $t }
    if ($null -ne $lppMed -and $lppMed -gt $bestLlamaPPMed) { $bestLlamaPPMed = $lppMed; $bestLlamaPPt = $t }
}

Write-Host $sep
Write-Host ("Best ours  TG: t={0,2}  {1} tok/s (median)" -f $bestOursTGt,  (Fmt $bestOursTGMed))
Write-Host ("Best llama TG: t={0,2}  {1} tok/s (median)" -f $bestLlamaTGt, (Fmt $bestLlamaTGMed))
Write-Host ("Best ours  PP: t={0,2}  {1} tok/s (median)" -f $bestOursPPt,  (Fmt $bestOursPPMed))
Write-Host ("Best llama PP: t={0,2}  {1} tok/s (median)" -f $bestLlamaPPt, (Fmt $bestLlamaPPMed))
Write-Host ""

# ── save CSV ──────────────────────────────────────────────────────────────────

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$outFile = "data/thread_sweep@${gitHash}_vs_llama@${LlamaVersion}_${modelName}_n${N}_${stamp}.csv"
$csvLines | Set-Content $outFile -Encoding UTF8
Write-Host "Saved: $outFile"
Write-Host "Total time: $([math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)) min"
