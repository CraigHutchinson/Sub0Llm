# overfit_ladder.ps1 — the MINIMAL-CASE capacity sweep (Ch29 --overfit).
#
# Runs `ch29_diffusion_training --overfit N` over a ladder of N and a set of model
# architectures, and prints a compact table: for each (arch, N), did it FIT (train recall
# ≥98%) and at what step. The point: bound how much capacity MEMORIZATION needs, which
# separates "do we need 8 layers?" (capacity) from "why doesn't it generalize?" (data).
#
#   N=1   degenerate — memorize one constant output (sanity that the loop works at all)
#   N=4/16/64  force progressively more real modeling to disambiguate windows
#
# A tiny arch that fits N=64 to 100% proves capacity is NOT the corpus-scale bottleneck —
# the founded +11pt was PROPORTIONS, not raw size, and generalization is data-bound.
#
# Usage:  pwsh diffusion/chapters/ch29_diffusion_training/overfit_ladder.ps1
param(
    [int[]]   $Ns     = @(1, 4, 16, 64),
    [int]     $Steps  = 8000,
    [int]     $Paras  = 400,
    [int]     $Vocab  = 512,
    [string]  $Bin    = "build-native/bin/ch29_diffusion_training.exe",
    [string]  $TmpRoot= "D:/tmp/overfit_ladder"
)

# (label, embed_dim, n_layers, n_heads, n_kv_heads, d_ff) — head_dim = embed/heads.
# All founded-proportioned (head_dim 32, d_ff = 4*D) so we vary SIZE, not proportions.
$archs = @(
    @{ name="D64-L2";  D=64;  L=2; H=2; KV=1; FF=256 },   # head_dim 32, ~0.15M
    @{ name="D128-L4"; D=128; L=4; H=4; KV=2; FF=512 },   # head_dim 32, ~1M
    @{ name="D256-L6"; D=256; L=6; H=8; KV=4; FF=1024 }   # founded, head_dim 32, ~6M
)

$rows = @()
foreach ($a in $archs) {
    foreach ($N in $Ns) {
        $dir = "$TmpRoot/$($a.name)_N$N"
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        $log = & $Bin --overfit $N --ckpt-dir $dir --paragraphs $Paras --vocab-size $Vocab `
            --embed-dim $a.D --n-layers $a.L --n-heads $a.H --n-kv-heads $a.KV --d-ff $a.FF `
            --threads 1 --eval-every 200 --steps $Steps 2>&1 | Out-String
        $fitStep = if ($log -match "FIT . — train recall .* at step (\d+)") { $matches[1] } else { "" }
        $verdict = if ($log -match "Verdict: (FIT|PARTIAL|FAIL)[^\n]*\(?(\d+)?") { $matches[1] } else { "?" }
        $overall = if ($log -match "overall TRAIN recall:.*\(([\d.]+)%\)") { $matches[1] } else { "?" }
        $params  = if ($log -match "params=(\d+)") { $matches[1] } else { "?" }
        $rows += [pscustomobject]@{
            Arch=$a.name; Params=$params; N=$N; Verdict=$verdict; Recall="$overall%"; FitStep=$fitStep
        }
        Write-Host ("  {0,-9} N={1,-3} -> {2,-7} recall {3,5}%  fit@{4}" -f $a.name,$N,$verdict,$overall,$fitStep)
    }
}
Write-Host "`n=== Overfit capacity ladder ==="
$rows | Format-Table -AutoSize
