# tmax_ab.ps1 — matched-budget A/B: does dropping the low-signal high-t tail help?
#
# t_max=1.0 trains t~U(0.02,1.0]: the top ~0.8% of draws FULLY mask the window (zero
# visible tokens, only the marginal is learnable) and the whole t>0.7 tail sits near the
# entropy floor (per-t diagnostic). t_max=0.8 removes that tail entirely — at T=64 every
# window keeps ≥~13 visible tokens, so every step has real context to denoise from.
#
# FAIR COMPARISON: identical arch / seed / corpus / FIXED step budget; early-stopping
# DISABLED (--patience huge) so both run the exact same number of steps = same compute.
# The held-out recall sweep is at fixed noises {0.10..0.75} (both ≤0.8), so both models
# are scored on the IDENTICAL task — the eval is t_max-independent. Lower held-out NELBO
# / higher recall at matched budget ⇒ the tail was wasted compute.
param(
    [double[]] $Tmax   = @(1.0, 0.8),
    [int]      $Steps  = 30000,
    [int]      $Paras  = 3000,
    [int]      $Vocab  = 512,
    [int]      $Threads= 8,
    [string]   $Bin    = "build-native/bin/ch29_diffusion_training.exe",
    [string]   $TmpRoot= "D:/tmp/tmax_ab"
)
# Founded proportions (head_dim 32, d_ff = 4*D), mid size for fast turnaround.
$D=128; $L=4; $H=4; $KV=2; $FF=512

foreach ($tm in $Tmax) {
    $tag = ("t{0:N2}" -f $tm) -replace '\.',''
    $dir = "$TmpRoot/$tag"
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    Write-Host "`n===== t_max=$tm  ($dir) ====="
    & $Bin --t-max $tm --ckpt-dir $dir --paragraphs $Paras --vocab-size $Vocab `
        --embed-dim $D --n-layers $L --n-heads $H --n-kv-heads $KV --d-ff $FF `
        --threads $Threads --steps $Steps --patience 100000 --seed 42 `
        --track-recall 2>&1 |
        Select-String -Pattern "overall:|breakdown:|held-out NELBO \(|Edge effect|Training done" |
        ForEach-Object { Write-Host "  t=$tm  $_" }
}
Write-Host "`nCompare the 'overall' and 'word-START' lines across the two t_max values."
