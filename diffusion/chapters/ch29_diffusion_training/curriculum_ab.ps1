# curriculum_ab.ps1 — convergence-gated curriculum vs the uniform formal objective.
#
# Curricula are SCHEDULES, so a fixed-step A/B is unfair (it catches the curriculum mid-climb on
# easy levels). The honest comparison is FINAL self-terminated recall: both run to convergence on
# the SAME arch / corpus / seed. The curriculum (--curriculum-converge) trains k=1,3,5,...,63 exact
# masked tokens, advancing on a per-epoch held-out NELBO plateau, then switches to the full
# objective; the baseline trains the uniform formal objective throughout. --steps is a high safety
# bound (the baseline self-terminates early; the curriculum needs room to climb all levels).
param(
    [int]     $Steps   = 80000,
    [int]     $Paras   = 3000,
    [int]     $Vocab   = 512,
    [int]     $Threads = 8,
    [int]     $KStep   = 2,
    [int]     $LvlPat  = 2,    # per-level plateau patience (epochs)
    [string]  $Bin     = "build-native/bin/ch29_diffusion_training.exe",
    [string]  $TmpRoot = "D:/tmp/curriculum_ab"
)
$D=128; $L=4; $H=4; $KV=2; $FF=512

$runs = @(
    @{ tag="baseline";   extra=@() },
    @{ tag="curric_k$KStep"; extra=@("--curriculum-converge","--curriculum-k-step",$KStep,"--curriculum-patience",$LvlPat) }
)
foreach ($r in $runs) {
    $dir = "$TmpRoot/$($r.tag)"
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    Write-Host "`n===== $($r.tag)  ($dir) ====="
    $args = @("--ckpt-dir",$dir,"--paragraphs",$Paras,"--vocab-size",$Vocab,
              "--embed-dim",$D,"--n-layers",$L,"--n-heads",$H,"--n-kv-heads",$KV,"--d-ff",$FF,
              "--threads",$Threads,"--steps",$Steps,"--seed",42) + $r.extra
    & $Bin @args 2>&1 |
        Select-String -Pattern "overall:|breakdown:|held-out NELBO \(|Edge effect|Training done|CONVERGED" |
        ForEach-Object { Write-Host "  [$($r.tag)]  $_" }
}
Write-Host "`nCompare 'overall' + 'word-START' (FINAL self-terminated recall) across baseline vs curriculum."
