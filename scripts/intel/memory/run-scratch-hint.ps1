param(
    [string]$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI',
    [string]$CompilerVersion = '2025.3',
    [ValidateRange(1, 1048576)][int]$Elements = 65536,
    [ValidateSet('shared', 'device', 'shared_prefetch', 'shared_advice')][string[]]$Arms = @('shared', 'device', 'shared_prefetch', 'shared_advice'),
    [Nullable[int]]$AdviceCode,
    [switch]$CompileOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$output = Join-Path $repo ('out/intel-review/scratch-hint/' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $output -Force | Out-Null
$compilerRoot = Join-Path $OneApiRoot "compiler/$CompilerVersion"
$compiler = Join-Path $compilerRoot 'bin/icx-cl.exe'
$vcvars = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
    Select-Object -First 1
if (-not $vcvars -or -not (Test-Path -LiteralPath $compiler)) { throw 'MSVC environment or Intel compiler missing' }
& cmd /d /c "`"$($vcvars.FullName)`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'vcvars environment import failed' }
$env:PATH = "$compilerRoot/bin;$compilerRoot/bin/compiler;$env:PATH"
$env:LIB = "$compilerRoot/lib;$env:LIB"
$source = Join-Path $repo 'benchmarks/intel/memory/scratch_hint.cpp'
$exe = Join-Path $output 'scratch_hint.exe'
$compileArgs = @('/nologo', '/std:c++20', '/EHsc', '-fsycl', '/O2', '/fp:precise', $source, "/Fe:$exe", "/Fo:$output/")
$hasAdviceCode = $PSBoundParameters.ContainsKey('AdviceCode')
$manifest = [ordered]@{
    schema = 'sub0.intel.scratch-hint-run.v1'
    purpose = 'I19 bounded write-only scratch and hint preparation'
    utc = [DateTime]::UtcNow.ToString('o')
    commit = (& git -C $repo rev-parse HEAD)
    probe_changes = @(& git -C $repo status --short -- benchmarks/intel/memory scripts/intel/memory docs/INTEL_IGPU_SCRATCH_HINT_SPIKE.md)
    compiler = $compiler
    compiler_version = @(& $compiler --version)
    source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    runtime_header_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tools/intel_probe/runtime.hpp') -Algorithm SHA256).Hash
    elements = $Elements
    arms = $Arms
    advice_code = if ($hasAdviceCode) { $AdviceCode } else { $null }
    compile_only = [bool]$CompileOnly
    build_arguments = $compileArgs
    runs = @()
}
$buildLog = Join-Path $output 'build.log'
New-Item -ItemType File -Path $buildLog | Out-Null
& $compiler @compileArgs 2>&1 | Tee-Object -FilePath $buildLog -Append
$manifest.build_exit_code = $LASTEXITCODE
if ($LASTEXITCODE -ne 0) {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
    throw "Scratch-hint build failed; evidence: $output"
}
$manifest.executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
if ($CompileOnly) {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
    Write-Output "Evidence: $output"
    return
}
$env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'
$runs = @()
foreach ($arm in $Arms) {
    $log = Join-Path $output "$arm.log"
    $arguments = @($arm, "$Elements")
    if ($arm -eq 'shared_advice' -and $hasAdviceCode) { $arguments += @('--advice-code', "$AdviceCode") }
    & $exe @arguments 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $lines = @(Get-Content -LiteralPath $log)
    $status = @($lines | Where-Object { $_ -match '^status=(pass|fail|unsupported)$' })
    $samples = @($lines | Where-Object { $_ -match '^(cold|warm),' })
    $shapeValid = $status.Count -eq 1
    if ($exitCode -eq 0) {
        $shapeValid = $shapeValid -and $status[0] -eq 'status=pass' -and $samples.Count -eq 8 -and
            @($samples | Where-Object { $_ -match "^cold,$arm," }).Count -eq 1 -and
            @($samples | Where-Object { $_ -match "^warm,$arm," }).Count -eq 7 -and
            @($lines | Where-Object { $_ -eq "selected_arm=$arm" }).Count -eq 1
    } elseif ($exitCode -eq 2) {
        $shapeValid = $shapeValid -and $status[0] -eq 'status=unsupported' -and $samples.Count -eq 0 -and
            @($lines | Where-Object { $_ -match '^reason=.+' }).Count -eq 1
    } else {
        $shapeValid = $shapeValid -and $status[0] -eq 'status=fail'
    }
    $runs += [ordered]@{
        arm = $arm
        arguments = $arguments
        exit_code = $exitCode
        status = if ($status.Count -eq 1) { $status[0].Substring(7) } else { 'invalid' }
        sample_shape_valid = [bool]$shapeValid
        raw_output = (Split-Path $log -Leaf)
        raw_sha256 = (Get-FileHash -LiteralPath $log -Algorithm SHA256).Hash
    }
    if (-not $shapeValid -or $exitCode -eq 1) { break }
}
$manifest.runs = $runs
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
Write-Output "Evidence: $output"
if (@($runs | Where-Object { -not $_.sample_shape_valid -or $_.exit_code -eq 1 }).Count -ne 0) {
    throw 'Scratch-hint probe failed'
}
exit 0
