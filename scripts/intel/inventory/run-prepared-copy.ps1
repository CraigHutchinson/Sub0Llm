param(
    [string]$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI',
    [string]$CompilerVersion = '2025.3',
    [ValidateSet(257, 1048576)][int[]]$Elements = @(257, 1048576),
    [switch]$CompileOnly
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$output = Join-Path $repo ('out/intel-review/prepared-copy/' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $output -Force | Out-Null
$compilerRoot = Join-Path $OneApiRoot "compiler/$CompilerVersion"
$compiler = Join-Path $compilerRoot 'bin/icx-cl.exe'
$vcvars = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' | Select-Object -First 1
if (-not $vcvars -or -not (Test-Path -LiteralPath $compiler)) { throw 'MSVC environment or Intel compiler missing' }
& cmd /d /c "`"$($vcvars.FullName)`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'vcvars environment import failed' }
$env:PATH = "$compilerRoot/bin;$compilerRoot/bin/compiler;$env:PATH"
$env:LIB = "$compilerRoot/lib;$env:LIB"
$env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'
$source = Join-Path $repo 'benchmarks/intel/mechanisms/prepared_copy.cpp'
$exe = Join-Path $output 'prepared_copy.exe'
$compileArgs = @('/nologo', '/std:c++20', '/EHsc', '-fsycl', '/O2', '/fp:precise', $source, "/Fe:$exe", "/Fo:$output/")
$manifest = [ordered]@{
    schema = 'sub0.intel.prepared-copy-run.v1'; purpose = 'prepared-copy'
    utc = [DateTime]::UtcNow.ToString('o'); repo = $repo; commit = (& git -C $repo rev-parse HEAD)
    probe_changes = @(& git -C $repo status --short -- benchmarks/intel/mechanisms/prepared_copy.cpp scripts/intel/inventory/run-prepared-copy.ps1 docs/INTEL_IGPU_PREPARED_COPY_SPIKE.md)
    os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    compiler = $compiler; compiler_version = @(& $compiler --version); vcvars = $vcvars.FullName
    arguments = $compileArgs; elements = $Elements; compile_only = [bool]$CompileOnly
    source_sha256 = (Get-FileHash -LiteralPath $source).Hash
    runtime_header_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tools/intel_probe/runtime.hpp')).Hash
    runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath).Hash
    power_configuration = @(& powercfg /getactivescheme)
    competing_processes = @(Get-Process -Name '*qwen*','*llama*','*train*','*transplant*','icx*','clang*','nvcc*' -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,CPU)
}
if (-not $CompileOnly) {
    $manifest.sycl_devices = @(& (Join-Path $compilerRoot 'bin/sycl-ls.exe'))
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
& $compiler @compileArgs 2>&1 | Tee-Object -FilePath (Join-Path $output 'build.log')
$manifest.build_exit_code = $LASTEXITCODE
if ($LASTEXITCODE -ne 0) {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
    throw "Probe build failed; evidence: $output"
}
$manifest.executable_sha256 = (Get-FileHash -LiteralPath $exe).Hash
if ($CompileOnly) {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
    Write-Output "Evidence: $output"
    return
}
$runs = @()
foreach ($count in $Elements) {
    $firstPath = Join-Path $output "fixture-$count-a.f32"
    $changedPath = Join-Path $output "fixture-$count-b.f32"
    foreach ($item in @(@{ Path=$firstPath; Delta=0 }, @{ Path=$changedPath; Delta=17 })) {
        $values = [float[]]::new($count)
        for ($i = 0; $i -lt $count; ++$i) { $values[$i] = (($i + $item.Delta) % 257) - 128 }
        $fixtureBytes = [byte[]]::new($values.Length * 4)
        [Buffer]::BlockCopy($values, 0, $fixtureBytes, 0, $fixtureBytes.Length)
        [IO.File]::WriteAllBytes($item.Path, $fixtureBytes)
    }
    $log = Join-Path $output "prepared-copy-$count.log"
    & $exe $firstPath $changedPath "$count" 2>&1 | Tee-Object -FilePath $log
    $runExit = $LASTEXITCODE
    $logText = Get-Content -Raw -LiteralPath $log
    $preparedStatus = if ($logText -match '(?m)^prepared_status=(supported|unsupported)\r?$') { $Matches[1] } else { 'missing' }
    $ordinaryCold = @([regex]::Matches($logText, '(?m)^cold,ordinary,')).Count
    $ordinaryWarm = @([regex]::Matches($logText, '(?m)^warm,ordinary,')).Count
    $stagingCold = @([regex]::Matches($logText, '(?m)^cold,host_usm_staging,')).Count
    $stagingWarm = @([regex]::Matches($logText, '(?m)^warm,host_usm_staging,')).Count
    $preparedCold = @([regex]::Matches($logText, '(?m)^cold,prepared,')).Count
    $preparedWarm = @([regex]::Matches($logText, '(?m)^warm,prepared,')).Count
    $shapeValid = $ordinaryCold -eq 1 -and $ordinaryWarm -eq 7 -and $stagingCold -eq 1 -and $stagingWarm -eq 7
    $shapeValid = $shapeValid -and (($preparedStatus -eq 'supported' -and $preparedCold -eq 1 -and $preparedWarm -eq 7) -or
        ($preparedStatus -eq 'unsupported' -and $preparedCold -eq 0 -and $preparedWarm -eq 0))
    if (-not $shapeValid) { $runExit = 1 }
    $runs += [ordered]@{ elements=$count; bytes=($count * 4); fixture_a_sha256=(Get-FileHash $firstPath).Hash; fixture_b_sha256=(Get-FileHash $changedPath).Hash; exit_code=$runExit; prepared_status=$preparedStatus; sample_shape_valid=$shapeValid; log=(Split-Path $log -Leaf) }
    if ($runExit -ne 0) { break }
}
$manifest.runs = $runs
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
Write-Output "Evidence: $output"
if ($runs.Count -ne $Elements.Count -or @($runs | Where-Object { $_.exit_code -ne 0 -or -not $_.sample_shape_valid }).Count -ne 0) { throw 'Prepared-copy probe failed' }
