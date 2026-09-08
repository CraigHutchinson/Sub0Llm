param(
    [string]$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI',
    [string]$CompilerVersion = '2025.3',
    [switch]$WithOneDnn,
    [switch]$DenseBenchmark,
    [switch]$MemoryBenchmark,
    [switch]$CheckFailures,
    [ValidateSet(257, 65536)][int]$Elements = 65536
)
$ErrorActionPreference = 'Stop'
if ($CheckFailures -and ($DenseBenchmark -or $MemoryBenchmark)) { throw 'Negative tests apply only to groundwork' }
if ($DenseBenchmark -and $MemoryBenchmark) { throw 'Choose one benchmark mode' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$scenario = if ($DenseBenchmark) { 'dense' } elseif ($MemoryBenchmark) { 'memory' } else { 'groundwork' }
$output = Join-Path $repo ("out/intel-review/$scenario/" + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $output -Force | Out-Null
$compilerRoot = Join-Path $OneApiRoot "compiler/$CompilerVersion"
$compiler = Join-Path $compilerRoot 'bin/icx-cl.exe'
$vcvars = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat' |
    Select-Object -First 1
if (-not $vcvars -or -not (Test-Path -LiteralPath $compiler)) { throw 'MSVC environment or Intel compiler missing' }
# Same established vcvars import as scripts/workflow.ps1; paths are installed-tool paths.
& cmd /d /c "`"$($vcvars.FullName)`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'vcvars environment import failed' }
$env:PATH = "$compilerRoot/bin;$compilerRoot/bin/compiler;$env:PATH"
$env:LIB = "$compilerRoot/lib;$env:LIB"
$env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'
$fixture = Join-Path $output 'mapped-fixture.f32'
$values = [float[]]::new($Elements)
for ($i = 0; $i -lt $values.Length; ++$i) { $values[$i] = ($i % 257) - 128 }
$fixtureBytes = [byte[]]::new($values.Length * 4)
[Buffer]::BlockCopy($values, 0, $fixtureBytes, 0, $fixtureBytes.Length)
[IO.File]::WriteAllBytes($fixture, $fixtureBytes)
$relativeSource = if ($DenseBenchmark) { 'benchmarks/intel/mechanisms/dense.cpp' } elseif ($MemoryBenchmark) { 'benchmarks/intel/mechanisms/memory.cpp' } else { 'tools/intel_probe/groundwork.cpp' }
$source = Join-Path $repo $relativeSource
$exe = Join-Path $output 'groundwork.exe'
$compileArgs = @("/DSUB0_PROBE_ELEMENTS=$Elements", '/nologo', '/std:c++20', '/EHsc', '-fsycl', '/Od', $source, "/Fe:$exe", "/Fo:$output/")
if ($DenseBenchmark) { $WithOneDnn = $true }
if ($DenseBenchmark -or $MemoryBenchmark) {
    $compileArgs = $compileArgs | Where-Object { $_ -ne '/Od' }
    $compileArgs += @('/O2', '/fp:precise')
}
if ($WithOneDnn) {
    $dnnl = Join-Path $OneApiRoot 'dnnl/2025.3'
    $env:PATH = "$dnnl/bin;$env:PATH"
    $compileArgs += @('/DSUB0_PROBE_DNNL', "/I$dnnl/include", (Join-Path $dnnl 'lib/dnnl.lib'))
}
$manifest = [ordered]@{
    schema = 'sub0.intel.groundwork.v1'; purpose = $scenario
    utc = [DateTime]::UtcNow.ToString('o'); repo = $repo
    commit = (& git -C $repo rev-parse HEAD); probe_changes = @(& git -C $repo status --short -- tools/intel_probe scripts/intel/inventory benchmarks/intel)
    os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    compiler = $compiler; compiler_version = @(& $compiler --version)
    vcvars = $vcvars.FullName; arguments = $compileArgs
    source_sha256 = (Get-FileHash -LiteralPath $source).Hash
    runtime_header_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo "tools/intel_probe/runtime.hpp")).Hash
    runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath).Hash
    power_configuration = @(& powercfg /getactivescheme)
    competing_processes = @(Get-Process -Name "*qwen*","*llama*","*train*","*transplant*","icx*","clang*","nvcc*" -ErrorAction SilentlyContinue | Select-Object ProcessName,Id,CPU)
    fixture_sha256 = (Get-FileHash -LiteralPath $fixture).Hash
    with_onednn = [bool]$WithOneDnn; elements = $Elements
    sycl_devices = @(& (Join-Path $compilerRoot 'bin/sycl-ls.exe'))
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
& $compiler @compileArgs 2>&1 | Tee-Object -FilePath (Join-Path $output 'build.log')
$buildExit = $LASTEXITCODE
$manifest.build_exit_code = $buildExit
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output "manifest.json")
if ($buildExit -ne 0) { throw "Probe build failed ($buildExit); evidence: $output" }
$manifest.executable_sha256 = (Get-FileHash -LiteralPath $exe).Hash
[string[]]$runArgs = if ($DenseBenchmark -or $MemoryBenchmark) { @() } else { @($fixture) }
& $exe @runArgs 2>&1 | Tee-Object -FilePath (Join-Path $output 'probe.log')
$probeExit = $LASTEXITCODE
$manifest.probe_exit_code = $probeExit
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
Write-Output "Evidence: $output"
if ($probeExit -ne 0) { throw "Probe failed ($probeExit)" }

if ($CheckFailures) {
    $short = Join-Path $output 'truncated.f32'
    [IO.File]::WriteAllBytes($short, [byte[]]::new(4))
    $bad = Join-Path $output 'bad-values.f32'
    [IO.File]::WriteAllBytes($bad, [byte[]]::new($fixtureBytes.Length))
    $cases = @(
        @{ name='missing_argument'; args=@(); reason='Usage:'; selector='level_zero:gpu' },
        @{ name='missing_file'; args=@((Join-Path $output 'absent.f32')); reason='Cannot open'; selector='level_zero:gpu' },
        @{ name='truncated'; args=@($short); reason='Wrong fixture extent'; selector='level_zero:gpu' },
        @{ name='bad_values'; args=@($bad); reason='Fixture contents invalid'; selector='level_zero:gpu' },
        @{ name='no_gpu_fallback'; args=@($fixture); reason='Required Intel'; selector='opencl:cpu' }
    )
    $results = foreach ($case in $cases) {
        $env:ONEAPI_DEVICE_SELECTOR = $case.selector
        [string[]]$caseArgs = $case.args
        $messages = @(& $exe @caseArgs 2>&1 | ForEach-Object { "$_" })
        $code = $LASTEXITCODE
        $messages | Set-Content -LiteralPath (Join-Path $output ($case.name + '.txt'))
        if ($code -eq 0 -or -not (($messages -join "`n").Contains($case.reason))) {
            throw "Negative case did not fail for the intended reason: $($case.name)"
        }
        @{case=$case.name; exit_code=$code; expected_reason=$case.reason; status='pass'}
    }
    $results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output 'negative-tests.json')
    Write-Output 'All five negative checks passed'
}
