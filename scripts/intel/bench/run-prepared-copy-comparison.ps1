param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string]$FixtureA,
    [Parameter(Mandatory)][string]$FixtureB,
    [Parameter(Mandatory)][ValidateRange(1, 1048576)][int]$Elements,
    [Parameter(Mandatory)][string]$EnvironmentManifest,
    [ValidateSet('ordinary', 'prepared', 'host_usm_staging')][string]$ArmA = 'ordinary',
    [ValidateSet('ordinary', 'prepared', 'host_usm_staging')][string]$ArmB = 'prepared',
    [ValidateRange(1, 1000)][int]$Pairs = 5,
    [string]$OutputDirectory = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$modulePath = Join-Path $PSScriptRoot 'IntelBenchmarkContract.psm1'
Import-Module $modulePath -Force

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$schemaPath = Join-Path $repo 'benchmarks/intel/common/benchmark-result-v1.schema.json'
$exePath = (Resolve-Path -LiteralPath $Executable).Path
$fixtureAPath = (Resolve-Path -LiteralPath $FixtureA).Path
$fixtureBPath = (Resolve-Path -LiteralPath $FixtureB).Path
$environmentManifestPath = (Resolve-Path -LiteralPath $EnvironmentManifest).Path

foreach ($requiredFile in @($exePath, $fixtureAPath, $fixtureBPath, $environmentManifestPath, $schemaPath, $modulePath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Required file missing: $requiredFile" }
}
if ($ArmA -eq $ArmB) { throw '-ArmA and -ArmB must be distinct' }
$expectedBytes = [long]$Elements * 4L
if ((Get-Item -LiteralPath $fixtureAPath).Length -ne $expectedBytes -or
    (Get-Item -LiteralPath $fixtureBPath).Length -ne $expectedBytes) {
    throw "Both fixtures must contain exactly $expectedBytes bytes for $Elements float32 elements"
}

$environmentManifestText = Get-Content -Raw -LiteralPath $environmentManifestPath
$capturedManifest = $environmentManifestText | ConvertFrom-Json -Depth 100
if (-not $capturedManifest) { throw 'Environment manifest is empty' }
$exeSha256 = Get-IntelBenchmarkSha256 $exePath
$fixtureASha256 = Get-IntelBenchmarkSha256 $fixtureAPath
$fixtureBSha256 = Get-IntelBenchmarkSha256 $fixtureBPath
if ($fixtureASha256 -eq $fixtureBSha256) {
    throw 'Fixture A and fixture B must differ so changed-source validation is meaningful'
}
$requiredManifestFields = @('schema', 'commit', 'os', 'compiler', 'compiler_version', 'source_sha256',
    'runtime_header_sha256', 'executable_sha256', 'sycl_devices')
foreach ($field in $requiredManifestFields) {
    if (-not ($capturedManifest.PSObject.Properties.Name -contains $field)) {
        throw "Environment/build manifest lacks required identity field: $field"
    }
}
if ($capturedManifest.schema -ne 'sub0.intel.prepared-copy-run.v1') {
    throw 'Environment/build manifest has the wrong schema for this consumer'
}
if ("$($capturedManifest.executable_sha256)".ToUpperInvariant() -ne $exeSha256) {
    throw 'Executable SHA-256 does not match the supplied environment/build manifest'
}
if (-not ((@($capturedManifest.sycl_devices) -join "`n") -match '\[level_zero:gpu\]')) {
    throw 'Environment manifest does not identify a Level Zero GPU'
}

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo ('out/intel-review/i21-prepared-copy/' +
        [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
} elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo $OutputDirectory
}
if (Test-Path -LiteralPath $OutputDirectory) { throw "Output directory already exists: $OutputDirectory" }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path

$sourceIdentity = Get-IntelBenchmarkSourceIdentity $repo
$sourceIdentity.repository = $repo
$sourceIdentity.runner_sha256 = Get-IntelBenchmarkSha256 $PSCommandPath
$sourceIdentity.module_sha256 = Get-IntelBenchmarkSha256 $modulePath
$sourceIdentity.schema_sha256 = Get-IntelBenchmarkSha256 $schemaPath

$powerConfiguration = @(& powercfg /getactivescheme 2>&1 | ForEach-Object { "$_" })
$competingProcesses = @(
    Get-Process -Name '*qwen*','*llama*','*train*','*transplant*','icx*','clang*','nvcc*' `
        -ErrorAction SilentlyContinue |
        Select-Object ProcessName, Id, CPU, WorkingSet64
)
$previousDeviceSelector = $env:ONEAPI_DEVICE_SELECTOR
$env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'

$schedule = @(New-IntelBenchmarkSchedule -Arms @($ArmA, $ArmB) -Pairs $Pairs)
$armDefinitions = [ordered]@{
    ordinary = 'Read-only mapped source copied directly to device USM without range preparation.'
    prepared = 'Read-only mapped source prepared once per process, copied directly to device USM, then released.'
    host_usm_staging = 'Read-only mapped source copied by the CPU into reused host USM before the device copy.'
}
$result = [ordered]@{
    schema = 'sub0.intel.benchmark-result.v1'
    run_id = Split-Path $output -Leaf
    created_utc = [DateTime]::UtcNow.ToString('o')
    purpose = 'I21 independent-process prepared-copy comparison groundwork'
    evidence_level = 'instruction'
    synthetic = $false
    outcome = 'fail'
    outcome_reason = 'run_incomplete'
    source_identity = $sourceIdentity
    artifacts = [ordered]@{
        executable = [ordered]@{
            path = $exePath
            sha256 = $exeSha256
            bytes = [long](Get-Item -LiteralPath $exePath).Length
        }
        fixture_a = [ordered]@{
            path = $fixtureAPath
            sha256 = $fixtureASha256
            bytes = [long](Get-Item -LiteralPath $fixtureAPath).Length
        }
        fixture_b = [ordered]@{
            path = $fixtureBPath
            sha256 = $fixtureBSha256
            bytes = [long](Get-Item -LiteralPath $fixtureBPath).Length
        }
        environment_manifest = [ordered]@{
            path = $environmentManifestPath
            sha256 = Get-IntelBenchmarkSha256 $environmentManifestPath
            bytes = [long](Get-Item -LiteralPath $environmentManifestPath).Length
        }
    }
    environment = [ordered]@{
        captured_manifest = $capturedManifest
        machine_name = [Environment]::MachineName
        os_description = [Runtime.InteropServices.RuntimeInformation]::OSDescription
        os_architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        process_architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
        powershell_version = $PSVersionTable.PSVersion.ToString()
        oneapi_device_selector = $env:ONEAPI_DEVICE_SELECTOR
        power_configuration = $powerConfiguration
        competing_processes = $competingProcesses
    }
    comparison = [ordered]@{
        design = 'alternating-paired-independent-process'
        arms = @($ArmA, $ArmB)
        arm_definitions = $armDefinitions
        pair_count = $Pairs
        planned_process_count = $schedule.Count
        schedule = $schedule
        process_isolation = $true
        raw_sample_retention = $true
        consumer_process_protocol = 'Each schedule entry starts a new prepared_copy.exe process. The target arm labels the retained comparison samples; the current executable also runs every available mode internally in fixed ordinary/prepared/host_usm_staging order.'
        comparison_eligible = $false
    }
    timing_provenance = [ordered]@{
        host_clock = 'std::chrono::steady_clock'
        async_completion = 'Each transfer and kernel interval ends after sycl::event completion through wait_and_throw; the process drains the queue before releasing prepared ranges and USM.'
        component_boundaries = [ordered]@{
            staging_ms = 'CPU std::copy_n of the mapped range into preallocated host USM.'
            transfer_ms = 'Before queue.memcpy host-to-device submission through that event wait_and_throw completion.'
            kernel_ms = 'Before queue.parallel_for submission through that event wait_and_throw completion.'
            readback_ms = 'Before queue.memcpy device-to-host submission through that event wait_and_throw completion.'
            validation_ms = 'CPU every-element verification only.'
            validation_inclusive_ms = 'Before staging/transfer through completed readback and every-element validation.'
        }
        process_elapsed_boundary = 'PowerShell Stopwatch around one native process invocation, including process startup, all modes, validation and teardown.'
        empty_harness_overhead = [ordered]@{
            status = 'unavailable'
            milliseconds = $null
            reason = 'The existing consumer has no empty-work mode; no device run was added solely to estimate launch overhead.'
        }
    }
    processes = @()
    limits = @(
        'The prepared-copy consumer executes all available modes internally in fixed order; process scheduling alternates target-arm retention but cannot reverse that internal order.',
        'A pass means the requested checked samples were collected. comparison_eligible remains false until a consumer can execute one selected arm per process or vary internal order.',
        'The supplied environment manifest is hashed and embedded; it must be captured for the same executable and hardware window.',
        'No aggregation, outlier exclusion, noise threshold or promotion decision is performed by this groundwork runner.'
    )
}

$resultPath = Join-Path $output 'result.json'
function Write-CurrentResult {
    $result | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $resultPath -Encoding utf8
}
Write-CurrentResult

try {
    foreach ($entry in $schedule) {
        $logName = 'pair-{0:D3}-position-{1}-{2}.log' -f $entry.pair_index, $entry.position_in_pair, $entry.arm
        $logPath = Join-Path $output $logName
        [string[]]$arguments = @($fixtureAPath, $fixtureBPath, "$Elements")
        $startedUtc = [DateTime]::UtcNow.ToString('o')
        $stopwatch = [Diagnostics.Stopwatch]::StartNew()
        [string[]]$lines = @(& $exePath @arguments 2>&1 | ForEach-Object { "$_" })
        $exitCode = $LASTEXITCODE
        $stopwatch.Stop()
        $lines | Set-Content -LiteralPath $logPath -Encoding utf8
        $parsed = ConvertFrom-IntelPreparedCopyOutput -Lines $lines -ExitCode $exitCode `
            -TargetArm $entry.arm -ProcessSequence $entry.sequence -PairIndex $entry.pair_index
        $processRecord = [ordered]@{
            sequence = $entry.sequence
            pair_index = $entry.pair_index
            position_in_pair = $entry.position_in_pair
            target_arm = $entry.arm
            started_utc = $startedUtc
            elapsed_ms = $stopwatch.Elapsed.TotalMilliseconds
            command = @($exePath) + $arguments
            exit_code = $exitCode
            outcome = $parsed.outcome
            outcome_reason = $parsed.outcome_reason
            raw_output = [ordered]@{
                path = $logName
                sha256 = Get-IntelBenchmarkSha256 $logPath
                bytes = [long](Get-Item -LiteralPath $logPath).Length
            }
            metadata = $parsed.metadata
            samples = $parsed.samples
        }
        $result.processes = @($result.processes) + @($processRecord)
        if ($parsed.outcome -ne 'pass') {
            $result.outcome = $parsed.outcome
            $result.outcome_reason = "Process $($entry.sequence), arm $($entry.arm): $($parsed.outcome_reason)"
            Write-CurrentResult
            break
        }
        Write-CurrentResult
    }

    if ($result.processes.Count -eq $schedule.Count -and
        @($result.processes | Where-Object outcome -ne 'pass').Count -eq 0) {
        $result.outcome = 'pass'
        $result.outcome_reason = $null
    }
    $contractErrors = @(Test-IntelBenchmarkResult $result)
    if ($contractErrors.Count -ne 0) {
        $result.outcome = 'fail'
        $result.outcome_reason = 'Contract validation failed: ' + ($contractErrors -join '; ')
    }
    Write-CurrentResult
} finally {
    $env:ONEAPI_DEVICE_SELECTOR = $previousDeviceSelector
}

Write-Output "Result: $resultPath"
if ($result.outcome -eq 'unsupported') { exit 2 }
if ($result.outcome -ne 'pass') { exit 1 }
