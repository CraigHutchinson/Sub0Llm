Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$modulePath = Join-Path $PSScriptRoot 'IntelBenchmarkContract.psm1'
Import-Module $modulePath -Force
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$schemaPath = Join-Path $repo 'benchmarks/intel/common/benchmark-result-v1.schema.json'

$null = Get-Content -Raw -LiteralPath $schemaPath | ConvertFrom-Json -Depth 100
$schedule = @(New-IntelBenchmarkSchedule -Arms @('ordinary', 'prepared') -Pairs 5)
$expectedOrder = 'ordinary,prepared,prepared,ordinary,ordinary,prepared,prepared,ordinary,ordinary,prepared'
if (($schedule.arm -join ',') -ne $expectedOrder) { throw 'Balanced alternating schedule validation failed' }
if (@($schedule | ForEach-Object { $_.arm } | Group-Object | Where-Object Count -ne 5).Count -ne 0) {
    throw 'Each arm must occur exactly once in every pair'
}

function New-SyntheticPreparedCopyLog {
    param([switch]$PreparedUnsupported)
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('schema=sub0.intel.prepared-copy.v1')
    $lines.Add('backend=level_zero')
    $lines.Add('cold_columns=phase,mode,bytes,setup_first_ms,setup_changed_ms,staging_ms,transfer_ms,kernel_ms,readback_ms,validation_ms,validation_inclusive_ms')
    $lines.Add('warm_columns=phase,mode,bytes,trial,staging_ms,transfer_ms,kernel_ms,readback_ms,validation_ms,validation_inclusive_ms')
    $lines.Add('release_columns=phase,mode,bytes,release_first_ms,release_changed_ms')
    if ($PreparedUnsupported) {
        $lines.Add('copy_optimize_macro=unavailable')
        $lines.Add('prepared_status=unsupported')
        $lines.Add('prepared_reason=synthetic unsupported path')
        $modes = @('ordinary', 'host_usm_staging')
    } else {
        $lines.Add('copy_optimize_macro=1')
        $lines.Add('prepared_status=supported')
        $modes = @('ordinary', 'prepared', 'host_usm_staging')
    }
    foreach ($mode in $modes) {
        $lines.Add("cold,$mode,1028,0,0,0,0.1,0.2,0.1,0.01,0.41")
        for ($trial = 0; $trial -lt 7; ++$trial) {
            $lines.Add("warm,$mode,1028,$trial,0,0.1,0.2,0.1,0.01,0.41")
        }
        $lines.Add("release,$mode,1028,0,0")
    }
    $lines.Add('verified_elements_per_run=257')
    $lines.Add('status=pass')
    return @($lines)
}

$supportedLines = @(New-SyntheticPreparedCopyLog)
$ordinary = ConvertFrom-IntelPreparedCopyOutput -Lines $supportedLines -ExitCode 0 `
    -TargetArm ordinary -ProcessSequence 0 -PairIndex 0
$prepared = ConvertFrom-IntelPreparedCopyOutput -Lines $supportedLines -ExitCode 0 `
    -TargetArm prepared -ProcessSequence 1 -PairIndex 0
if ($ordinary.outcome -ne 'pass' -or $prepared.outcome -ne 'pass') {
    throw 'Supported synthetic output did not parse as pass'
}
if (@($ordinary.samples | Where-Object selected_for_comparison).Count -ne 9) {
    throw 'Raw target-arm sample retention shape is wrong'
}

$unsupportedLines = @(New-SyntheticPreparedCopyLog -PreparedUnsupported)
$unsupported = ConvertFrom-IntelPreparedCopyOutput -Lines $unsupportedLines -ExitCode 0 `
    -TargetArm prepared -ProcessSequence 0 -PairIndex 0
if ($unsupported.outcome -ne 'unsupported') { throw 'Unsupported output was not classified explicitly' }
$failed = ConvertFrom-IntelPreparedCopyOutput -Lines @('status=fail', 'error=synthetic failure') -ExitCode 1 `
    -TargetArm ordinary -ProcessSequence 0 -PairIndex 0
if ($failed.outcome -ne 'fail') { throw 'Failed output was not classified explicitly' }

$zeroHash = '0' * 64
function New-SyntheticProcessRecord {
    param([int]$Sequence, [string]$Arm, $Parsed)
    return [ordered]@{
        sequence = $Sequence
        pair_index = 0
        position_in_pair = $Sequence
        target_arm = $Arm
        started_utc = '2026-09-09T00:00:00Z'
        elapsed_ms = 1.0
        command = @('synthetic-prepared-copy.exe', 'a.f32', 'b.f32', '257')
        exit_code = 0
        outcome = 'pass'
        outcome_reason = $null
        raw_output = [ordered]@{ path = "synthetic-$Sequence.log"; sha256 = $zeroHash; bytes = 1 }
        metadata = $Parsed.metadata
        samples = $Parsed.samples
    }
}
$result = [ordered]@{
    schema = 'sub0.intel.benchmark-result.v1'
    run_id = 'synthetic-validation'
    created_utc = '2026-09-09T00:00:00Z'
    purpose = 'schema validation only'
    evidence_level = 'instruction'
    synthetic = $true
    outcome = 'pass'
    outcome_reason = $null
    source_identity = [ordered]@{
        repository = 'synthetic'
        commit = $null
        b09_label = 'nogit'
        worktree_state = 'unknown'
        dirty_paths = @()
        runner_sha256 = $zeroHash
        module_sha256 = $zeroHash
        schema_sha256 = $zeroHash
    }
    artifacts = [ordered]@{
        executable = [ordered]@{ path = 'synthetic.exe'; sha256 = $zeroHash; bytes = 1 }
        fixture_a = [ordered]@{ path = 'a.f32'; sha256 = $zeroHash; bytes = 1028 }
        fixture_b = [ordered]@{ path = 'b.f32'; sha256 = $zeroHash; bytes = 1028 }
        environment_manifest = [ordered]@{ path = 'environment.json'; sha256 = $zeroHash; bytes = 1 }
    }
    environment = [ordered]@{
        captured_manifest = [ordered]@{ synthetic = $true }
        machine_name = 'synthetic'
        os_description = 'synthetic'
        os_architecture = 'X64'
        process_architecture = 'X64'
        powershell_version = $PSVersionTable.PSVersion.ToString()
        oneapi_device_selector = 'level_zero:gpu'
        power_configuration = @('synthetic')
        competing_processes = @()
    }
    comparison = [ordered]@{
        design = 'alternating-paired-independent-process'
        arms = @('ordinary', 'prepared')
        arm_definitions = [ordered]@{ ordinary = 'synthetic'; prepared = 'synthetic' }
        pair_count = 1
        planned_process_count = 2
        schedule = @(New-IntelBenchmarkSchedule -Arms @('ordinary', 'prepared') -Pairs 1)
        process_isolation = $true
        raw_sample_retention = $true
        consumer_process_protocol = 'synthetic validation'
        comparison_eligible = $false
    }
    timing_provenance = [ordered]@{
        host_clock = 'std::chrono::steady_clock'
        async_completion = 'synthetic wait_and_throw contract'
        component_boundaries = [ordered]@{ transfer_ms = 'synthetic boundary' }
        process_elapsed_boundary = 'synthetic boundary'
        empty_harness_overhead = [ordered]@{
            status = 'unavailable'; milliseconds = $null; reason = 'synthetic validation'
        }
    }
    processes = @(
        (New-SyntheticProcessRecord -Sequence 0 -Arm ordinary -Parsed $ordinary),
        (New-SyntheticProcessRecord -Sequence 1 -Arm prepared -Parsed $prepared)
    )
    limits = @('Synthetic validation is not performance evidence.')
}

$contractErrors = @(Test-IntelBenchmarkResult $result)
if ($contractErrors.Count -ne 0) { throw "Contract validation failed: $($contractErrors -join '; ')" }
$json = $result | ConvertTo-Json -Depth 100
if (-not (Test-Json -Json $json -SchemaFile $schemaPath -ErrorAction Stop)) {
    throw 'Synthetic result failed JSON Schema validation'
}
Write-Output 'I21 benchmark contract validation passed (schema, schedule, raw parsing, pass/fail/unsupported).'
