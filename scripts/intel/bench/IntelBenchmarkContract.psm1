Set-StrictMode -Version Latest

function Get-IntelBenchmarkSha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-IntelBenchmarkSourceIdentity {
    param([Parameter(Mandatory)][string]$Repository)

    [string[]]$commitLines = @(& git -C $Repository rev-parse HEAD 2>$null)
    $commitExit = $LASTEXITCODE
    [string[]]$shortLines = @(& git -C $Repository rev-parse --short HEAD 2>$null)
    $shortExit = $LASTEXITCODE
    [string[]]$statusLines = @(& git -C $Repository status --porcelain=v1 --untracked-files=all 2>$null)
    $statusExit = $LASTEXITCODE

    $commit = if ($commitExit -eq 0 -and $commitLines.Count -eq 1) { $commitLines[0].Trim() } else { $null }
    $short = if ($shortExit -eq 0 -and $shortLines.Count -eq 1) { $shortLines[0].Trim() } else { 'nogit' }
    $state = if ($statusExit -ne 0) { 'unknown' } elseif ($statusLines.Count -eq 0) { 'clean' } else { 'dirty' }
    $b09Label = if ($short -eq 'nogit') {
        'nogit'
    } elseif ($state -eq 'clean') {
        $short
    } else {
        "$short-$state"
    }

    return [ordered]@{
        commit = $commit
        b09_label = $b09Label
        worktree_state = $state
        dirty_paths = @($statusLines)
    }
}

function New-IntelBenchmarkSchedule {
    param(
        [Parameter(Mandatory)][ValidateCount(2, 2)][string[]]$Arms,
        [Parameter(Mandatory)][ValidateRange(1, 1000)][int]$Pairs
    )
    if ($Arms[0] -eq $Arms[1]) { throw 'Comparison arms must be distinct' }

    $schedule = [System.Collections.Generic.List[object]]::new($Pairs * 2)
    $sequence = 0
    for ($pair = 0; $pair -lt $Pairs; ++$pair) {
        $order = if (($pair % 2) -eq 0) { @($Arms[0], $Arms[1]) } else { @($Arms[1], $Arms[0]) }
        for ($position = 0; $position -lt 2; ++$position) {
            $schedule.Add([ordered]@{
                sequence = $sequence
                pair_index = $pair
                position_in_pair = $position
                arm = $order[$position]
            })
            ++$sequence
        }
    }
    return @($schedule)
}

function ConvertTo-IntelInvariantDouble {
    param([Parameter(Mandatory)][string]$Value)
    $number = 0.0
    if (-not [double]::TryParse(
        $Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$number)) {
        throw "Invalid invariant floating-point value: $Value"
    }
    return $number
}

function New-IntelPreparedCopySample {
    param(
        [Parameter(Mandatory)][string[]]$Fields,
        [Parameter(Mandatory)][int]$ProcessSequence,
        [Parameter(Mandatory)][int]$PairIndex,
        [Parameter(Mandatory)][string]$TargetArm
    )
    $phase = $Fields[0]
    $mode = $Fields[1]
    $sample = [ordered]@{
        process_sequence = $ProcessSequence
        pair_index = $PairIndex
        target_arm = $TargetArm
        selected_for_comparison = ($mode -eq $TargetArm)
        phase = $phase
        mode = $mode
        bytes = [long]$Fields[2]
        trial = $null
        setup_first_ms = $null
        setup_changed_ms = $null
        staging_ms = $null
        transfer_ms = $null
        kernel_ms = $null
        readback_ms = $null
        validation_ms = $null
        validation_inclusive_ms = $null
        release_first_ms = $null
        release_changed_ms = $null
    }
    switch ($phase) {
        'cold' {
            if ($Fields.Count -ne 11) { throw "Cold sample has $($Fields.Count) columns; expected 11" }
            $sample.setup_first_ms = ConvertTo-IntelInvariantDouble $Fields[3]
            $sample.setup_changed_ms = ConvertTo-IntelInvariantDouble $Fields[4]
            $sample.staging_ms = ConvertTo-IntelInvariantDouble $Fields[5]
            $sample.transfer_ms = ConvertTo-IntelInvariantDouble $Fields[6]
            $sample.kernel_ms = ConvertTo-IntelInvariantDouble $Fields[7]
            $sample.readback_ms = ConvertTo-IntelInvariantDouble $Fields[8]
            $sample.validation_ms = ConvertTo-IntelInvariantDouble $Fields[9]
            $sample.validation_inclusive_ms = ConvertTo-IntelInvariantDouble $Fields[10]
        }
        'warm' {
            if ($Fields.Count -ne 10) { throw "Warm sample has $($Fields.Count) columns; expected 10" }
            $sample.trial = [int]$Fields[3]
            $sample.staging_ms = ConvertTo-IntelInvariantDouble $Fields[4]
            $sample.transfer_ms = ConvertTo-IntelInvariantDouble $Fields[5]
            $sample.kernel_ms = ConvertTo-IntelInvariantDouble $Fields[6]
            $sample.readback_ms = ConvertTo-IntelInvariantDouble $Fields[7]
            $sample.validation_ms = ConvertTo-IntelInvariantDouble $Fields[8]
            $sample.validation_inclusive_ms = ConvertTo-IntelInvariantDouble $Fields[9]
        }
        'release' {
            if ($Fields.Count -ne 5) { throw "Release sample has $($Fields.Count) columns; expected 5" }
            $sample.release_first_ms = ConvertTo-IntelInvariantDouble $Fields[3]
            $sample.release_changed_ms = ConvertTo-IntelInvariantDouble $Fields[4]
        }
        default { throw "Unknown prepared-copy phase: $phase" }
    }
    return $sample
}

function ConvertFrom-IntelPreparedCopyOutput {
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][AllowEmptyString()][string[]]$Lines,
        [Parameter(Mandatory)][int]$ExitCode,
        [Parameter(Mandatory)][ValidateSet('ordinary', 'prepared', 'host_usm_staging')][string]$TargetArm,
        [Parameter(Mandatory)][int]$ProcessSequence,
        [Parameter(Mandatory)][int]$PairIndex
    )

    $metadata = [ordered]@{}
    $samples = [System.Collections.Generic.List[object]]::new()
    try {
        foreach ($lineObject in $Lines) {
            $line = "$lineObject".Trim()
            if (-not $line) { continue }
            if ($line -match '^(cold|warm|release),') {
                $fields = @($line.Split(','))
                $samples.Add((New-IntelPreparedCopySample -Fields $fields -ProcessSequence $ProcessSequence `
                    -PairIndex $PairIndex -TargetArm $TargetArm))
            } elseif ($line -match '^([^=]+)=(.*)$') {
                $metadata[$Matches[1]] = $Matches[2]
            }
        }

        if ($ExitCode -ne 0) {
            $reason = if ($metadata.Contains('error')) { $metadata.error } else { "process_exit_$ExitCode" }
            return [ordered]@{ outcome = 'fail'; outcome_reason = $reason; metadata = $metadata; samples = @($samples) }
        }
        if (-not $metadata.Contains('schema') -or $metadata.schema -ne 'sub0.intel.prepared-copy.v1') {
            throw 'Missing or unexpected prepared-copy output schema'
        }
        if (-not $metadata.Contains('backend') -or $metadata.backend -ne 'level_zero') {
            throw 'Prepared-copy process did not report the required level_zero backend'
        }
        if (-not $metadata.Contains('status') -or $metadata.status -ne 'pass') {
            throw 'Prepared-copy process did not report status=pass'
        }
        if ($TargetArm -eq 'prepared' -and $metadata.prepared_status -eq 'unsupported') {
            $reason = if ($metadata.Contains('prepared_reason')) { $metadata.prepared_reason } else { 'prepared mode unsupported' }
            return [ordered]@{ outcome = 'unsupported'; outcome_reason = $reason; metadata = $metadata; samples = @($samples) }
        }

        $targetSamples = @($samples | Where-Object { $_.mode -eq $TargetArm })
        if (@($targetSamples | Where-Object phase -eq 'cold').Count -ne 1 -or
            @($targetSamples | Where-Object phase -eq 'warm').Count -ne 7 -or
            @($targetSamples | Where-Object phase -eq 'release').Count -ne 1) {
            throw "Target arm $TargetArm did not emit one cold, seven warm and one release sample"
        }
        $expectedTrials = @(0, 1, 2, 3, 4, 5, 6)
        $actualTrials = @($targetSamples | Where-Object phase -eq 'warm' | ForEach-Object trial)
        if (($actualTrials -join ',') -ne ($expectedTrials -join ',')) {
            throw "Target arm $TargetArm emitted unexpected warm trial indices"
        }
        return [ordered]@{ outcome = 'pass'; outcome_reason = $null; metadata = $metadata; samples = @($samples) }
    } catch {
        return [ordered]@{ outcome = 'fail'; outcome_reason = $_.Exception.Message; metadata = $metadata; samples = @($samples) }
    }
}

function Test-IntelBenchmarkResult {
    param([Parameter(Mandatory)]$Result)
    $errors = [System.Collections.Generic.List[string]]::new()
    if ($Result.schema -ne 'sub0.intel.benchmark-result.v1') { $errors.Add('Unexpected result schema') }
    if ($Result.outcome -notin @('pass', 'fail', 'unsupported')) { $errors.Add('Invalid outcome') }
    if (-not $Result.source_identity -or -not $Result.artifacts -or -not $Result.environment) {
        $errors.Add('Identity sections are required')
    }
    if (-not $Result.comparison -or -not $Result.timing_provenance) {
        $errors.Add('Comparison and timing provenance are required')
    }
    if ($Result.outcome -eq 'pass' -and @($Result.processes).Count -ne [int]$Result.comparison.planned_process_count) {
        $errors.Add('A passing result must contain every planned process')
    }
    foreach ($process in @($Result.processes)) {
        if ($process.outcome -notin @('pass', 'fail', 'unsupported')) { $errors.Add('Process has invalid outcome') }
        if (-not $process.raw_output.sha256 -or @($process.samples).Count -eq 0) {
            $errors.Add("Process $($process.sequence) lacks raw output identity or parsed samples")
        }
    }
    return @($errors)
}

Export-ModuleMember -Function Get-IntelBenchmarkSha256, Get-IntelBenchmarkSourceIdentity, `
    New-IntelBenchmarkSchedule, ConvertFrom-IntelPreparedCopyOutput, Test-IntelBenchmarkResult
