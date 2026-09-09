param(
    [string]$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI',
    [string]$CompilerVersion = '2025.3',
    [string]$LevelZeroInclude = '',
    [string]$LevelZeroLibrary = '',
    [switch]$PreparedCopyApi,
    [switch]$Run
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$output = Join-Path $repo ('out/intel-review/usm-capabilities/' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff'))
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
$env:ONEAPI_DEVICE_SELECTOR = 'level_zero:gpu'

$source = Join-Path $repo 'tools/intel_probe/usm_capabilities.cpp'
$exe = Join-Path $output 'usm-capabilities.exe'
$compileArgs = @('/nologo', '/std:c++20', '/EHsc', '-fsycl', '/Od', $source, "/Fe:$exe", "/Fo:$output/")
if ($PreparedCopyApi) { $compileArgs += '/DSUB0_PROBE_PREPARED_COPY_API' }
if ([bool]$LevelZeroInclude -ne [bool]$LevelZeroLibrary) {
    throw '-LevelZeroInclude and -LevelZeroLibrary must be supplied together'
}
$levelZeroHeader = $null
$levelZeroHeaderSha256 = $null
$levelZeroImportLibrary = $null
$levelZeroImportLibrarySha256 = $null
$levelZeroLoaderRuntimeIdentity = [ordered]@{
    status = 'not_requested'
    path = $null
    file_version = $null
    capture = $null
    source = $null
    reason = $null
}
if ($LevelZeroInclude) {
    $header = Join-Path $LevelZeroInclude 'level_zero/ze_api.h'
    if (-not (Test-Path -LiteralPath $header -PathType Leaf)) {
        throw "Level Zero header missing below: $LevelZeroInclude"
    }
    if (-not (Test-Path -LiteralPath $LevelZeroLibrary -PathType Leaf)) {
        throw "Level Zero loader import library missing: $LevelZeroLibrary"
    }
    $levelZeroHeader = (Resolve-Path -LiteralPath $header).Path
    $levelZeroHeaderSha256 = (Get-FileHash -LiteralPath $levelZeroHeader -Algorithm SHA256).Hash
    $levelZeroImportLibrary = (Resolve-Path -LiteralPath $LevelZeroLibrary).Path
    $levelZeroImportLibrarySha256 =
        (Get-FileHash -LiteralPath $levelZeroImportLibrary -Algorithm SHA256).Hash
    $levelZeroLoaderRuntimeIdentity.status = if ($Run) {
        'not_implemented_requires_probe_change'
    } else {
        'not_loaded_compile_only'
    }
    $levelZeroLoaderRuntimeIdentity.capture = 'not_implemented'
    $levelZeroLoaderRuntimeIdentity.reason =
        'Windows resolves ze_loader.dll when the probe starts; the probe does not yet report its loaded path or file version, and compile inputs are not used to guess them.'
    $compileArgs += @(
        '/DSUB0_ENABLE_LEVEL_ZERO_INVENTORY',
        "/I$((Resolve-Path -LiteralPath $LevelZeroInclude).Path)",
        $levelZeroImportLibrary
    )
}
$manifest = [ordered]@{
    schema = 'sub0.intel.usm-capabilities.v1'; utc = [DateTime]::UtcNow.ToString('o')
    repo = $repo; commit = (& git -C $repo rev-parse HEAD)
    probe_changes = @(& git -C $repo status --short -- tools/intel_probe/usm_capabilities.cpp scripts/intel/inventory/run-usm-capabilities.ps1 docs/INTEL_IGPU_USM_CAPABILITY_SPIKE.md)
    os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    compiler = $compiler; compiler_version = @(& $compiler --version); vcvars = $vcvars.FullName
    arguments = $compileArgs; source_sha256 = (Get-FileHash -LiteralPath $source).Hash
    runtime_header_sha256 = (Get-FileHash -LiteralPath (Join-Path $repo 'tools/intel_probe/runtime.hpp')).Hash
    runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath).Hash
    prepared_copy_api = [bool]$PreparedCopyApi
    level_zero_include = $LevelZeroInclude; level_zero_library = $LevelZeroLibrary
    level_zero_header = $levelZeroHeader; level_zero_header_sha256 = $levelZeroHeaderSha256
    level_zero_loader_import_library = $levelZeroImportLibrary
    level_zero_loader_import_library_sha256 = $levelZeroImportLibrarySha256
    level_zero_loader_runtime_identity = $levelZeroLoaderRuntimeIdentity
    execution_requested = [bool]$Run
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
& $compiler @compileArgs 2>&1 | Tee-Object -FilePath (Join-Path $output 'build.log')
$manifest.build_exit_code = $LASTEXITCODE
if ($LASTEXITCODE -ne 0) {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
    throw "Probe build failed; evidence: $output"
}
$manifest.executable_sha256 = (Get-FileHash -LiteralPath $exe).Hash
if ($Run) {
    & $exe 2>&1 | Tee-Object -FilePath (Join-Path $output 'probe.log')
    $manifest.probe_exit_code = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) {
        $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
        throw "Probe failed; evidence: $output"
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'manifest.json')
Write-Output "Evidence: $output"
