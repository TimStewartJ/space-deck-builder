<#
.SYNOPSIS
    Train one seed of a published training preset end to end.

.DESCRIPTION
    By default this registers a Task Scheduler launch through
    launch_training.ps1 so training runs outside the caller's process tree,
    then returns immediately. The scheduled process re-invokes this script
    with -Foreground, which:

      1. trains the first chunk from scratch with the preset's complete
         learning-rate and self-play horizons pinned;
      2. continues through chunked_resume.ps1 in fresh Python processes of at
         most the preset's chunk size, evaluating between chunks;
      3. runs the preset's locked final gauntlet on the last update.

    Preset values come from TRAINING_PRESETS in src/config.py. Re-running with
    the same -RunRoot and parameters resumes from the last validated
    checkpoint; mismatched parameters are rejected.

.PARAMETER Updates
    Override the preset's final update, for smoke runs.

.PARAMETER ChunkSize
    Override the preset's fresh-process chunk size, for smoke runs.

.EXAMPLE
    .\scripts\train_reference.ps1 -Seed 0

.EXAMPLE
    .\scripts\train_reference.ps1 -Seed 0 -Updates 2 -ChunkSize 1 `
        -Episodes 64 -InterimEvalGames 6 -FinalEvalGames 12 `
        -NumWorkers 1 -EvalNumWorkers 1 -RunRoot logs\reference\smoke
#>
[CmdletBinding()]
param(
    [int] $Seed = 0,
    [string] $Preset = "reference-selfplay",
    [string] $RunRoot = "",
    [string] $Name = "",
    [string] $Python = "",
    [int] $Updates = 0,
    [int] $ChunkSize = 0,
    [int] $Episodes = 0,
    [int] $InterimEvalGames = 0,
    [int] $FinalEvalGames = 0,
    [int] $NumWorkers = 0,
    [int] $EvalNumWorkers = 0,
    [string] $ProcessorAffinityMask = "Auto",
    [switch] $Foreground
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($Seed -lt 0) {
    throw "Seed must be non-negative, got $Seed"
}
if ($Preset -cnotmatch "^[a-z0-9-]+$") {
    throw "Preset names contain only lowercase letters, digits, and hyphens."
}
if (-not $RunRoot) {
    $RunRoot = Join-Path $Repo "logs\reference\${Preset}_seed$Seed"
}
if (-not [System.IO.Path]::IsPathRooted($RunRoot)) {
    $RunRoot = Join-Path $Repo $RunRoot
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
foreach ($pair in @(
    @("Updates", $Updates), @("ChunkSize", $ChunkSize),
    @("Episodes", $Episodes), @("InterimEvalGames", $InterimEvalGames),
    @("FinalEvalGames", $FinalEvalGames), @("NumWorkers", $NumWorkers),
    @("EvalNumWorkers", $EvalNumWorkers)
)) {
    if ($pair[1] -lt 0) {
        throw "$($pair[0]) must be non-negative, got $($pair[1])"
    }
}

if (-not $Foreground) {
    if (-not $Name) {
        $Name = "sdb_reference_${Preset}_seed$Seed".Replace("-", "_")
    }
    $overrides = @()
    foreach ($pair in @(
        @("Updates", $Updates), @("ChunkSize", $ChunkSize),
        @("Episodes", $Episodes), @("InterimEvalGames", $InterimEvalGames),
        @("FinalEvalGames", $FinalEvalGames), @("NumWorkers", $NumWorkers),
        @("EvalNumWorkers", $EvalNumWorkers)
    )) {
        if ($pair[1] -gt 0) {
            $overrides += "-$($pair[0]) $($pair[1])"
        }
    }
    $scriptPath = Join-Path $PSScriptRoot "train_reference.ps1"
    $command = (
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -File " +
        "`"$scriptPath`" -Foreground -Python {python} -Seed $Seed " +
        "-Preset $Preset -RunRoot `"$RunRoot`" " +
        "-ProcessorAffinityMask $ProcessorAffinityMask " +
        ($overrides -join " ")
    ).TrimEnd()
    $launchArgs = @{
        Name = $Name
        Command = $command
        Repo = $Repo
        ProcessorAffinityMask = $ProcessorAffinityMask
        # A finished, failed, or stopped launch leaves its task registered.
        # Replacing it lets the same command resume; the launcher still
        # refuses to replace a running task.
        ReplaceExisting = $true
    }
    if ($Python) {
        $launchArgs.Python = $Python
    }
    & (Join-Path $PSScriptRoot "launch_training.ps1") @launchArgs
    Write-Host "  run root: $RunRoot"
    exit 0
}

if (-not $Python) {
    $Python = Join-Path $Repo ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Invoke-RepoPython([string[]] $Arguments) {
    Push-Location $Repo
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $Python @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
        Pop-Location
    }
    if ($code -ne 0) {
        throw (
            "python $($Arguments -join ' ') failed: " +
            ($output -join [Environment]::NewLine)
        )
    }
    return ($output -join [Environment]::NewLine)
}

function Assert-CheckpointUpdate([string] $Checkpoint, [int] $Expected) {
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Checkpoint does not exist: $Checkpoint"
    }
    $validator = (
        "import sys; from src.config import load_checkpoint; " +
        "c = load_checkpoint(sys.argv[1], map_location='cpu'); " +
        "u = int(c.get('update') or 0); " +
        "sys.exit(0 if u == int(sys.argv[2]) else " +
        "f'checkpoint update {u} != expected {sys.argv[2]}')"
    )
    Invoke-RepoPython @("-c", $validator, $Checkpoint, [string]$Expected) |
        Out-Null
    return (Resolve-Path -LiteralPath $Checkpoint).Path
}

function Save-Manifest([object] $Manifest, [string] $Path) {
    $temporary = "$Path.tmp"
    $Manifest | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$presetJson = Invoke-RepoPython @(
    "-c",
    (
        "import json, sys; from src.config import get_training_preset; " +
        "print(json.dumps(get_training_preset(sys.argv[1]).to_dict()))"
    ),
    $Preset
)
$presetInfo = $presetJson | ConvertFrom-Json
$targetUpdate = if ($Updates) { $Updates } else {
    [int]$presetInfo.train_defaults.updates
}
$chunk = if ($ChunkSize) { $ChunkSize } else { [int]$presetInfo.chunk_size }
$interimGames = if ($InterimEvalGames) { $InterimEvalGames } else {
    [int]$presetInfo.interim_eval_games
}
$finalGames = if ($FinalEvalGames) { $FinalEvalGames } else {
    [int]$presetInfo.final_eval_games
}
$evalWorkers = if ($EvalNumWorkers) { $EvalNumWorkers } else {
    [int]$presetInfo.eval_num_workers
}
if ($targetUpdate -le 0 -or $chunk -le 0) {
    throw "Updates and ChunkSize must be positive"
}

$commonArgs = @("--preset", $Preset, "--seed", [string]$Seed)
if ($Episodes) {
    $commonArgs += @("--episodes", [string]$Episodes)
}
if ($NumWorkers) {
    $commonArgs += @("--num-workers", [string]$NumWorkers)
}
$manifestPath = Join-Path $RunRoot "reference_manifest.json"
$bootstrapTarget = [math]::Min($chunk, $targetUpdate)
$bootstrapIsFinal = $bootstrapTarget -eq $targetUpdate
$bootstrapGames = if ($bootstrapIsFinal) { $finalGames } else { $interimGames }
$invocation = [PSCustomObject]@{
    preset = $Preset
    seed = $Seed
    target_update = $targetUpdate
    chunk_size = $chunk
    interim_eval_games = $interimGames
    final_eval_games = $finalGames
    eval_num_workers = $evalWorkers
    common_args = @($commonArgs)
}
$invocationJson = $invocation | ConvertTo-Json -Depth 5 -Compress

if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $existing = $manifest.invocation | ConvertTo-Json -Depth 5 -Compress
    if ($existing -ne $invocationJson) {
        throw (
            "Existing reference manifest does not match this invocation. " +
            "Use a new -RunRoot. Existing: $existing Requested: $invocationJson"
        )
    }
    if ([string]$manifest.state -eq "complete") {
        Write-Host "[train_reference] already complete: $($manifest.final_checkpoint)"
        exit 0
    }
    $manifest.state = "running"
    $manifest.failure = $null
} else {
    $manifest = [PSCustomObject]@{
        schema_version = 1
        state = "running"
        invocation = $invocation
        repository = $Repo
        python = $Python
        started_at = (Get-Date).ToString("o")
        finished_at = $null
        failure = $null
        bootstrap = $null
        chunks_manifest = $null
        final_checkpoint = $null
    }
}
Save-Manifest $manifest $manifestPath

try {
    $bootstrapCheckpoint = $null
    if ($manifest.bootstrap -and [string]$manifest.bootstrap.state -eq "complete") {
        $bootstrapCheckpoint = Assert-CheckpointUpdate `
            ([string]$manifest.bootstrap.checkpoint) $bootstrapTarget
        Write-Host "[train_reference] bootstrap already complete: $bootstrapCheckpoint"
    } else {
        $bootstrapModels = Join-Path $RunRoot "bootstrap_models"
        New-Item -ItemType Directory -Force -Path $bootstrapModels | Out-Null
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $log = Join-Path $RunRoot "bootstrap_0_to_${bootstrapTarget}_$stamp.log"
        $errorLog = "$log.err"
        $arguments = @("-u", "-m", "src", "train") + $commonArgs + @(
            "--models-dir", $bootstrapModels,
            "--updates", [string]$bootstrapTarget,
            "--lr-horizon", [string]$targetUpdate,
            "--self-play-horizon", [string]$targetUpdate,
            "--eval-games", [string]$bootstrapGames,
            "--eval-num-workers", [string]$evalWorkers
        )
        $manifest.bootstrap = [PSCustomObject]@{
            state = "running"
            target_update = $bootstrapTarget
            log = $log
            error_log = $errorLog
            checkpoint = $null
            exit_code = $null
            started_at = (Get-Date).ToString("o")
            finished_at = $null
        }
        Save-Manifest $manifest $manifestPath
        Write-Host "[train_reference] bootstrap 0 -> $bootstrapTarget; log=$log"
        Push-Location $Repo
        $previousErrorAction = $ErrorActionPreference
        try {
            # Windows PowerShell 5.1 turns redirected native stderr into
            # terminating errors under "Stop"; keep the child running and its
            # warnings or traceback in the error log.
            $ErrorActionPreference = "Continue"
            & $Python @arguments 1> $log 2> $errorLog
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorAction
            Pop-Location
        }
        $manifest.bootstrap.exit_code = $exitCode
        $manifest.bootstrap.finished_at = (Get-Date).ToString("o")
        if ($exitCode -ne 0) {
            $manifest.bootstrap.state = "failed"
            throw "bootstrap exited with code $exitCode; see $errorLog"
        }
        $saved = @(
            Select-String -LiteralPath $log -Pattern "Checkpoint saved:\s+(.+)$"
        )
        $candidate = $null
        for ($index = $saved.Count - 1; $index -ge 0; $index--) {
            $path = $saved[$index].Matches[0].Groups[1].Value.Trim()
            if (-not [System.IO.Path]::IsPathRooted($path)) {
                $path = Join-Path $Repo $path
            }
            if ((Split-Path $path -Leaf) -match "upd${bootstrapTarget}_wins") {
                $candidate = [System.IO.Path]::GetFullPath($path)
                break
            }
        }
        if (-not $candidate) {
            $manifest.bootstrap.state = "failed"
            throw "bootstrap log did not report a checkpoint for update $bootstrapTarget"
        }
        $bootstrapCheckpoint = Assert-CheckpointUpdate $candidate $bootstrapTarget
        $manifest.bootstrap.checkpoint = $bootstrapCheckpoint
        $manifest.bootstrap.state = "complete"
        Save-Manifest $manifest $manifestPath
    }

    if ($bootstrapIsFinal) {
        $finalCheckpoint = $bootstrapCheckpoint
    } else {
        $chunksDir = Join-Path $RunRoot "chunks"
        $chunkArgs = $commonArgs + @(
            "--eval-games", [string]$finalGames,
            "--eval-num-workers", [string]$evalWorkers
        )
        $manifest.chunks_manifest = Join-Path $chunksDir "manifest.json"
        Save-Manifest $manifest $manifestPath
        & (Join-Path $PSScriptRoot "chunked_resume.ps1") `
            -StartCheckpoint $bootstrapCheckpoint `
            -TargetUpdate $targetUpdate `
            -ChunkSize $chunk `
            -Tag ("reference_${Preset}_seed$Seed".Replace("-", "_")) `
            -ExtraArgs $chunkArgs `
            -InterimEvalGames $interimGames `
            -InterimNumWorkers $evalWorkers `
            -LogsDir $chunksDir `
            -WorkingDir $Repo `
            -Python $Python `
            -ProcessorAffinityMask $ProcessorAffinityMask
        $chunkManifest = Get-Content `
            -LiteralPath $manifest.chunks_manifest -Raw | ConvertFrom-Json
        if ([string]$chunkManifest.state -ne "complete") {
            throw "chunked resume did not complete: $($chunkManifest.failure)"
        }
        $finalCheckpoint = Assert-CheckpointUpdate `
            ([string]$chunkManifest.final_checkpoint) $targetUpdate
    }

    $manifest.final_checkpoint = $finalCheckpoint
    $manifest.state = "complete"
    $manifest.finished_at = (Get-Date).ToString("o")
    Save-Manifest $manifest $manifestPath
    Write-Host "[train_reference] complete: $finalCheckpoint"
} catch {
    $manifest.state = "failed"
    $manifest.failure = $_.Exception.Message
    $manifest.finished_at = (Get-Date).ToString("o")
    Save-Manifest $manifest $manifestPath
    throw
}
