# Run one training arc in fresh-process chunks while preserving resume state.
#
# Launch this script outside the Copilot Bridge process tree (for example,
# through Task Scheduler). Each chunk starts a new Python process, resumes the
# latest checkpoint, and pins the learning-rate horizon to TargetUpdate.
#
# Example:
#   .\scripts\chunked_resume.ps1 `
#       -StartCheckpoint "models\ppo_agent_..._upd213_wins-1.pth" `
#       -TargetUpdate 263 `
#       -ChunkSize 25 `
#       -Tag "selfplay_cap1" `
#       -ExtraArgs @("--engine","vec","--episodes","2000","--self-play",`
#                    "--self-play-schedule","constant","--self-play-ratio","0.5",`
#                    "--pfsp","variance","--vec-max-active-snapshots","1",`
#                    "--opponents","random,heuristic,simple","--num-workers","1",`
#                    "--vec-threads","4","--seed","0","--eval-games","3200")

param(
    [Parameter(Mandatory = $true)] [string] $StartCheckpoint,
    [Parameter(Mandatory = $true)] [int] $TargetUpdate,
    [Parameter(Mandatory = $true)] [string] $Tag,
    [int] $ChunkSize = 25,
    [int] $LrHorizon = 0,
    [int] $SelfPlayHorizon = 0,
    [string[]] $ExtraArgs = @(),
    [int] $InterimEvalGames = 6,
    [string] $LogsDir = "",
    [string] $WorkingDir = "",
    [string] $Python = ""
)

$ErrorActionPreference = "Stop"

if (-not $WorkingDir) {
    $WorkingDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if (-not $LogsDir) {
    $LogsDir = Join-Path $WorkingDir "results\chunked_training\$Tag"
}
if (-not [System.IO.Path]::IsPathRooted($LogsDir)) {
    $LogsDir = Join-Path $WorkingDir $LogsDir
}
$LogsDir = [System.IO.Path]::GetFullPath($LogsDir)
$ModelsDir = Join-Path $LogsDir "models"
if (-not $Python) {
    $Python = Join-Path $WorkingDir ".venv\Scripts\python.exe"
}
if (-not [System.IO.Path]::IsPathRooted($StartCheckpoint)) {
    $StartCheckpoint = Join-Path $WorkingDir $StartCheckpoint
}

if ($ChunkSize -le 0) {
    throw "ChunkSize must be positive, got $ChunkSize"
}
if ($TargetUpdate -le 0) {
    throw "TargetUpdate must be positive, got $TargetUpdate"
}
if ($InterimEvalGames -le 0) {
    throw "InterimEvalGames must be positive, got $InterimEvalGames"
}
if ($LrHorizon -eq 0) {
    $LrHorizon = $TargetUpdate
}
if ($SelfPlayHorizon -eq 0) {
    $SelfPlayHorizon = $TargetUpdate
}
if ($LrHorizon -lt $TargetUpdate) {
    throw "LrHorizon must be at least TargetUpdate"
}
if ($SelfPlayHorizon -lt $TargetUpdate) {
    throw "SelfPlayHorizon must be at least TargetUpdate"
}
foreach (
    $reserved in @(
        "--resume",
        "--updates",
        "--lr-horizon",
        "--self-play-horizon",
        "--models-dir"
    )
) {
    $hasEqualsForm = @(
        $ExtraArgs | Where-Object {
            $_.StartsWith(
                "$reserved=",
                [System.StringComparison]::OrdinalIgnoreCase
            )
        }
    ).Count -gt 0
    if (($ExtraArgs -contains $reserved) -or $hasEqualsForm) {
        throw "ExtraArgs must not contain orchestrator-owned option $reserved"
    }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
New-Item -ItemType Directory -Force -Path $LogsDir, $ModelsDir | Out-Null

function Get-CheckpointUpdate([string] $Checkpoint) {
    if ((Split-Path $Checkpoint -Leaf) -match "upd(\d+)_wins") {
        return [int]$Matches[1]
    }
    throw "Could not parse update number from checkpoint: $Checkpoint"
}

function Assert-CheckpointUpdate(
    [string] $Checkpoint,
    [int] $ExpectedUpdate
) {
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Checkpoint does not exist: $Checkpoint"
    }
    $resolved = (Resolve-Path -LiteralPath $Checkpoint).Path
    $validator = @"
import sys
from src.config import load_checkpoint
checkpoint = load_checkpoint(sys.argv[1], map_location='cpu')
actual = int(checkpoint.get('update') or 0)
expected = int(sys.argv[2])
if actual != expected:
    raise RuntimeError(f'checkpoint update {actual} != expected {expected}')
"@
    Push-Location $WorkingDir
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $validationOutput = & $Python -c $validator `
            $resolved ([string]$ExpectedUpdate) 2>&1
        $validationExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
        Pop-Location
    }
    if ($validationExit -ne 0) {
        throw (
            "Checkpoint validation failed for $resolved`: " +
            ($validationOutput -join [Environment]::NewLine)
        )
    }
    return $resolved
}

function Get-LoggedCheckpoint(
    [string] $Log,
    [int] $ExpectedUpdate
) {
    $lines = @(
        Select-String `
            -LiteralPath $Log `
            -Pattern "Checkpoint saved:\s+(.+)$"
    )
    for ($index = $lines.Count - 1; $index -ge 0; $index--) {
        $candidate = $lines[$index].Matches[0].Groups[1].Value.Trim()
        if (-not [System.IO.Path]::IsPathRooted($candidate)) {
            $candidate = Join-Path $WorkingDir $candidate
        }
        $candidate = [System.IO.Path]::GetFullPath($candidate)
        if ((Get-CheckpointUpdate $candidate) -eq $ExpectedUpdate) {
            $modelRoot = [System.IO.Path]::GetFullPath($ModelsDir).TrimEnd(
                [System.IO.Path]::DirectorySeparatorChar
            ) + [System.IO.Path]::DirectorySeparatorChar
            if (
                -not $candidate.StartsWith(
                    $modelRoot,
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            ) {
                throw "Logged checkpoint is outside isolated models dir: $candidate"
            }
            return $candidate
        }
    }
    return $null
}

function Save-Manifest([object] $Manifest, [string] $Path) {
    $temporary = "$Path.tmp"
    $Manifest | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Set-ManifestProperty(
    [object] $Manifest,
    [string] $Name,
    [object] $Value
) {
    if ($Manifest.PSObject.Properties[$Name]) {
        $Manifest.$Name = $Value
    } else {
        $Manifest | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

$manifestPath = Join-Path $LogsDir "manifest.json"
$startUpdate = Get-CheckpointUpdate $StartCheckpoint
if ($startUpdate -ge $TargetUpdate) {
    throw (
        "Start checkpoint update $startUpdate must be below " +
        "TargetUpdate $TargetUpdate"
    )
}
$startCheckpoint = [System.IO.Path]::GetFullPath($StartCheckpoint)

if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw |
        ConvertFrom-Json
    $expectedExtraArgs = ConvertTo-Json -InputObject @($ExtraArgs) -Compress
    $actualExtraArgs = ConvertTo-Json `
        -InputObject @($manifest.extra_args) `
        -Compress
    $manifestLrHorizon = if (
        $manifest.PSObject.Properties["lr_horizon"]
    ) {
        [int]$manifest.lr_horizon
    } else {
        [int]$manifest.target_update
    }
    $manifestSelfPlayHorizon = if (
        $manifest.PSObject.Properties["self_play_horizon"]
    ) {
        [int]$manifest.self_play_horizon
    } else {
        [int]$manifest.target_update
    }
    $manifestStart = [System.IO.Path]::GetFullPath(
        [string]$manifest.start_checkpoint
    )
    $expectedStart = [System.IO.Path]::GetFullPath($startCheckpoint)
    $mismatches = @()
    if ([int]$manifest.schema_version -ne 1) {
        $mismatches += "schema_version"
    }
    if ([string]$manifest.tag -ne $Tag) {
        $mismatches += "tag"
    }
    if ([int]$manifest.target_update -ne $TargetUpdate) {
        $mismatches += "target_update"
    }
    if ([int]$manifest.chunk_size -ne $ChunkSize) {
        $mismatches += "chunk_size"
    }
    if ([int]$manifest.interim_eval_games -ne $InterimEvalGames) {
        $mismatches += "interim_eval_games"
    }
    if ($manifestLrHorizon -ne $LrHorizon) {
        $mismatches += "lr_horizon"
    }
    if ($manifestSelfPlayHorizon -ne $SelfPlayHorizon) {
        $mismatches += "self_play_horizon"
    }
    if ($manifestStart -ne $expectedStart) {
        $mismatches += "start_checkpoint"
    }
    if (
        [System.IO.Path]::GetFullPath(
            [string]$manifest.working_directory
        ) -ne [System.IO.Path]::GetFullPath($WorkingDir)
    ) {
        $mismatches += "working_directory"
    }
    if (
        [System.IO.Path]::GetFullPath(
            [string]$manifest.models_directory
        ) -ne [System.IO.Path]::GetFullPath($ModelsDir)
    ) {
        $mismatches += "models_directory"
    }
    if ($actualExtraArgs -ne $expectedExtraArgs) {
        $mismatches += "extra_args"
    }
    if ($mismatches.Count) {
        throw (
            "Existing manifest does not match this invocation: " +
            ($mismatches -join ", ")
        )
    }
    Set-ManifestProperty $manifest "lr_horizon" $manifestLrHorizon
    Set-ManifestProperty `
        $manifest `
        "self_play_horizon" `
        $manifestSelfPlayHorizon
    if ([string]$manifest.state -eq "complete") {
        $finalCheckpoint = Assert-CheckpointUpdate `
            ([string]$manifest.final_checkpoint) `
            $TargetUpdate
        Write-Host (
            "[chunked_resume:$Tag] already complete: $finalCheckpoint"
        )
        exit 0
    }
    $currentUpdate = [int]$manifest.current_update
    if ($currentUpdate -lt $startUpdate -or $currentUpdate -gt $TargetUpdate) {
        throw "Manifest current update is invalid: $currentUpdate"
    }
    $currentCheckpoint = Assert-CheckpointUpdate `
        ([string]$manifest.current_checkpoint) `
        $currentUpdate
    if ($manifest.current_chunk) {
        $interrupted = [PSCustomObject]@{
            state = "interrupted"
            from_update = [int]$manifest.current_chunk.from_update
            target_update = [int]$manifest.current_chunk.target_update
            from_checkpoint = $currentCheckpoint
            candidate_checkpoint = $null
            log = [string]$manifest.current_chunk.log
            error_log = [string]$manifest.current_chunk.error_log
            metrics = $null
            exit_code = $null
            started_at = [string]$manifest.current_chunk.started_at
            finished_at = (Get-Date).ToString("o")
        }
        $manifest.chunks = @($manifest.chunks) + @($interrupted)
    }
    Set-ManifestProperty $manifest "state" "running"
    Set-ManifestProperty $manifest "failure" $null
    Set-ManifestProperty $manifest "finished_at" $null
    Set-ManifestProperty $manifest "current_chunk" $null
    Set-ManifestProperty `
        $manifest `
        "last_resumed_at" `
        (Get-Date).ToString("o")
} else {
    $startCheckpoint = Assert-CheckpointUpdate `
        $startCheckpoint `
        $startUpdate
    $currentCheckpoint = $startCheckpoint
    $currentUpdate = $startUpdate
    $manifest = [PSCustomObject]@{
        schema_version = 1
        state = "running"
        tag = $Tag
        target_update = $TargetUpdate
        chunk_size = $ChunkSize
        interim_eval_games = $InterimEvalGames
        lr_horizon = $LrHorizon
        self_play_horizon = $SelfPlayHorizon
        start_checkpoint = $currentCheckpoint
        current_checkpoint = $currentCheckpoint
        current_update = $currentUpdate
        working_directory = $WorkingDir
        models_directory = $ModelsDir
        extra_args = @($ExtraArgs)
        started_at = (Get-Date).ToString("o")
        last_resumed_at = $null
        finished_at = $null
        current_chunk = $null
        failure = $null
        source_final_checkpoint = $null
        final_checkpoint = $null
        chunks = @()
    }
}
Save-Manifest $manifest $manifestPath

Write-Host (
    "[chunked_resume:$Tag] start=$currentCheckpoint " +
    "current=$currentUpdate target=$TargetUpdate chunk=$ChunkSize"
)

while ($currentUpdate -lt $TargetUpdate) {
    $remaining = $TargetUpdate - $currentUpdate
    $thisChunk = [math]::Min($ChunkSize, $remaining)
    $chunkTarget = $currentUpdate + $thisChunk
    $isFinalChunk = $chunkTarget -eq $TargetUpdate
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $LogsDir "upd${currentUpdate}_to_${chunkTarget}_$timestamp.log"
    $errorLog = "$log.err"

    $arguments = @(
        "-u", "-m", "src", "train"
    ) + $ExtraArgs + @(
        "--models-dir", $ModelsDir,
        "--resume", $currentCheckpoint,
        "--updates", [string]$thisChunk,
        "--lr-horizon", [string]$LrHorizon,
        "--self-play-horizon", [string]$SelfPlayHorizon
    )
    if (-not $isFinalChunk) {
        $arguments += @("--eval-games", [string]$InterimEvalGames)
    }

    Write-Host (
        "[chunked_resume:$Tag] chunk start: " +
        "$currentUpdate -> $chunkTarget; log=$log"
    )
    $startedAt = Get-Date
    $manifest.current_chunk = @{
        from_update = $currentUpdate
        target_update = $chunkTarget
        log = $log
        error_log = $errorLog
        started_at = $startedAt.ToString("o")
    }
    Save-Manifest $manifest $manifestPath
    Push-Location $WorkingDir
    try {
        & $Python @arguments 1> $log 2> $errorLog
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($null -eq $exitCode) {
        $exitCode = -1
    }
    $candidateCheckpoint = $null
    $validatedCheckpoint = $null
    $failureReason = $null
    try {
        $candidateCheckpoint = Get-LoggedCheckpoint $log $chunkTarget
    } catch {
        $failureReason = $_.Exception.Message
    }
    if ($exitCode -ne 0) {
        $failureReason = "chunk process exited with code $exitCode"
    } elseif (-not $candidateCheckpoint) {
        $failureReason = (
            "child log did not report a checkpoint for update $chunkTarget"
        )
    } elseif (-not $failureReason) {
        try {
            $validatedCheckpoint = Assert-CheckpointUpdate `
                $candidateCheckpoint `
                $chunkTarget
        } catch {
            $failureReason = $_.Exception.Message
        }
    }
    $newUpdate = if ($validatedCheckpoint) {
        $chunkTarget
    } else {
        $currentUpdate
    }
    $metricsCopy = $null
    $metricsLine = Select-String `
        -LiteralPath $log `
        -Pattern "Metrics file:\s+(.+)$" |
        Select-Object -First 1
    if ($metricsLine -and $metricsLine.Matches.Count) {
        $metricsPath = $metricsLine.Matches[0].Groups[1].Value.Trim()
        if (-not [System.IO.Path]::IsPathRooted($metricsPath)) {
            $metricsPath = Join-Path $WorkingDir $metricsPath
        }
        if (Test-Path -LiteralPath $metricsPath -PathType Leaf) {
            $metricsCopy = Join-Path $LogsDir (
                "metrics_upd${currentUpdate}_to_${newUpdate}_$timestamp.jsonl"
            )
            Copy-Item -LiteralPath $metricsPath -Destination $metricsCopy
        }
    }

    $chunkRecord = [PSCustomObject]@{
        state = if ($failureReason) { "failed" } else { "complete" }
        from_update = $currentUpdate
        target_update = $chunkTarget
        actual_update = $newUpdate
        from_checkpoint = $currentCheckpoint
        candidate_checkpoint = $candidateCheckpoint
        new_checkpoint = $validatedCheckpoint
        log = $log
        error_log = $errorLog
        metrics = $metricsCopy
        exit_code = $exitCode
        started_at = $startedAt.ToString("o")
        finished_at = (Get-Date).ToString("o")
    }
    $manifest.chunks = @($manifest.chunks) + @($chunkRecord)
    $manifest.current_chunk = $null

    if ($failureReason) {
        $manifest.state = "failed"
        $manifest.finished_at = (Get-Date).ToString("o")
        $manifest.failure = $failureReason
        Save-Manifest $manifest $manifestPath
        throw $manifest.failure
    }

    $currentCheckpoint = $validatedCheckpoint
    $currentUpdate = $chunkTarget
    $manifest.current_checkpoint = $currentCheckpoint
    $manifest.current_update = $currentUpdate
    Save-Manifest $manifest $manifestPath
    Write-Host "[chunked_resume:$Tag] chunk complete: update $currentUpdate"
}

$manifest.state = "complete"
$manifest.finished_at = (Get-Date).ToString("o")
$manifest.source_final_checkpoint = $currentCheckpoint
$manifest.final_checkpoint = Join-Path $LogsDir "final_checkpoint.pth"
Copy-Item `
    -LiteralPath $currentCheckpoint `
    -Destination $manifest.final_checkpoint `
    -Force
$manifest.final_checkpoint = Assert-CheckpointUpdate `
    $manifest.final_checkpoint `
    $TargetUpdate
Save-Manifest $manifest $manifestPath
Write-Host "[chunked_resume:$Tag] complete: $currentCheckpoint"
