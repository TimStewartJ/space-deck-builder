# Two-phase hp3 experiment validating the new --resume feature:
#   Phase 1: train hp3 (attn/attn, clip-eps=0.3, entropy=0.01) FRESH for 50 updates
#   Phase 2: --resume from phase 1's final checkpoint, train 50 more updates
#
# End state: a hp3@100 checkpoint produced via resume (validates the feature
# in real conditions). Compare to sum_mlp@100 to answer "does hp3 architecture
# beat sum at full compute budget?"
#
# Each phase writes its own log + manifest entry. The orchestrator records
# the phase-1 final checkpoint path so phase 2 can reference it for --resume.
#
# Usage: pwsh -File scripts/hp3_full_resume.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$py = "E:\resume-training\.venv\Scripts\python.exe"
$logDir = "E:\resume-training\logs"
$manifestFile = "$logDir\hp3_resume_manifest.json"

$commonArgs = @(
    "-u", "-m", "src", "train",
    "--pool-type", "attention",
    "--actor-type", "attention",
    "--clip-eps", "0.3",
    "--entropy", "0.01",
    "--episodes", "16000",
    "--num-workers", "16",
    "--self-play",
    "--eval-every", "20",
    "--updates", "50"
)

function Read-Manifest {
    if (Test-Path $manifestFile) {
        return Get-Content $manifestFile -Raw | ConvertFrom-Json
    }
    return @{ phases = @() }
}

function Write-Manifest($m) {
    $m | ConvertTo-Json -Depth 10 | Set-Content $manifestFile
}

function Get-PhaseEntry($manifest, $phaseKey) {
    foreach ($p in $manifest.phases) {
        if ($p.key -eq $phaseKey) { return $p }
    }
    return $null
}

function Add-PhaseEntry($manifest, $entry) {
    # Convert to a typed array so PowerShell doesn't squash single-item arrays
    $existing = @($manifest.phases | Where-Object { $_.key -ne $entry.key })
    $manifest.phases = @($existing) + @($entry)
}

function Run-Phase {
    param(
        [string]$key,
        [string[]]$extraArgs
    )

    $manifest = Read-Manifest
    $existing = Get-PhaseEntry $manifest $key
    if ($existing -and $existing.completed -eq $true -and (Test-Path $existing.final_checkpoint)) {
        Write-Host "[$key] Already completed (checkpoint: $($existing.final_checkpoint)); skipping."
        return $existing.final_checkpoint
    }

    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = "$logDir\hp3_resume_${key}_$ts.log"

    Write-Host "[$key] Launching: $py $($commonArgs + $extraArgs -join ' ')"
    Write-Host "[$key] Log: $log"

    $allArgs = $commonArgs + $extraArgs
    $proc = Start-Process -FilePath $py -ArgumentList $allArgs `
        -WorkingDirectory "E:\resume-training" -WindowStyle Hidden `
        -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"

    $started = (Get-Date).ToString("o")
    Write-Host "[$key] PID=$($proc.Id) started at $started"

    # Track the modelfiles before we start so we can identify the new ones
    $modelsBefore = @{}
    foreach ($f in (Get-ChildItem E:\resume-training\models\*.pth -ErrorAction SilentlyContinue)) {
        $modelsBefore[$f.FullName] = $true
    }

    # Block until process exits, polling every 60s with a status print
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        $tail = Get-Content $log -Tail 1 -ErrorAction SilentlyContinue
        Write-Host "[$key] (still running, PID=$($proc.Id)): $tail"
    }
    # Force ExitCode refresh; Start-Process sometimes leaves it null
    $proc.WaitForExit(5000) | Out-Null
    $exitCode = if ($null -eq $proc.ExitCode) { 0 } else { $proc.ExitCode }
    $ended = (Get-Date).ToString("o")
    Write-Host "[$key] Exited with code $exitCode at $ended"

    # Find the final checkpoint produced by this phase (latest upd50 written
    # after start, or the highest-update file written by this run).
    $candidates = Get-ChildItem E:\resume-training\models\*.pth | Where-Object {
        -not $modelsBefore.ContainsKey($_.FullName)
    } | Sort-Object @{ Expression = {
        if ($_.Name -match 'upd(\d+)_') { [int]$Matches[1] } else { 0 }
    }; Descending = $true }, LastWriteTime -Descending

    if ($candidates.Count -eq 0) {
        Write-Host "[$key] ERROR: no new checkpoints written"
        return $null
    }
    $finalCheckpoint = $candidates[0].FullName
    Write-Host "[$key] Final checkpoint: $finalCheckpoint"

    # Update manifest
    $manifest = Read-Manifest
    $entry = [PSCustomObject]@{
        key                = $key
        started_at         = $started
        ended_at           = $ended
        wall_s             = ((Get-Date $ended) - (Get-Date $started)).TotalSeconds
        exit_code          = $exitCode
        log                = $log
        final_checkpoint   = $finalCheckpoint
        extra_args         = $extraArgs
        completed          = ($exitCode -eq 0)
    }
    Add-PhaseEntry $manifest $entry
    Write-Manifest $manifest

    return $finalCheckpoint
}

# --- Phase 1: fresh hp3 to upd 50 ---
$phase1Final = Run-Phase -key "phase1_fresh_hp3_to_50" -extraArgs @()

if (-not $phase1Final) {
    Write-Host "Phase 1 failed; aborting."
    exit 1
}

# --- Phase 2: resume to upd 100 ---
$phase2Final = Run-Phase -key "phase2_resume_to_100" -extraArgs @("--resume", $phase1Final)

if (-not $phase2Final) {
    Write-Host "Phase 2 failed."
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "Both phases complete."
Write-Host "  Phase 1 (fresh upd1-50):   $phase1Final"
Write-Host "  Phase 2 (resume upd51-100): $phase2Final"
Write-Host "========================================"
