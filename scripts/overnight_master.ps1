# Overnight master orchestrator.
#
# Sequential, idempotent, manifest-driven workflow. Each phase records its
# status to a JSON manifest so the orchestrator can be safely restarted —
# completed phases are skipped on re-run.
#
# Phases:
#   0. wait_for_sum_baseline    — block until sum_mlp@100 baseline checkpoint exists
#   1. safeguard_sum_100        — copy to master models/ for safety
#   2. showdown_tournament_v1   — hp3@50, hp3@100, sum@100, random/heuristic/simple
#   3. hp3_resume_100_to_200    — --resume from hp3@100, train 100 more updates
#   4. safeguard_hp3_200        — copy hp3@200 to master models/
#   5. sum_resume_100_to_200    — --resume from sum@100, train 100 more updates
#   6. safeguard_sum_200        — copy sum@200 to master models/
#   7. final_tournament         — hp3@100, hp3@200, sum@100, sum@200, builtins (1500 g/p)
#   8. summary_report           — write a markdown morning report
#
# Status is also written to overnight_master_status.json (current_phase + start time)
# so the cron monitor can give meaningful health updates.

$ErrorActionPreference = "Stop"
Set-Location E:\resume-training

$py = "E:\resume-training\.venv\Scripts\python.exe"
$logDir = "E:\resume-training\logs"
$manifestFile = "$logDir\overnight_master_manifest.json"
$statusFile = "$logDir\overnight_master_status.json"
$masterModelsDir = "E:\space-deck-builder\models"

if (-not (Test-Path $masterModelsDir)) {
    New-Item -ItemType Directory -Path $masterModelsDir | Out-Null
}

# ---------- helpers ----------

function Read-Manifest {
    if (Test-Path $manifestFile) {
        return Get-Content $manifestFile -Raw | ConvertFrom-Json
    }
    return @{ phases = @(); started_at = (Get-Date).ToString("o") }
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
    $existing = @($manifest.phases | Where-Object { $_.key -ne $entry.key })
    $manifest.phases = @($existing) + @($entry)
}

function Set-Status($phase, $detail) {
    $s = @{
        current_phase = $phase
        detail        = $detail
        updated_at    = (Get-Date).ToString("o")
    }
    $s | ConvertTo-Json -Depth 5 | Set-Content $statusFile
}

function Skip-If-Done($phaseKey) {
    $manifest = Read-Manifest
    $existing = Get-PhaseEntry $manifest $phaseKey
    if ($existing -and $existing.completed -eq $true) {
        Write-Host "[$phaseKey] Already complete; skipping."
        return $existing
    }
    return $null
}

function Mark-Done($phaseKey, $extra) {
    $manifest = Read-Manifest
    $entry = [PSCustomObject]@{
        key          = $phaseKey
        completed    = $true
        completed_at = (Get-Date).ToString("o")
        extra        = $extra
    }
    Add-PhaseEntry $manifest $entry
    Write-Manifest $manifest
}

function Run-DetachedPython {
    param(
        [string]$logPath,
        [string[]]$pythonArgs,
        [string]$tag
    )

    $proc = Start-Process -FilePath $py -ArgumentList $pythonArgs `
        -WorkingDirectory "E:\resume-training" -WindowStyle Hidden `
        -PassThru -RedirectStandardOutput $logPath -RedirectStandardError "${logPath}.err"

    $started = (Get-Date).ToString("o")
    Write-Host "[$tag] PID=$($proc.Id) started at $started"
    Set-Status $tag "PID=$($proc.Id) log=$logPath"

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        $tail = Get-Content $logPath -Tail 1 -ErrorAction SilentlyContinue
        Write-Host "[$tag] (still running, PID=$($proc.Id)): $tail"
    }
    $proc.WaitForExit(5000) | Out-Null
    $exitCode = if ($null -eq $proc.ExitCode) { 0 } else { $proc.ExitCode }
    Write-Host "[$tag] Exited with code $exitCode"
    return @{ exit_code = $exitCode; log = $logPath; started_at = $started; ended_at = (Get-Date).ToString("o") }
}

function Find-NewestCheckpointMatching {
    param(
        [string]$pattern,
        [datetime]$mustBeAfter
    )
    $candidates = Get-ChildItem "E:\resume-training\models\$pattern" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $mustBeAfter } |
        Sort-Object LastWriteTime -Descending
    if ($candidates.Count -eq 0) { return $null }
    return $candidates[0].FullName
}

# ---------- Phase 0: wait for sum_mlp@100 baseline ----------

if (-not (Skip-If-Done "wait_for_sum_baseline")) {
    Write-Host "[wait_for_sum_baseline] Polling for sum_mlp@100 baseline checkpoint..."
    Set-Status "wait_for_sum_baseline" "polling"

    $sumBaselineLog = "$logDir\sum_mlp_baseline_20260417_215819.log"
    $sumBaselineCheckpoint = $null
    $startedWaiting = Get-Date
    $timeoutMin = 120

    while ($true) {
        $tail = Get-Content $sumBaselineLog -Tail 5 -ErrorAction SilentlyContinue
        $hasComplete = $tail -join "`n" | Select-String -Pattern "Training complete\."
        if ($hasComplete) {
            # Find the upd100 checkpoint written today
            $sumBaselineCheckpoint = Find-NewestCheckpointMatching "ppo_agent_0417_*upd100_wins*.pth" -mustBeAfter (Get-Date "2026-04-17 21:58:00")
            if (-not $sumBaselineCheckpoint) {
                $sumBaselineCheckpoint = Find-NewestCheckpointMatching "ppo_agent_0418_*upd100_wins*.pth" -mustBeAfter (Get-Date "2026-04-17 21:58:00")
            }
            if ($sumBaselineCheckpoint) {
                Write-Host "[wait_for_sum_baseline] Found sum@100: $sumBaselineCheckpoint"
                break
            }
        }
        if (((Get-Date) - $startedWaiting).TotalMinutes -gt $timeoutMin) {
            Write-Host "[wait_for_sum_baseline] TIMEOUT after $timeoutMin min — aborting"
            Set-Status "wait_for_sum_baseline" "timeout"
            exit 1
        }
        $latestUpdate = (Get-Content $sumBaselineLog -Tail 30 -ErrorAction SilentlyContinue |
            Select-String -Pattern "--- Update (\d+)/100" |
            Select-Object -Last 1).Matches.Groups[1].Value
        Write-Host "[wait_for_sum_baseline] Still waiting (sum_mlp on update $latestUpdate/100)..."
        Set-Status "wait_for_sum_baseline" "sum_mlp upd $latestUpdate/100"
        Start-Sleep -Seconds 120
    }

    Mark-Done "wait_for_sum_baseline" @{ sum_baseline_checkpoint = $sumBaselineCheckpoint }
}

# Re-read the path from manifest (works even on resume)
$sumBaselineCheckpoint = (Get-PhaseEntry (Read-Manifest) "wait_for_sum_baseline").extra.sum_baseline_checkpoint
Write-Host "[orchestrator] sum_baseline_checkpoint = $sumBaselineCheckpoint"

# ---------- Phase 1: safeguard sum@100 ----------

if (-not (Skip-If-Done "safeguard_sum_100")) {
    Set-Status "safeguard_sum_100" "copying"
    $sumBaselineCopy = Join-Path $masterModelsDir (Split-Path $sumBaselineCheckpoint -Leaf)
    Copy-Item $sumBaselineCheckpoint $sumBaselineCopy -Force
    Write-Host "[safeguard_sum_100] Copied to $sumBaselineCopy"
    Mark-Done "safeguard_sum_100" @{ master_path = $sumBaselineCopy }
}

# ---------- Phase 2: showdown tournament v1 ----------

if (-not (Skip-If-Done "showdown_tournament_v1")) {
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $tourLog = "$logDir\overnight_showdown_v1_$ts.log"
    $tourOut = "E:\resume-training\analysis\elo_showdown_v1_$ts"
    Set-Status "showdown_tournament_v1" "running"

    $args = @(
        "-u", "-m", "src", "elo",
        "--checkpoints",
        "E:\resume-training\models\ppo_agent_0417_1826_upd50_wins3194.pth",
        "E:\resume-training\models\ppo_agent_0417_1940_upd100_wins3200.pth",
        $sumBaselineCheckpoint,
        "--agents", "random,heuristic,simple",
        "--games-per-pair", "1000",
        "--num-workers", "12",
        "--analyze",
        "--output-dir", $tourOut
    )
    $r = Run-DetachedPython -logPath $tourLog -pythonArgs $args -tag "showdown_v1"
    Mark-Done "showdown_tournament_v1" @{ log = $tourLog; output_dir = $tourOut; exit_code = $r.exit_code }
}

# ---------- Phase 3: resume hp3@100 -> hp3@200 ----------

if (-not (Skip-If-Done "hp3_resume_100_to_200")) {
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = "$logDir\overnight_hp3_resume_to_200_$ts.log"
    Set-Status "hp3_resume_100_to_200" "training"

    $args = @(
        "-u", "-m", "src", "train",
        "--resume", "E:\resume-training\models\ppo_agent_0417_1940_upd100_wins3200.pth",
        "--pool-type", "attention", "--actor-type", "attention",
        "--clip-eps", "0.3", "--entropy", "0.01",
        "--episodes", "16000", "--num-workers", "16",
        "--self-play", "--eval-every", "20",
        "--updates", "100"
    )
    $startTime = Get-Date
    $r = Run-DetachedPython -logPath $log -pythonArgs $args -tag "hp3_to_200"

    # Final checkpoint: highest upd200 written after start
    $hp3At200 = Find-NewestCheckpointMatching "ppo_agent_*upd200_wins*.pth" -mustBeAfter $startTime
    Mark-Done "hp3_resume_100_to_200" @{
        log              = $log
        exit_code        = $r.exit_code
        final_checkpoint = $hp3At200
    }
}

$hp3At200 = (Get-PhaseEntry (Read-Manifest) "hp3_resume_100_to_200").extra.final_checkpoint

# ---------- Phase 4: safeguard hp3@200 ----------

if (-not (Skip-If-Done "safeguard_hp3_200")) {
    if ($hp3At200 -and (Test-Path $hp3At200)) {
        $copy = Join-Path $masterModelsDir (Split-Path $hp3At200 -Leaf)
        Copy-Item $hp3At200 $copy -Force
        Write-Host "[safeguard_hp3_200] Copied to $copy"
        Mark-Done "safeguard_hp3_200" @{ master_path = $copy }
    } else {
        Write-Host "[safeguard_hp3_200] WARNING: hp3@200 not found, skipping"
        Mark-Done "safeguard_hp3_200" @{ warning = "hp3@200 not found" }
    }
}

# ---------- Phase 5: resume sum@100 -> sum@200 ----------

if (-not (Skip-If-Done "sum_resume_100_to_200")) {
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = "$logDir\overnight_sum_resume_to_200_$ts.log"
    Set-Status "sum_resume_100_to_200" "training"

    $args = @(
        "-u", "-m", "src", "train",
        "--resume", $sumBaselineCheckpoint,
        "--pool-type", "sum", "--actor-type", "mlp",
        "--clip-eps", "0.3", "--entropy", "0.01",
        "--episodes", "16000", "--num-workers", "16",
        "--self-play", "--eval-every", "20",
        "--updates", "100"
    )
    $startTime = Get-Date
    $r = Run-DetachedPython -logPath $log -pythonArgs $args -tag "sum_to_200"

    $sumAt200 = Find-NewestCheckpointMatching "ppo_agent_*upd200_wins*.pth" -mustBeAfter $startTime
    Mark-Done "sum_resume_100_to_200" @{
        log              = $log
        exit_code        = $r.exit_code
        final_checkpoint = $sumAt200
    }
}

$sumAt200 = (Get-PhaseEntry (Read-Manifest) "sum_resume_100_to_200").extra.final_checkpoint

# ---------- Phase 6: safeguard sum@200 ----------

if (-not (Skip-If-Done "safeguard_sum_200")) {
    if ($sumAt200 -and (Test-Path $sumAt200)) {
        $copy = Join-Path $masterModelsDir (Split-Path $sumAt200 -Leaf)
        Copy-Item $sumAt200 $copy -Force
        Write-Host "[safeguard_sum_200] Copied to $copy"
        Mark-Done "safeguard_sum_200" @{ master_path = $copy }
    } else {
        Write-Host "[safeguard_sum_200] WARNING: sum@200 not found, skipping"
        Mark-Done "safeguard_sum_200" @{ warning = "sum@200 not found" }
    }
}

# ---------- Phase 7: final tournament ----------

if (-not (Skip-If-Done "final_tournament")) {
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = "$logDir\overnight_final_tournament_$ts.log"
    $out = "E:\resume-training\analysis\elo_final_$ts"
    Set-Status "final_tournament" "running"

    $checkpoints = @(
        "E:\resume-training\models\ppo_agent_0417_1940_upd100_wins3200.pth",  # hp3@100
        $sumBaselineCheckpoint                                                  # sum@100
    )
    if ($hp3At200 -and (Test-Path $hp3At200)) { $checkpoints += $hp3At200 }
    if ($sumAt200 -and (Test-Path $sumAt200)) { $checkpoints += $sumAt200 }

    $args = @(
        "-u", "-m", "src", "elo",
        "--checkpoints"
    ) + $checkpoints + @(
        "--agents", "random,heuristic,simple",
        "--games-per-pair", "1500",
        "--num-workers", "12",
        "--analyze",
        "--output-dir", $out
    )
    $r = Run-DetachedPython -logPath $log -pythonArgs $args -tag "final_tournament"
    Mark-Done "final_tournament" @{ log = $log; output_dir = $out; exit_code = $r.exit_code }
}

# ---------- Phase 8: summary report ----------

if (-not (Skip-If-Done "summary_report")) {
    Set-Status "summary_report" "writing"
    $report = "$logDir\overnight_master_report.md"
    $manifest = Read-Manifest

    $sb = New-Object System.Text.StringBuilder
    $null = $sb.AppendLine("# Overnight Run Report")
    $null = $sb.AppendLine("Generated: $((Get-Date).ToString('o'))")
    $null = $sb.AppendLine("")
    $null = $sb.AppendLine("## Phase Summary")
    $null = $sb.AppendLine("")
    foreach ($p in $manifest.phases) {
        $null = $sb.AppendLine("### $($p.key)")
        $null = $sb.AppendLine("- completed: $($p.completed)")
        $null = $sb.AppendLine("- completed_at: $($p.completed_at)")
        if ($p.extra) {
            $null = $sb.AppendLine("- extra:")
            $null = $sb.AppendLine("  ``````")
            $null = $sb.AppendLine("  $(($p.extra | ConvertTo-Json -Depth 5))")
            $null = $sb.AppendLine("  ``````")
        }
        $null = $sb.AppendLine("")
    }

    # Append final tournament leaderboard if available
    $finalEntry = Get-PhaseEntry $manifest "final_tournament"
    if ($finalEntry -and $finalEntry.extra.log -and (Test-Path $finalEntry.extra.log)) {
        $leaderboard = Get-Content $finalEntry.extra.log | Select-String -Pattern "ELO LEADERBOARD" -Context 0, 15
        $null = $sb.AppendLine("## Final Tournament Leaderboard")
        $null = $sb.AppendLine("``````")
        if ($leaderboard) {
            $null = $sb.AppendLine($leaderboard.ToString())
        } else {
            $null = $sb.AppendLine("(not found in log)")
        }
        $null = $sb.AppendLine("``````")
    }

    Set-Content $report $sb.ToString()
    # Copy to master worktree for safety
    Copy-Item $report (Join-Path $masterModelsDir "..\logs\overnight_master_report.md") -Force -ErrorAction SilentlyContinue
    Write-Host "[summary_report] Wrote $report"
    Mark-Done "summary_report" @{ report_path = $report }
}

Set-Status "DONE" "all phases complete"
Write-Host ""
Write-Host "========================================"
Write-Host "Overnight orchestrator complete."
Write-Host "Report: $logDir\overnight_master_report.md"
Write-Host "========================================"
