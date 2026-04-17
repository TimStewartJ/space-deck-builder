# Overnight architecture sweep orchestrator.
#
# Pipeline:
#   1. Train four 100-update runs, one per (pool_type, actor_type) combo,
#      sequentially, with the same rollout budget as the earlier 30-update
#      comparison (16000 episodes * 16 workers, self-play, eval every 10).
#   2. Elo tournament across all final checkpoints that completed.
#   3. If time permits, a final bonus training run using the Elo winner's
#      architecture.
#
# Soft deadlines (all America/Los_Angeles local time):
#   $stopNewTrainings  -> skip launching a new training run after this time
#   $stopElo           -> skip Elo after this time (abort mid-phase if reached)
#   $hardDeadline      -> everything must be wrapped up by this time
#
# Status is continuously written to logs\overnight_status.json so a scheduled
# wake-up session can report progress without needing to read the raw logs.
#
# Abort hooks:
#   * An ABORT.flag file in logs\ interrupts between phases.
#   * A scheduled 08:30 session hard-kills remaining python.exe processes.

$ErrorActionPreference = "Stop"
$root = "E:\attention-pooling"
$py   = "$root\.venv\Scripts\python.exe"
$logs = "$root\logs"
$modelsDir = "$root\models"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# --- Deadlines ---
# Overridable via env vars so a resumed orchestrator can push deadlines out.
# Times are interpreted as "today" local; if the computed time is already in
# the past, it rolls forward by a day.
function _Deadline([string]$envKey, [int]$defH, [int]$defM) {
    $v = [Environment]::GetEnvironmentVariable($envKey)
    if ($v) {
        try { return [DateTime]::Parse($v) } catch { }
    }
    return Get-Date -Hour $defH -Minute $defM -Second 0
}
$stopNewTrainings = _Deadline "SWEEP_STOP_NEW" 6 30
$stopElo          = _Deadline "SWEEP_STOP_ELO" 7 30
$hardDeadline     = _Deadline "SWEEP_HARD_DEADLINE" 8 30

foreach ($name in "stopNewTrainings","stopElo","hardDeadline") {
    $val = Get-Variable -Name $name -ValueOnly
    if ($val -lt (Get-Date)) {
        Set-Variable -Name $name -Value $val.AddDays(1)
    }
}

# Run keys listed here (space-separated via SWEEP_SKIP_RUNS) are skipped
# entirely — useful when resuming after a crash where some runs already
# completed.
$skipRuns = @()
$skipEnv = [Environment]::GetEnvironmentVariable("SWEEP_SKIP_RUNS")
if ($skipEnv) { $skipRuns = $skipEnv -split '\s+' }

$abortFlag   = "$logs\OVERNIGHT_ABORT.flag"
$statusFile  = "$logs\overnight_status.json"
$masterLog   = "$logs\overnight_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$manifestFile = "$logs\overnight_manifest.json"

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Add-Content -Path $masterLog -Value $line
}

function Check-Abort {
    if (Test-Path $abortFlag) {
        Log "ABORT flag detected - stopping."
        return $true
    }
    if ((Get-Date) -ge $hardDeadline) {
        Log "Hard deadline ($hardDeadline) passed - stopping."
        return $true
    }
    return $false
}

# Initialize manifest
if (-not (Test-Path $manifestFile)) {
    @{
        started_at = (Get-Date).ToString("o")
        runs = @()
        elo = $null
        bonus = $null
    } | ConvertTo-Json -Depth 10 | Set-Content $manifestFile
}

function Save-Manifest($m) {
    $m | ConvertTo-Json -Depth 10 | Set-Content $manifestFile
}

function Write-Status([string]$phase, $extra = @{}) {
    $status = @{
        phase = $phase
        updated_at = (Get-Date).ToString("o")
        hard_deadline = $hardDeadline.ToString("o")
        stop_new_trainings = $stopNewTrainings.ToString("o")
        master_log = $masterLog
    }
    foreach ($k in $extra.Keys) { $status[$k] = $extra[$k] }
    $status | ConvertTo-Json -Depth 10 | Set-Content $statusFile
}

# --- Phase 1: four training runs ---
$runs = @(
    @{ key = "sum_mlp";       pool = "sum";       actor = "mlp"       },
    @{ key = "attn_attn";     pool = "attention"; actor = "attention" },
    @{ key = "sum_attn";      pool = "sum";       actor = "attention" },
    @{ key = "attn_mlp";      pool = "attention"; actor = "mlp"       }
)

$commonArgs = @(
    "-u", "-m", "src", "train",
    "--episodes", "16000",
    "--num-workers", "16",
    "--self-play",
    "--updates", "100",
    "--eval-every", "20",
    "--eval-games", "200"
)

$completed = @()
# Rehydrate from existing manifest (supports resume after crash).
$existingManifest = Get-Content $manifestFile -Raw | ConvertFrom-Json
foreach ($prev in @($existingManifest.runs)) {
    if ($prev -and $prev.exit_code -eq 0 -and $prev.final_checkpoint) {
        $completed += $prev
        if (-not ($skipRuns -contains $prev.key)) { $skipRuns += $prev.key }
        Log "Rehydrated completed run: $($prev.key) -> $($prev.final_checkpoint)"
    }
}
foreach ($r in $runs) {
    if (Check-Abort) { break }
    if ($skipRuns -contains $r.key) {
        Log "SKIP $($r.key) (listed in SWEEP_SKIP_RUNS)"
        continue
    }
    if ((Get-Date) -ge $stopNewTrainings) {
        Log "stopNewTrainings ($stopNewTrainings) reached - skipping $($r.key) and remaining."
        break
    }

    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $runLog = "$logs\overnight_$($r.key)_$ts.log"
    Write-Status "training" @{ run = $r.key; pool = $r.pool; actor = $r.actor; run_log = $runLog }
    Log "=== START $($r.key)  pool=$($r.pool) actor=$($r.actor)  log=$runLog ==="

    $argsList = $commonArgs + @("--pool-type", $r.pool, "--actor-type", $r.actor)
    "=== $($r.key) start $(Get-Date -Format o) ===" | Out-File $runLog
    $runStart = Get-Date
    $exitCode = -1
    try {
        # Launch python as an independent process so we can enforce a
        # shutdown-hang watchdog: even when training logs "Training complete."
        # the mp workers / CUDA context occasionally refuse to exit, which
        # would otherwise wedge the orchestrator indefinitely (observed
        # 2026-04-17 overnight sum_mlp hang).
        $proc = Start-Process -FilePath $py -ArgumentList $argsList `
            -WorkingDirectory (Get-Location) -NoNewWindow -PassThru `
            -RedirectStandardOutput $runLog.Replace('.log','.stdout.log') `
            -RedirectStandardError  $runLog.Replace('.log','.stderr.log')
        Log "Launched $($r.key) pid=$($proc.Id)"

        # Poll loop: detect normal exit, abort flag, hard deadline, and
        # the hung-shutdown case (log shows completion but process keeps running).
        $completeSeenAt = $null
        $shutdownGraceSec = 180
        while (-not $proc.HasExited) {
            Start-Sleep -Seconds 15
            if (Check-Abort) { break }
            if ((Get-Date) -ge $hardDeadline) {
                Log "Hard deadline hit during $($r.key) - killing pid=$($proc.Id)"
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                break
            }
            # Merge stdout/stderr tails into the main log so the manifest
            # paths stay authoritative.
            foreach ($suffix in ".stdout.log",".stderr.log") {
                $side = $runLog.Replace('.log',$suffix)
                if (Test-Path $side) {
                    Get-Content $side -Tail 2000 -ErrorAction SilentlyContinue |
                        Out-File -Append $runLog -ErrorAction SilentlyContinue
                    Clear-Content $side -ErrorAction SilentlyContinue
                }
            }
            # Watchdog: detect the shutdown hang.
            $tail = Get-Content $runLog -Tail 200 -ErrorAction SilentlyContinue
            if ($tail -match "Training complete\.") {
                if (-not $completeSeenAt) {
                    $completeSeenAt = Get-Date
                    Log "$($r.key): saw 'Training complete.' - grace $shutdownGraceSec s before force-kill"
                }
                elseif (((Get-Date) - $completeSeenAt).TotalSeconds -ge $shutdownGraceSec) {
                    Log "$($r.key): shutdown hang after grace - force-killing process tree pid=$($proc.Id)"
                    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" |
                        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                    break
                }
            }
        }
        if ($proc.HasExited) {
            $exitCode = $proc.ExitCode
        } else {
            # Ensure any surviving tree is reaped so the next run starts clean.
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $exitCode = -2
        }
    } catch {
        Log "Run $($r.key) threw: $_"
        $exitCode = -1
    }
    $runEnd = Get-Date
    "=== $($r.key) done $(Get-Date -Format o) exit=$exitCode ===" | Out-File -Append $runLog

    # Find the newest checkpoint matching the training timestamp window.
    $ckpt = Get-ChildItem $modelsDir -Filter "ppo_agent_*.pth" |
        Where-Object { $_.LastWriteTime -ge $runStart -and $_.LastWriteTime -le $runEnd.AddMinutes(2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    $entry = @{
        key = $r.key
        pool = $r.pool
        actor = $r.actor
        started_at = $runStart.ToString("o")
        ended_at = $runEnd.ToString("o")
        wall_s = [int]($runEnd - $runStart).TotalSeconds
        exit_code = $exitCode
        log = $runLog
        final_checkpoint = if ($ckpt) { $ckpt.FullName } else { $null }
    }

    $m = Get-Content $manifestFile -Raw | ConvertFrom-Json
    $m.runs = @($m.runs) + @($entry)
    Save-Manifest $m

    Log "=== END $($r.key) exit=$exitCode wall=$([int]($runEnd-$runStart).TotalSeconds)s checkpoint=$($entry.final_checkpoint) ==="
    if ($exitCode -eq 0 -and $entry.final_checkpoint) { $completed += $entry }
}

# --- Phase 2: Elo tournament ---
$eloLog = "$logs\overnight_elo_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
if ($completed.Count -ge 2 -and (Get-Date) -lt $stopElo -and -not (Test-Path $abortFlag)) {
    Write-Status "elo" @{ candidates = $completed.Count; elo_log = $eloLog }
    Log "=== START Elo  candidates=$($completed.Count)  log=$eloLog ==="

    $checkpointPaths = $completed | ForEach-Object { $_.final_checkpoint }
    $gamesPerPair = 500
    $eloArgs = @(
        "-u", "-m", "src", "elo",
        "--checkpoints"
    ) + $checkpointPaths + @(
        "--agents", "random,heuristic,simple",
        "--games-per-pair", "$gamesPerPair",
        "--num-workers", "16",
        "--simulation-device", "cuda"
    )
    $eloStart = Get-Date
    try {
        & $py $eloArgs *>> $eloLog
        $eloExit = $LASTEXITCODE
    } catch {
        Log "Elo threw: $_"
        $eloExit = -1
    }
    $eloEnd = Get-Date
    Log "=== END Elo exit=$eloExit wall=$([int]($eloEnd-$eloStart).TotalSeconds)s ==="

    $m = Get-Content $manifestFile -Raw | ConvertFrom-Json
    $m.elo = @{
        log = $eloLog
        started_at = $eloStart.ToString("o")
        ended_at = $eloEnd.ToString("o")
        exit_code = $eloExit
        games_per_pair = $gamesPerPair
        checkpoints = $checkpointPaths
    }
    Save-Manifest $m

    # Try to parse winner from the leaderboard output (rank 1 that is a
    # checkpoint, not a builtin).
    $leaderboard = Get-Content $eloLog
    $winnerKey = $null
    $winnerLabel = $null
    $inLb = $false
    foreach ($ln in $leaderboard) {
        if ($ln -match "^Rank\s+Name") { $inLb = $true; continue }
        if ($inLb -and $ln -match "^1\s+(\S+)\s+") {
            $winnerLabel = $Matches[1]
            foreach ($c in $completed) {
                if ($c.final_checkpoint -and $c.final_checkpoint -like "*$winnerLabel*") {
                    $winnerKey = $c.key
                    break
                }
            }
            break
        }
    }
    if (-not $winnerKey -and $completed.Count -gt 0) {
        # Fall back: whichever run completed first (arbitrary but deterministic).
        $winnerKey = $completed[0].key
    }
    $m = Get-Content $manifestFile -Raw | ConvertFrom-Json
    $m.elo | Add-Member -NotePropertyName winner_label -NotePropertyValue $winnerLabel -Force
    $m.elo | Add-Member -NotePropertyName winner_key   -NotePropertyValue $winnerKey   -Force
    Save-Manifest $m
    Log "Elo winner label=$winnerLabel  key=$winnerKey"
} else {
    $abortSet = Test-Path $abortFlag
    $now = Get-Date
    Log "Skipping Elo candidates=$($completed.Count) time=$now stopElo=$stopElo abort=$abortSet"
}

# --- Phase 3: bonus training run with the Elo winner's config ---
$m = Get-Content $manifestFile -Raw | ConvertFrom-Json
$winnerKey = $null
if ($m.elo) { $winnerKey = $m.elo.winner_key }
if ($winnerKey -and (Get-Date) -lt $stopNewTrainings -and -not (Test-Path $abortFlag)) {
    $winRun = $runs | Where-Object { $_.key -eq $winnerKey } | Select-Object -First 1
    if ($winRun) {
        $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
        $bonusLog = "$logs\overnight_bonus_$($winRun.key)_$ts.log"
        Write-Status "bonus_training" @{ run = $winRun.key; log = $bonusLog }
        Log "=== START BONUS $($winRun.key) pool=$($winRun.pool) actor=$($winRun.actor) log=$bonusLog ==="
        $bonusArgs = $commonArgs + @("--pool-type", $winRun.pool, "--actor-type", $winRun.actor)
        $bonusStart = Get-Date
        try {
            & $py $bonusArgs *>> $bonusLog
            $bonusExit = $LASTEXITCODE
        } catch {
            Log "Bonus run threw: $_"
            $bonusExit = -1
        }
        $bonusEnd = Get-Date
        Log "=== END BONUS $($winRun.key) exit=$bonusExit wall=$([int]($bonusEnd-$bonusStart).TotalSeconds)s ==="
        $m = Get-Content $manifestFile -Raw | ConvertFrom-Json
        $m.bonus = @{
            run = $winRun.key
            log = $bonusLog
            started_at = $bonusStart.ToString("o")
            ended_at = $bonusEnd.ToString("o")
            exit_code = $bonusExit
        }
        Save-Manifest $m
    }
} else {
    $abortSet = Test-Path $abortFlag
    $now = Get-Date
    Log "Skipping bonus training winner=$winnerKey time=$now stopNewTrainings=$stopNewTrainings abort=$abortSet"
}

Write-Status "done" @{}
Log "=== OVERNIGHT SWEEP DONE ==="
