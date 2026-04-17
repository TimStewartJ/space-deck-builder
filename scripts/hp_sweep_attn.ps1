# Hyperparameter sweep: 3 attn_attn variants to test if attention can match
# sum_mlp@100 with better tuning. Each run is 50 updates, same data settings as
# the original overnight sweep (16k episodes, 16 workers, self-play).
#
# Variants:
#   hp1_lr_lower      : lr=1.5e-4               (lower LR compensates for smaller grad-norm)
#   hp2_more_epochs   : lr=3e-4 epochs=8        (more PPO updates per rollout)
#   hp3_loose_clip    : lr=3e-4 clip-eps=0.3 entropy=0.01  (allow bigger policy moves, less exploration noise)
#
# Each writes its own log, checkpoints, and metrics. After all 3 finish, run an
# Elo tournament against sum_mlp@100 + attn_attn@100 baselines to compare.
#
# Usage: pwsh -File scripts/hp_sweep_attn.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$py = "E:\attention-pooling\.venv\Scripts\python.exe"
$logDir = "E:\attention-pooling\logs"
$manifestFile = "$logDir\hp_sweep_manifest.json"

$commonArgs = @(
    "-u", "-m", "src", "train",
    "--pool-type", "attention",
    "--actor-type", "attention",
    "--episodes", "16000",
    "--num-workers", "16",
    "--self-play",
    "--updates", "50",
    "--eval-every", "20",
    "--eval-games", "200"
)

$variants = @(
    @{ key = "hp1_lr_lower";    extra = @("--lr", "0.00015") },
    @{ key = "hp2_more_epochs"; extra = @("--lr", "0.0003", "--epochs", "8") },
    @{ key = "hp3_loose_clip";  extra = @("--lr", "0.0003", "--clip-eps", "0.3", "--entropy", "0.01") }
)

# Initialize/rehydrate manifest so we can resume after a crash.
if (-not (Test-Path $manifestFile)) {
    @{ started_at = (Get-Date).ToString("o"); runs = @() } | ConvertTo-Json -Depth 10 | Set-Content $manifestFile
}
$manifest = Get-Content $manifestFile -Raw | ConvertFrom-Json
$completedKeys = @($manifest.runs | Where-Object { $_.exit_code -eq 0 } | ForEach-Object { $_.key })

function Log([string]$msg) {
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "[$ts] $msg"
}

foreach ($v in $variants) {
    if ($completedKeys -contains $v.key) {
        Log "SKIP $($v.key) (already completed)"
        continue
    }

    $ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
    $log = "$logDir\hp_$($v.key)_$ts.log"
    $args_ = $commonArgs + $v.extra
    Log "START $($v.key)  ->  $log"
    Log "  args: $($args_ -join ' ')"

    $startTime = Get-Date
    $proc = Start-Process -FilePath $py -ArgumentList $args_ `
        -WorkingDirectory "E:\attention-pooling" `
        -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" `
        -WindowStyle Hidden -PassThru

    Log "  PID $($proc.Id) running..."

    # Wait with periodic readiness check + 180s grace after "Training complete." appears.
    $trainingCompleteTime = $null
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 30
        if (Test-Path $log) {
            $tail = Get-Content $log -Tail 60 -ErrorAction SilentlyContinue
            if (-not $trainingCompleteTime -and ($tail -match "Training complete")) {
                $trainingCompleteTime = Get-Date
                Log "  Training complete signal observed; allowing 180s grace before force-kill"
            }
        }
        if ($trainingCompleteTime -and ((Get-Date) - $trainingCompleteTime).TotalSeconds -gt 180) {
            Log "  Grace period exceeded; force-killing $($proc.Id)"
            try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
            break
        }
    }

    $proc.WaitForExit(5000) | Out-Null
    $exit = if ($proc.ExitCode -eq $null) { 0 } else { $proc.ExitCode }
    $endTime = Get-Date
    $wall = [int]($endTime - $startTime).TotalSeconds

    # Find the latest checkpoint produced during this window.
    $finalCkpt = Get-ChildItem "E:\attention-pooling\models\ppo_agent_*upd50_wins*.pth" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $startTime -and $_.LastWriteTime -le $endTime.AddMinutes(2) } |
        Sort-Object LastWriteTime | Select-Object -Last 1

    Log "DONE  $($v.key)  exit=$exit  wall=${wall}s  ckpt=$($finalCkpt.Name)"

    # Append to manifest.
    $manifest = Get-Content $manifestFile -Raw | ConvertFrom-Json
    $entry = @{
        key = $v.key
        started_at = $startTime.ToString("o")
        ended_at = $endTime.ToString("o")
        wall_s = $wall
        exit_code = $exit
        log = $log
        final_checkpoint = if ($finalCkpt) { $finalCkpt.FullName } else { $null }
        extra_args = $v.extra -join ' '
    }
    $manifest.runs = @($manifest.runs) + $entry
    $manifest | ConvertTo-Json -Depth 10 | Set-Content $manifestFile
}

Log "All variants done. Manifest: $manifestFile"
