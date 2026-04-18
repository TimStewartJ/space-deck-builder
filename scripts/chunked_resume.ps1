# Chunked resume orchestrator.
#
# Resumes PPO training in fixed-size chunks (one fresh process per chunk) to
# work around a ROCm/HIP runtime crash observed when long-lived training
# processes accumulate ~25-30 updates worth of InferenceServer + worker
# spawn/teardown cycles. Symptom: c10::cuda::memcpy_and_sync failure inside
# torch_hip.dll partway through rollout of update N+25-ish; see overnight
# crash logs from 2026-04-18.
#
# Each chunk runs `--updates <chunk>` from the latest checkpoint. After the
# chunk exits cleanly, the script finds the new highest-update checkpoint
# and chains the next process from it. Aborts on any non-zero exit.
#
# Usage:
#   .\scripts\chunked_resume.ps1 `
#       -StartCheckpoint "models\ppo_agent_..._upd100_wins3198.pth" `
#       -TargetUpdate 200 `
#       -ChunkSize 25 `
#       -Tag "hp3_chunked" `
#       -ExtraArgs @("--pool-type","attention","--actor-type","attention",`
#                    "--clip-eps","0.3","--entropy","0.01",`
#                    "--episodes","16000","--num-workers","16",`
#                    "--self-play","--eval-every","20")
#
# Logs go to logs\chunk_<tag>_<N>_<timestamp>.log. Manifest goes to
# logs\chunked_resume_<tag>.json.

param(
    [Parameter(Mandatory = $true)] [string] $StartCheckpoint,
    [Parameter(Mandatory = $true)] [int]    $TargetUpdate,
    [Parameter(Mandatory = $true)] [string] $Tag,
    [int]      $ChunkSize  = 25,
    [string[]] $ExtraArgs  = @(),
    [string]   $LogsDir    = "E:\resume-training\logs",
    [string]   $ModelsDir  = "E:\resume-training\models",
    [string]   $WorkingDir = "E:\resume-training",
    [string]   $Python     = "E:\resume-training\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

function Get-CurrentUpdate([string] $ckpt) {
    if ($ckpt -match "upd(\d+)_wins") { return [int]$Matches[1] }
    throw "Could not parse update number from checkpoint name: $ckpt"
}

function Find-Latest-Checkpoint([int] $afterUpdate, [datetime] $afterTime) {
    Get-ChildItem $ModelsDir -Filter "ppo_agent_*upd*_wins*.pth" |
        Where-Object { $_.LastWriteTime -ge $afterTime } |
        ForEach-Object {
            if ($_.Name -match "upd(\d+)_wins") {
                [PSCustomObject]@{ File = $_; Update = [int]$Matches[1] }
            }
        } |
        Where-Object { $_.Update -gt $afterUpdate } |
        Sort-Object Update -Descending |
        Select-Object -First 1
}

$manifestPath = Join-Path $LogsDir "chunked_resume_$Tag.json"
$manifest = @{
    tag               = $Tag
    target_update     = $TargetUpdate
    chunk_size        = $ChunkSize
    start_checkpoint  = $StartCheckpoint
    started_at        = (Get-Date).ToString("o")
    chunks            = @()
}

$currentCkpt = $StartCheckpoint
$currentUpd  = Get-CurrentUpdate $currentCkpt

Write-Host "[chunked_resume:$Tag] start=$currentCkpt currentUpd=$currentUpd target=$TargetUpdate chunk=$ChunkSize"

while ($currentUpd -lt $TargetUpdate) {
    $remaining = $TargetUpdate - $currentUpd
    $thisChunk = [math]::Min($ChunkSize, $remaining)

    $ts  = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $LogsDir "chunk_${Tag}_upd$($currentUpd + $thisChunk)_$ts.log"

    $args = @(
        "-u", "-m", "src", "train",
        "--resume", $currentCkpt,
        "--updates", "$thisChunk"
    ) + $ExtraArgs

    Write-Host "[chunked_resume:$Tag] chunk start: from=upd$currentUpd → upd$($currentUpd + $thisChunk)  log=$log"
    $startTime = Get-Date
    $proc = Start-Process -FilePath $Python -ArgumentList $args `
        -WorkingDirectory $WorkingDir -WindowStyle Hidden `
        -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        $tail = Get-Content $log -Tail 1 -ErrorAction SilentlyContinue
        Write-Host "[chunked_resume:$Tag] (running PID=$($proc.Id)): $tail"
    }
    $proc.WaitForExit(5000) | Out-Null
    $exitCode = if ($null -eq $proc.ExitCode) { 0 } else { $proc.ExitCode }

    $newCkpt = Find-Latest-Checkpoint -afterUpdate $currentUpd -afterTime $startTime
    $newCkptPath = if ($newCkpt) { $newCkpt.File.FullName } else { $null }
    $newUpd      = if ($newCkpt) { $newCkpt.Update } else { $currentUpd }

    $manifest.chunks += @{
        chunk_size       = $thisChunk
        from_update      = $currentUpd
        to_update_target = $currentUpd + $thisChunk
        to_update_actual = $newUpd
        from_ckpt        = $currentCkpt
        new_ckpt         = $newCkptPath
        log              = $log
        exit_code        = $exitCode
        started_at       = $startTime.ToString("o")
        ended_at         = (Get-Date).ToString("o")
    }
    $manifest.last_update = $newUpd
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath

    if ($exitCode -ne 0) {
        Write-Host "[chunked_resume:$Tag] CHUNK CRASHED with exit=$exitCode after upd$newUpd. Aborting."
        $manifest.aborted_at      = (Get-Date).ToString("o")
        $manifest.aborted_reason  = "chunk exit_code=$exitCode (last good upd=$newUpd)"
        $manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath
        exit 1
    }
    if ($newUpd -le $currentUpd) {
        Write-Host "[chunked_resume:$Tag] Chunk exited 0 but no new checkpoint past upd$currentUpd. Aborting."
        $manifest.aborted_at      = (Get-Date).ToString("o")
        $manifest.aborted_reason  = "no progress (chunk exit_code=0 but no checkpoint past upd$currentUpd)"
        $manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath
        exit 2
    }

    Write-Host "[chunked_resume:$Tag] chunk done: upd$currentUpd → upd$newUpd (exit=0)"
    $currentCkpt = $newCkptPath
    $currentUpd  = $newUpd
}

$manifest.completed_at = (Get-Date).ToString("o")
$manifest.final_ckpt   = $currentCkpt
$manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath
Write-Host "[chunked_resume:$Tag] DONE. final=$currentCkpt"
