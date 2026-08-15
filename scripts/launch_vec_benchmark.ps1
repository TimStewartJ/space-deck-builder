<#
.SYNOPSIS
    Launch the vector primitive benchmark outside the caller's process tree.

.DESCRIPTION
    Runs through Windows Task Scheduler at priority 4 (Normal). This is
    required for trustworthy numbers on this machine: identical rollout code
    measured about 3.7x slower when launched as a descendant of the agent
    runtime, even after its process priority was raised to Normal.

    The task's parent is the scheduler service rather than the caller. Output
    is written to a timestamped log and structured JSON file.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [int]$Games = 16000,
    [int]$Steps = 200,
    [int]$Reps = 5,
    [string]$Backends = "numpy,numba,torch-cpu,torch-cuda",
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Name = ""
)

$ErrorActionPreference = "Stop"

$python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "E:\space-deck-builder\.venv\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "No interpreter found. Run: uv sync --extra rocm"
}

$logDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$runId = [guid]::NewGuid().ToString("N").Substring(0, 8)
$log = Join-Path $logDir "vecbench_$stamp.log"
$json = Join-Path $logDir "vecbench_$stamp.json"
$taskName = if ($Name) { $Name } else { "sdb_vecbench_$stamp`_$runId" }

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    throw (
        "Scheduled task '$taskName' already exists. Use a unique -Name or " +
        "omit -Name to generate one automatically."
    )
}

$benchmarkArgs = @(
    "--games", $Games,
    "--steps", $Steps,
    "--reps", $Reps,
    "--backends", "`"$Backends`"",
    "--require-all",
    "--json-out", "`"$json`""
) -join " "

$inner = (
    "set `"PYTHONPATH=$Repo`" && cd /d `"$Repo`" && " +
    "`"$python`" -u scripts\bench_vec_primitives.py $benchmarkArgs " +
    "> `"$log`" 2>&1"
)

if (-not $PSCmdlet.ShouldProcess(
    $taskName,
    "Register and start detached vector primitive benchmark"
)) {
    Write-Host "Would execute via cmd.exe:"
    Write-Host "  $inner"
    return
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $inner"
$settings = New-ScheduledTaskSettingsSet `
    -Priority 4 `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances Parallel

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Settings $settings `
    -Force | Out-Null
$registeredLastRun = (
    Get-ScheduledTaskInfo -TaskName $taskName
).LastRunTime
Start-ScheduledTask -TaskName $taskName

$started = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 250
    $task = Get-ScheduledTask -TaskName $taskName
    $info = Get-ScheduledTaskInfo -TaskName $taskName
    # Capture Task Scheduler's actual never-run sentinel above rather than
    # assuming DateTime.MinValue (Windows commonly reports a 1999 timestamp).
    # A strictly newer LastRunTime proves this invocation started even if a
    # tiny smoke run already returned to Ready before the first 250ms poll.
    if ($task.State -eq "Running" -or $info.LastRunTime -gt $registeredLastRun) {
        $started = $true
        break
    }
}
if (-not $started) {
    throw "Scheduled task '$taskName' did not start within five seconds."
}

Write-Host "Launched '$taskName' outside the caller's process tree at Normal priority."
Write-Host "  log:    $log"
Write-Host "  json:   $json"
Write-Host "  follow: Get-Content '$log' -Wait -Tail 30"
Write-Host "  stop:   Stop-ScheduledTask -TaskName '$taskName'"
Write-Host "  clean:  Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
