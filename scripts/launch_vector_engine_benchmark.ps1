<#
.SYNOPSIS
    Launch the Gate 2 rules-engine benchmark outside the caller's process tree.

.DESCRIPTION
    Uses Windows Task Scheduler at Normal priority and emits timestamped log
    and JSON artifacts. Launch context is load-bearing on this machine:
    identical workloads run several times slower as agent-runtime descendants.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [int]$Games = 16000,
    [int]$Steps = 200,
    [int]$IntegratedSteps = 40,
    [int]$Reps = 5,
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
$taskName = if ($Name) { $Name } else { "sdb_gate2_$stamp`_$runId" }
$log = Join-Path $logDir "gate2_$stamp.log"
$json = Join-Path $logDir "gate2_$stamp.json"

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Scheduled task '$taskName' already exists."
}

$benchmarkArgs = @(
    "--games", $Games,
    "--steps", $Steps,
    "--integrated-steps", $IntegratedSteps,
    "--reps", $Reps,
    "--json-out", "`"$json`""
) -join " "
$inner = (
    "set `"PYTHONPATH=$Repo`" && cd /d `"$Repo`" && " +
    "`"$python`" -u scripts\bench_vector_engine.py $benchmarkArgs " +
    "> `"$log`" 2>&1"
)

if (-not $PSCmdlet.ShouldProcess(
    $taskName,
    "Register and start detached Gate 2 vector-engine benchmark"
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
    if (
        $task.State -eq "Running" -or
        $info.LastRunTime -gt $registeredLastRun
    ) {
        $started = $true
        break
    }
}
if (-not $started) {
    throw "Scheduled task '$taskName' did not start within five seconds."
}

Write-Host "Launched '$taskName' outside the caller tree at Normal priority."
Write-Host "  log:    $log"
Write-Host "  json:   $json"
Write-Host "  follow: Get-Content '$log' -Wait -Tail 30"
Write-Host "  stop:   Stop-ScheduledTask -TaskName '$taskName'"
Write-Host "  clean:  Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
