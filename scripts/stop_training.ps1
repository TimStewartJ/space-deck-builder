<#
.SYNOPSIS
    Stop one exact launcher-owned training task and its complete process tree.

.DESCRIPTION
    Verifies the task's unique launch identity against its manifest and action
    process before stopping it. The scheduled action owns a kill-on-close
    Windows Job Object, so terminating that exact action atomically terminates
    all current and concurrently spawning descendants without PID-tree races.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Name,
    [string] $Repo = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
if ($Name -cnotmatch "^sdb_[A-Za-z0-9_.-]+$") {
    throw "Invalid training task name: $Name"
}
$TaskPath = "\"
$Repo = (Resolve-Path $Repo).Path
. (Join-Path $PSScriptRoot "training_task_lock.ps1")
$taskLock = Enter-TrainingTaskLock -Name $Name
try {
$task = Get-ScheduledTask `
    -TaskPath $TaskPath `
    -TaskName $Name `
    -ErrorAction SilentlyContinue
if (-not $task) {
    throw "Training task '$Name' does not exist at task path '$TaskPath'."
}

$descriptionPrefix = "Space Deck Builder launch "
$taskDescription = [string]$task.Description
if (
    -not $taskDescription.StartsWith(
        $descriptionPrefix,
        [StringComparison]::Ordinal
    )
) {
    throw (
        "Task '$TaskPath$Name' was not registered by launch_training.ps1; " +
        "refusing to stop it."
    )
}
$taskLaunchId = $taskDescription.Substring($descriptionPrefix.Length)
$parsedLaunchId = [guid]::Empty
if (-not [guid]::TryParse($taskLaunchId, [ref]$parsedLaunchId)) {
    throw "Task '$TaskPath$Name' has an invalid launch identity."
}
$taskLaunchId = $parsedLaunchId.ToString("D")

$manifestCandidates = Get-ChildItem `
    -LiteralPath (Join-Path $Repo "logs") `
    -Filter "$Name`_*.json" |
    Sort-Object LastWriteTime -Descending
$manifest = $null
foreach ($candidate in $manifestCandidates) {
    try {
        $candidateState = Get-Content `
            -LiteralPath $candidate.FullName `
            -Raw |
            ConvertFrom-Json
    } catch {
        Write-Warning (
            "Ignoring unreadable launch manifest $($candidate.FullName): " +
            $_.Exception.Message
        )
        continue
    }
    if (
        [string]$candidateState.task -ceq $Name -and
        [string]$candidateState.task_path -ceq $TaskPath -and
        [string]$candidateState.launch_id -ceq $taskLaunchId
    ) {
        $manifest = $candidate
        $state = $candidateState
        break
    }
}
if (-not $manifest) {
    throw (
        "No launch manifest matches the current instance of " +
        "'$TaskPath$Name'."
    )
}

function Get-CurrentProcessRecord {
    param([int] $ProcessId)
    return @(
        Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
    ) | Select-Object -First 1
}

function New-ProcessIdentity {
    param(
        [object] $Process,
        [int] $Depth
    )
    return [PSCustomObject]@{
        ProcessId = [int]$Process.ProcessId
        ParentProcessId = [int]$Process.ParentProcessId
        CreationTime = (
            [datetime]$Process.CreationDate
        ).ToUniversalTime()
        ExecutablePath = [string]$Process.ExecutablePath
        Name = [string]$Process.Name
        Depth = $Depth
    }
}

function Test-ProcessRecordIdentity {
    param(
        [object] $Process,
        [object] $Identity
    )
    $currentCreation = (
        [datetime]$Process.CreationDate
    ).ToUniversalTime()
    $sameCreation = (
        [math]::Abs(
            ($currentCreation - $Identity.CreationTime).TotalMilliseconds
        ) -lt 1
    )
    $sameName = (
        [string]$Process.Name
    ).Equals(
        $Identity.Name,
        [StringComparison]::OrdinalIgnoreCase
    )
    $sameExecutable = (
        -not $Identity.ExecutablePath -or
        (
            [string]$Process.ExecutablePath
        ).Equals(
            $Identity.ExecutablePath,
            [StringComparison]::OrdinalIgnoreCase
        )
    )
    return $sameCreation -and $sameName -and $sameExecutable
}

function Test-ProcessIdentity {
    param([object] $Identity)
    $current = Get-CurrentProcessRecord $Identity.ProcessId
    if (-not $current) {
        return $false
    }
    return Test-ProcessRecordIdentity `
        -Process $current `
        -Identity $Identity
}

function Get-VerifiedProcessTree {
    param([object] $RootIdentity)
    $identities = @{
        $RootIdentity.ProcessId = $RootIdentity
    }
    for ($pass = 0; $pass -lt 8; $pass++) {
        $added = 0
        $snapshot = @(Get-CimInstance Win32_Process)
        $snapshotById = @{}
        foreach ($process in $snapshot) {
            $snapshotById[[int]$process.ProcessId] = $process
        }
        foreach ($process in $snapshot) {
            $processId = [int]$process.ProcessId
            $parentId = [int]$process.ParentProcessId
            if (
                $identities.ContainsKey($processId) -or
                -not $identities.ContainsKey($parentId)
            ) {
                continue
            }
            $parent = $snapshotById[$parentId]
            $parentIdentity = $identities[$parentId]
            if (
                -not $parent -or
                -not (
                    Test-ProcessRecordIdentity `
                        -Process $parent `
                        -Identity $parentIdentity
                )
            ) {
                continue
            }
            $creation = (
                [datetime]$process.CreationDate
            ).ToUniversalTime()
            if ($creation -lt $parentIdentity.CreationTime) {
                continue
            }
            $identities[$processId] = New-ProcessIdentity `
                -Process $process `
                -Depth ($parentIdentity.Depth + 1)
            $added += 1
        }
        if ($added -eq 0) {
            break
        }
    }
    return @($identities.Values)
}

$taskIsRunning = [string]$task.State -eq "Running"
if ($taskIsRunning -and -not $state.job_object_assigned) {
    throw (
        "The running task has no confirmed kill-on-close Job Object; " +
        "refusing an unsafe stop."
    )
}
if ($taskIsRunning -and -not $state.task_action_pid) {
    throw (
        "The running task has no recorded action process identity; " +
        "refusing an unsafe stop."
    )
}

$rootIdentity = $null
$rootMatches = $false
if (
    $state.task_action_pid -and
    $state.task_action_created_at -and
    $state.task_action_executable
) {
    $rootIdentity = [PSCustomObject]@{
        ProcessId = [int]$state.task_action_pid
        ParentProcessId = 0
        CreationTime = (
            [datetime]$state.task_action_created_at
        ).ToUniversalTime()
        ExecutablePath = [string]$state.task_action_executable
        Name = [IO.Path]::GetFileName(
            [string]$state.task_action_executable
        )
        Depth = 0
    }
    $rootMatches = Test-ProcessIdentity $rootIdentity
}
if ($taskIsRunning -and -not $rootMatches) {
    throw (
        "The running task's recorded action process identity does not match; " +
        "refusing to stop an unverified process tree."
    )
}

$verifiedTree = if ($rootMatches) {
    @(Get-VerifiedProcessTree $rootIdentity)
} else {
    @()
}
Stop-ScheduledTask -InputObject $task -ErrorAction Stop

if ($rootMatches) {
    $graceDeadline = [datetime]::UtcNow.AddSeconds(5)
    while (
        [datetime]::UtcNow -lt $graceDeadline -and
        (Test-ProcessIdentity $rootIdentity)
    ) {
        Start-Sleep -Milliseconds 100
    }
    if (Test-ProcessIdentity $rootIdentity) {
        throw (
            "Task Scheduler did not terminate the verified action process; " +
            "refusing a PID-based fallback."
        )
    }

    $jobDeadline = [datetime]::UtcNow.AddSeconds(5)
    do {
        $remaining = @(
            $verifiedTree |
                Where-Object { Test-ProcessIdentity $_ }
        )
        if (-not $remaining.Count) {
            break
        }
        Start-Sleep -Milliseconds 100
    } while ([datetime]::UtcNow -lt $jobDeadline)
    if ($remaining.Count) {
        throw (
            "The Job Object did not terminate verified process ids: " +
            (($remaining | ForEach-Object ProcessId) -join ", ")
        )
    }
}

$state.state = "stopped"
$stoppedAt = (Get-Date).ToString("o")
if ($state.PSObject.Properties["stopped_at"]) {
    $state.stopped_at = $stoppedAt
} else {
    $state | Add-Member `
        -NotePropertyName stopped_at `
        -NotePropertyValue $stoppedAt
}
$temporary = "$($manifest.FullName).stop-$PID.tmp"
$backup = "$($manifest.FullName).stop-$PID.bak"
$state | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $temporary -Encoding utf8
[IO.File]::Replace($temporary, $manifest.FullName, $backup)
Remove-Item -LiteralPath $backup -Force
Write-Host (
    "Stopped '$Name' through its Job Object " +
    "($($verifiedTree.Count) process identity record(s) verified)."
)
} finally {
    $taskLock.Dispose()
}
