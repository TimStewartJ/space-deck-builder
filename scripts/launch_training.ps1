<#
.SYNOPSIS
    Launch training through Windows Task Scheduler with production placement.

.DESCRIPTION
    The Task Scheduler boundary keeps training outside the caller's process
    tree. The scheduled PowerShell process sets its priority and optional CPU
    affinity before starting Python; all training subprocesses inherit that
    placement. On the project's i7-12700K host, Auto selects logical CPUs 0-15
    so training uses the P-cores instead of saturating the four E-cores.

.PARAMETER TrainArgs
    Command-line arguments appended to `python -m src train`.

.PARAMETER Command
    A complete command line to run instead of `python -m src train`, such as
    an evaluation, tournament, or orchestration script. `{python}` is replaced
    with the resolved GPU-capable interpreter. Cannot be combined with
    TrainArgs.

.PARAMETER ProcessorAffinityMask
    Auto, None, a decimal mask, or a hexadecimal mask such as 0xFFFF.

.PARAMETER ReplaceExisting
    Replace an existing non-running task with the same name. Running tasks are
    never replaced.

.EXAMPLE
    .\scripts\launch_training.ps1 -TrainArgs "--seed 0 --updates 400"

.EXAMPLE
    .\scripts\launch_training.ps1 -Name sdb_eval_seed0 `
        -Command "{python} -u -m src eval --model results\run\final.pth --games 15000 --seed 0"
#>
[CmdletBinding()]
param(
    [string] $TrainArgs = "",
    [string] $Command = "",
    [string] $Repo = (Split-Path -Parent $PSScriptRoot),
    [string] $Python = "",
    [string] $Name = "sdb_train",
    [ValidateSet("Normal", "AboveNormal", "High")]
    [string] $ProcessPriority = "Normal",
    [string] $ProcessorAffinityMask = "Auto",
    [switch] $ReplaceExisting
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "training_process_placement.ps1")
. (Join-Path $PSScriptRoot "training_task_lock.ps1")

if ($Name -cnotmatch "^sdb_[A-Za-z0-9_.-]+$") {
    throw (
        "Task names must start with 'sdb_' and contain only letters, " +
        "numbers, dot, underscore, or hyphen."
    )
}
if ($TrainArgs -and $Command) {
    throw "Pass either -TrainArgs or -Command, not both."
}
$TaskPath = "\"
$Repo = (Resolve-Path $Repo).Path
$pythonCandidates = if ($Python) {
    @($Python)
} else {
    @(
        (Join-Path $Repo ".venv\Scripts\python.exe"),
        "E:\space-deck-builder\.venv\Scripts\python.exe"
    )
}
$python = $null
$pythonFailures = @()
foreach ($candidate in $pythonCandidates | Select-Object -Unique) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        $pythonFailures += "$candidate (missing)"
        continue
    }
    & $candidate -c "import torch; assert torch.cuda.is_available()" `
        *> $null
    if ($LASTEXITCODE -eq 0) {
        $python = (Resolve-Path $candidate).Path
        break
    }
    $pythonFailures += "$candidate (torch/ROCm unavailable)"
}
if (-not $python) {
    throw (
        "No GPU-capable interpreter found. Run uv sync --extra rocm. " +
        "Checked: $($pythonFailures -join '; ')"
    )
}
$resolvedMask = Resolve-TrainingAffinityMask $ProcessorAffinityMask
$affinityLabel = if ($null -eq $resolvedMask) {
    "all"
} else {
    "0x{0:X}" -f $resolvedMask
}
$taskLock = Enter-TrainingTaskLock -Name $Name
try {
$existingTask = Get-ScheduledTask `
    -TaskPath $TaskPath `
    -TaskName $Name `
    -ErrorAction SilentlyContinue
if ($existingTask) {
    if (-not $ReplaceExisting) {
        throw (
            "Scheduled task '$Name' already exists. Choose a unique -Name, " +
            "unregister it, or pass -ReplaceExisting."
        )
    }
    if ([string]$existingTask.State -eq "Running") {
        throw "Scheduled task '$Name' is running and cannot be replaced."
    }
    Unregister-ScheduledTask -InputObject $existingTask -Confirm:$false
}

$logDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "$Name`_$stamp.log"
$launchManifest = Join-Path $logDir "$Name`_$stamp.json"
$launchId = [guid]::NewGuid().ToString("D")
$taskDescription = "Space Deck Builder launch $launchId"

$workload = if ($Command) {
    $Command.Replace("{python}", "`"$python`"")
} else {
    "`"$python`" -u -m src train $TrainArgs"
}
# Group the workload so the log captures every command of a chained
# -Command (for example "a && b"), not only the last one.
$taskCommand = (
    "set `"PYTHONPATH=$Repo`" && cd /d `"$Repo`" && " +
    "($workload) > `"$log`" 2>&1"
)
$encodedCommand = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($taskCommand)
)
$encodedManifest = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes($launchManifest)
)
$runner = @"
`$ErrorActionPreference = "Stop"
`$command = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String("$encodedCommand")
)
`$manifestPath = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String("$encodedManifest")
)
function Save-LaunchState(`$state) {
    `$temporary = "`$manifestPath.action-`$PID.tmp"
    `$backup = "`$manifestPath.action-`$PID.bak"
    `$state | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath `$temporary -Encoding utf8
    [IO.File]::Replace(`$temporary, `$manifestPath, `$backup)
    Remove-Item -LiteralPath `$backup -Force
}
`$state = Get-Content -LiteralPath `$manifestPath -Raw | ConvertFrom-Json
`$jobHandle = [IntPtr]::Zero
`$exitCode = 1
try {
    if (`$state.state -ne "registered") {
        throw (
            "Launch manifest is not startable from state '`$(`$state.state)'."
        )
    }
    `$jobSource = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class SdbTrainingJob
{
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    private const uint PROCESS_TERMINATE = 0x0001;
    private const uint PROCESS_SET_QUOTA = 0x0100;
    private const int JobObjectExtendedLimitInformation = 9;

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(
        IntPtr jobAttributes,
        string name
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        int infoClass,
        ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION info,
        uint infoLength
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(
        uint desiredAccess,
        bool inheritHandle,
        uint processId
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(
        IntPtr job,
        IntPtr process
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    public static IntPtr CreateAndAssign(uint processId)
    {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        try
        {
            var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            uint size = (uint)Marshal.SizeOf(info);
            if (!SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ref info,
                size
            ))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }

            IntPtr process = OpenProcess(
                PROCESS_TERMINATE | PROCESS_SET_QUOTA,
                false,
                processId
            );
            if (process == IntPtr.Zero)
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            try
            {
                if (!AssignProcessToJobObject(job, process))
                {
                    throw new Win32Exception(Marshal.GetLastWin32Error());
                }
            }
            finally
            {
                CloseHandle(process);
            }
            return job;
        }
        catch
        {
            CloseHandle(job);
            throw;
        }
    }

}
'@
    Add-Type -TypeDefinition `$jobSource -Language CSharp
    `$jobHandle = [SdbTrainingJob]::CreateAndAssign([uint32]`$PID)
    `$process = Get-Process -Id `$PID
    `$process.PriorityClass = "$ProcessPriority"
$(if ($null -ne $resolvedMask) {
    "    `$process.ProcessorAffinity = [IntPtr]$resolvedMask"
})
    `$actionProcess = Get-CimInstance Win32_Process -Filter "ProcessId=`$PID"
    `$state.state = "running"
    `$state.job_object_assigned = `$true
    `$state.task_action_pid = `$PID
    `$state.task_action_created_at = `$actionProcess.CreationDate.ToString("o")
    `$state.task_action_executable = `$actionProcess.ExecutablePath
    `$state.started_at = (Get-Date).ToString("o")
    Save-LaunchState `$state
    & `$env:ComSpec /d /s /c `$command
    `$exitCode = `$LASTEXITCODE
    `$state.state = if (`$exitCode -eq 0) { "complete" } else { "failed" }
} catch {
    `$state.state = "failed"
    `$state.error_message = `$_.Exception.Message
} finally {
    `$state.exit_code = `$exitCode
    `$state.finished_at = (Get-Date).ToString("o")
    Save-LaunchState `$state
}
exit `$exitCode
"@
$encodedRunner = [Convert]::ToBase64String(
    [Text.Encoding]::Unicode.GetBytes($runner)
)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedRunner"
$settings = New-ScheduledTaskSettingsSet `
    -Priority 4 `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskPath $TaskPath `
    -TaskName $Name `
    -Description $taskDescription `
    -Action $action `
    -Settings $settings |
    Out-Null
@{
    schema_version = 2
    launch_id = $launchId
    task = $Name
    task_path = $TaskPath
    repository = $Repo
    python = $python
    train_args = $TrainArgs
    command = $workload
    process_priority = $ProcessPriority
    processor_affinity = $affinityLabel
    log = $log
    state = "registered"
    job_object_assigned = $false
    task_action_pid = $null
    task_action_created_at = $null
    task_action_executable = $null
    started_at = $null
    finished_at = $null
    exit_code = $null
    error_message = $null
    launched_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $launchManifest -Encoding utf8
Start-ScheduledTask -TaskPath $TaskPath -TaskName $Name
$started = $false
$startupFailure = $null
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 250
    try {
        $launchState = Get-Content `
            -LiteralPath $launchManifest `
            -Raw |
            ConvertFrom-Json
    } catch {
        # The scheduled action rewrites this manifest as it starts, so a read
        # can meet a sharing violation or a partial write. Keep polling until
        # the handshake times out.
        $startupFailure = (
            "Could not read the startup manifest: " +
            $_.Exception.Message
        )
        continue
    }
    $startupFailure = $null
    if ($launchState.state -in @("running", "complete")) {
        $started = $true
        break
    }
    if ($launchState.state -eq "failed") {
        $startupFailure = (
            "The scheduled action failed during launch: " +
            $launchState.error_message
        )
        break
    }
}
if (-not $started) {
    $cancellationMessage = if ($startupFailure) {
        $startupFailure
    } else {
        "Startup handshake timed out."
    }
    $temporary = "$launchManifest.launcher-$PID.tmp"
    $backup = "$launchManifest.launcher-$PID.bak"
    $cancelWriteError = $null
    $cleanupErrors = @()
    try {
        $launchState = Get-Content `
            -LiteralPath $launchManifest `
            -Raw |
            ConvertFrom-Json
        $launchState.state = "cancelled"
        $launchState.error_message = $cancellationMessage
        $launchState.finished_at = (Get-Date).ToString("o")
        $launchState | ConvertTo-Json -Depth 6 |
            Set-Content -LiteralPath $temporary -Encoding utf8
        [IO.File]::Replace($temporary, $launchManifest, $backup)
        Remove-Item -LiteralPath $backup -Force
    } catch {
        $cancelWriteError = $_.Exception.Message
    } finally {
        try {
            Disable-ScheduledTask `
                -TaskPath $TaskPath `
                -TaskName $Name |
                Out-Null
        } catch {
            $cleanupErrors += $_.Exception.Message
        }
        try {
            Stop-ScheduledTask `
                -TaskPath $TaskPath `
                -TaskName $Name `
                -ErrorAction Stop
        } catch {
            $cleanupErrors += $_.Exception.Message
        }
    }
    for ($attempt = 0; $attempt -lt 50; $attempt++) {
        $taskState = (
            Get-ScheduledTask `
                -TaskPath $TaskPath `
                -TaskName $Name
        ).State
        if ([string]$taskState -ne "Running") {
            break
        }
        Start-Sleep -Milliseconds 100
    }
    try {
        $launchState = Get-Content `
            -LiteralPath $launchManifest `
            -Raw |
            ConvertFrom-Json
        $launchState.state = "cancelled"
        $launchState.error_message = $cancellationMessage
        $launchState.finished_at = (Get-Date).ToString("o")
        $launchState | ConvertTo-Json -Depth 6 |
            Set-Content -LiteralPath $temporary -Encoding utf8
        [IO.File]::Replace($temporary, $launchManifest, $backup)
        Remove-Item -LiteralPath $backup -Force
    } catch {
        if (-not $cancelWriteError) {
            $cancelWriteError = $_.Exception.Message
        }
    }
    if ([string]$taskState -eq "Running") {
        throw (
            "Scheduled task '$Name' did not complete its startup handshake " +
            "and could not be stopped within five seconds."
        )
    }
    try {
        Unregister-ScheduledTask `
            -TaskPath $TaskPath `
            -TaskName $Name `
            -Confirm:$false
    } catch {
        $cleanupErrors += $_.Exception.Message
    }
    if ($cancelWriteError -or $cleanupErrors.Count) {
        throw (
            "Scheduled task '$Name' was stopped after its startup timeout, " +
            "but cleanup reported: " +
            (@($cancelWriteError) + $cleanupErrors -join "; ")
        )
    }
    throw (
        "Scheduled task '$Name' was cancelled: $cancellationMessage"
    )
}

Write-Host (
    "Launched '$Name' through Task Scheduler " +
    "(priority=$ProcessPriority, affinity=$affinityLabel)."
)
Write-Host "  log:    $log"
Write-Host "  status: Get-ScheduledTask -TaskPath '\' -TaskName '$Name'"
Write-Host "  follow: Get-Content '$log' -Wait -Tail 20"
Write-Host (
    "  stop:   .\scripts\stop_training.ps1 -Name '$Name' " +
    "-Repo '$Repo'"
)
} finally {
    $taskLock.Dispose()
}
