<#
.SYNOPSIS
Train queued reference-style arcs on a fixed number of lanes.

.DESCRIPTION
Each lane is one Task Scheduler run of this script. A lane claims the first
pending job file in the queue folder by renaming it, trains the job to
completion with train_reference.ps1 -Foreground, records the outcome in the
file name, and claims the next job, until none is pending. Several lanes
share one queue, so arcs of different lengths keep every lane busy.

A job is a JSON file whose name orders the queue:

    { "preset": "reference-selfplay", "seed": 0,
      "run_root": "D:\\runs\\r1600_seed0",
      "parameters": { "Updates": 1600 } }

"parameters" are passed to train_reference.ps1 by name (Updates, ChunkSize,
Episodes, InterimEvalGames, FinalEvalGames, NumWorkers, EvalNumWorkers).

File states: job.json is pending, job.json.<lane>.running is claimed,
job.json.done is complete, job.json.failed has failed. A lane started again
first resumes the jobs it had claimed; train_reference.ps1 then resumes each
run root from its manifest.

Start lanes outside the agent process tree, for example three lanes:

    foreach ($lane in 1..3) {
        .\scripts\launch_training.ps1 -Name "sdb_queue_lane$lane" -Command (
            "powershell -NoProfile -ExecutionPolicy Bypass -File " +
            "scripts\run_queue.ps1 -Queue D:\runs\queue -Lane lane$lane " +
            "-Python {python}")
    }

On the project's machine three lanes saturate the GPU; five or more
oversubscribe the performance cores (results/training_throughput.md).
#>
param(
    [Parameter(Mandatory = $true)] [string] $Queue,
    [Parameter(Mandatory = $true)] [string] $Lane,
    [string] $Python = ""
)

$ErrorActionPreference = "Stop"
if ($Lane -notmatch "^[A-Za-z0-9_-]+$") {
    throw "Lane names contain only letters, digits, underscores, and hyphens."
}
$Queue = (Resolve-Path -LiteralPath $Queue).Path
$trainReference = Join-Path $PSScriptRoot "train_reference.ps1"
$allowed = @(
    "Updates", "ChunkSize", "Episodes", "InterimEvalGames",
    "FinalEvalGames", "NumWorkers", "EvalNumWorkers"
)

function Write-Lane([string] $Message) {
    Write-Host "[run_queue $Lane $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

function Get-Claim {
    # Filter by full name: Win32 wildcards also match 8.3 short names.
    $mine = Get-ChildItem -LiteralPath $Queue -File |
        Where-Object { $_.Name -like "*.json.$Lane.running" } |
        Sort-Object Name | Select-Object -First 1
    if ($mine) {
        return $mine.FullName
    }
    foreach ($job in Get-ChildItem -LiteralPath $Queue -File |
            Where-Object { $_.Name -like "*.json" } |
            Sort-Object Name) {
        $claimed = "$($job.FullName).$Lane.running"
        try {
            # A rename either succeeds for exactly one lane or fails.
            Move-Item -LiteralPath $job.FullName -Destination $claimed
            return $claimed
        } catch {
            continue
        }
    }
    return $null
}

function Invoke-Job([string] $Claimed) {
    $job = Get-Content -LiteralPath $Claimed -Raw | ConvertFrom-Json
    if (-not $job.preset -or $null -eq $job.seed -or -not $job.run_root) {
        throw "job needs preset, seed and run_root"
    }
    $arguments = @{
        Foreground = $true
        Preset = [string]$job.preset
        Seed = [int]$job.seed
        RunRoot = [string]$job.run_root
    }
    if ($Python) {
        $arguments.Python = $Python
    }
    if ($job.parameters) {
        foreach ($property in $job.parameters.PSObject.Properties) {
            if ($allowed -notcontains $property.Name) {
                throw "unsupported job parameter: $($property.Name)"
            }
            $arguments[$property.Name] = [int]$property.Value
        }
    }
    & $trainReference @arguments
    $manifest = Join-Path ([string]$job.run_root) "reference_manifest.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "train_reference.ps1 wrote no manifest at $manifest"
    }
    $state = [string](Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).state
    if ($state -ne "complete") {
        throw "run root $($job.run_root) ended in state '$state'"
    }
}

while ($true) {
    $claimed = Get-Claim
    if (-not $claimed) {
        Write-Lane "no pending jobs"
        break
    }
    $base = $claimed.Substring(0, $claimed.Length - ".$Lane.running".Length)
    Write-Lane "training $(Split-Path -Leaf $base)"
    try {
        Invoke-Job $claimed
        Move-Item -LiteralPath $claimed -Destination "$base.done" -Force
        Write-Lane "finished $(Split-Path -Leaf $base)"
    } catch {
        Move-Item -LiteralPath $claimed -Destination "$base.failed" -Force
        Write-Lane "FAILED $(Split-Path -Leaf $base): $_"
    }
}
