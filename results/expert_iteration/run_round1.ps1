# Run the pre-registered expert iteration round 1 (docs/decisions/0012).
#
# Launch outside the agent process tree, from the repository root:
#   .\scripts\launch_training.ps1 -Name sdb_expert_iteration_round1 `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\expert_iteration\run_round1.ps1 -Python {python}"
#
# Every step skips work whose output already exists, and self-play resumes
# from its shards, so re-running continues an interrupted round.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [string] $WorkDir = "logs\expert_iteration\round1",
    [string] $OutDir = "results\expert_iteration\round1",
    [int] $Workers = 2,
    [int] $BatchSize = 1024,
    [int] $NumbaThreads = 4
)

$ErrorActionPreference = "Stop"
$source = "results\optimized_selfplay_baseline\checkpoints\selfplay_seed0.pth"
$data = Join-Path $WorkDir "data"
$student = Join-Path $WorkDir "distilled_seed0.pth"
New-Item -ItemType Directory -Force -Path $WorkDir, $OutDir | Out-Null

function Invoke-Step([string] $Name, [string[]] $Arguments) {
    $log = Join-Path $OutDir "$Name.log"
    Write-Host "[round1] $Name"
    $previous = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 turns redirected native stderr into
        # terminating errors under "Stop".
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 1>> $log 2>> "$log.err"
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "$Name exited with $code; see $log.err"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $data "summary.json"))) {
    Invoke-Step "selfplay" @(
        "-u", "-m", "src", "search-selfplay",
        "--model", $source, "--games", "8192", "--budget", "64",
        "--max-candidates", "8", "--seed", "0",
        "--workers", [string]$Workers, "--batch-size", [string]$BatchSize,
        "--numba-threads", [string]$NumbaThreads, "--output-dir", $data
    )
}
Copy-Item -LiteralPath (Join-Path $data "summary.json") -Destination (Join-Path $OutDir "selfplay_summary.json") -Force

if (-not (Test-Path -LiteralPath $student)) {
    Invoke-Step "distill" @(
        "-u", "-m", "src", "distill",
        "--model", $source, "--data", $data, "--output", $student,
        "--lr", "1e-4", "--batch-size", "8192", "--epochs", "8", "--patience", "2",
        "--value-coef", "0.5", "--grad-clip", "0.5", "--validation-modulus", "20",
        "--seed", "0"
    )
}
Copy-Item -LiteralPath "$student.history.json" -Destination (Join-Path $OutDir "distill_history.json") -Force

if (-not (Test-Path -LiteralPath (Join-Path $OutDir "gate_elo_T0.json"))) {
    Invoke-Step "gate_elo_T0" @(
        "-u", "-m", "src", "elo",
        "--checkpoints", $student, $source,
        "--checkpoint-temperatures", "0",
        "--games-per-pair", "2000", "--num-workers", "16",
        "--output-json", (Join-Path $OutDir "gate_elo_T0.json")
    )
}

if (-not (Test-Path -LiteralPath (Join-Path $OutDir "gate_gauntlet_T0.json"))) {
    Invoke-Step "gate_gauntlet_T0" @(
        "-u", "-m", "src", "eval",
        "--model", $student, "--opponents", "random,heuristic,simple",
        "--games", "15000", "--seed", "0", "--num-workers", "16",
        "--temperature", "0", "--output-json", (Join-Path $OutDir "gate_gauntlet_T0.json")
    )
}

if (-not (Test-Path -LiteralPath (Join-Path $OutDir "diagnostic_search_eval.json"))) {
    Invoke-Step "diagnostic_search_eval" @(
        "-u", "-m", "src", "search-eval",
        "--model", $student, "--budgets", "0", "64", "--games", "1000", "--seed", "0",
        "--workers", [string]$Workers, "--batch-size", [string]$BatchSize,
        "--numba-threads", [string]$NumbaThreads,
        "--output-json", (Join-Path $OutDir "diagnostic_search_eval.json")
    )
}
Write-Host "[round1] complete"
