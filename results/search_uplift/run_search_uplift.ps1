# Reproduce results/search_uplift.md from the repository root.
#
# Launch outside the agent process tree, for example:
#   .\scripts\launch_training.ps1 -Name sdb_search_uplift `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\search_uplift\run_search_uplift.ps1 -Python {python}"
#
# Pre-registered design: the full budget ladder (0, 16, 64, 256) on
# selfplay_seed0, then budgets 0 and 64 on selfplay_seed1 and selfplay_seed2.
# Every arm plays the same checkpoint at temperature 0 as its baseline with
# exact seat alternation. Each run resumes from its games journal.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [string] $OutDir = "results\search_uplift",
    [int] $Games = 1000,
    [int] $Workers = 16,
    [int] $NumbaThreads = 1
)

$ErrorActionPreference = "Stop"
$checkpoints = "results\optimized_selfplay_baseline\checkpoints"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-SearchEval([string] $Seed, [string[]] $Budgets, [string] $Name) {
    $json = Join-Path $OutDir "$Name.json"
    if (Test-Path -LiteralPath $json) {
        Write-Host "skip $Name (complete)"
        return
    }
    $log = Join-Path $OutDir "$Name.log"
    $arguments = @(
        "-u", "-m", "src", "search-eval",
        "--model", (Join-Path $checkpoints "selfplay_seed$Seed.pth"),
        "--budgets"
    ) + $Budgets + @(
        "--games", [string]$Games,
        "--baseline", "greedy",
        "--seed", $Seed,
        "--workers", [string]$Workers,
        "--numba-threads", [string]$NumbaThreads,
        "--output-json", $json
    )
    Write-Host "search-eval $Name"
    $previous = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 turns redirected native stderr into
        # terminating errors under "Stop".
        $ErrorActionPreference = "Continue"
        & $Python @arguments 1>> $log 2>> "$log.err"
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "search-eval $Name exited with $code; see $log.err"
    }
}

Invoke-SearchEval "0" @("0", "16", "64", "256") "selfplay_seed0_ladder"
Invoke-SearchEval "1" @("0", "64") "selfplay_seed1_budget64"
Invoke-SearchEval "2" @("0", "64") "selfplay_seed2_budget64"
Write-Host "search uplift evidence complete"
