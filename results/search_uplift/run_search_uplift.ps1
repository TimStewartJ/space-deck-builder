# Reproduce results/search_uplift.md from the repository root.
#
# Launch outside the agent process tree, for example:
#   .\scripts\launch_training.ps1 -Name sdb_search_uplift `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\search_uplift\run_search_uplift.ps1 -Python {python}"
#
# Pre-registered design: the full budget ladder (0, 16, 64, 256) on
# selfplay_seed0 with 1,000 games per budget, then budgets 0 and 64 on
# selfplay_seed1 and selfplay_seed2 with 500 games per budget. Every arm plays
# the same checkpoint at temperature 0 as its baseline with exact seat
# alternation. Each run resumes from its games journal.
#
# The published evidence was produced at commit 1849b19 with eight
# unbatched workers. Since docs/decisions/0011, search seeds come from public
# root digests and worlds are dealt by a counter-based hash, so rerunning this
# script samples different search worlds on the same deals. Topology settings
# only change speed; two workers with 1,024 concurrent games each measured
# fastest on the RX 9070 host.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [string] $OutDir = "results\search_uplift",
    [int] $Games = 1000,
    [int] $ReplicationGames = 500,
    [int] $Workers = 2,
    [int] $NumbaThreads = 4,
    [int] $BatchSize = 1024,
    [string] $Device = "cuda"
)

$ErrorActionPreference = "Stop"
$checkpoints = "results\optimized_selfplay_baseline\checkpoints"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-SearchEval([string] $Seed, [string[]] $Budgets, [string] $Name, [int] $GameCount) {
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
        "--games", [string]$GameCount,
        "--baseline", "greedy",
        "--seed", $Seed,
        "--workers", [string]$Workers,
        "--numba-threads", [string]$NumbaThreads,
        "--batch-size", [string]$BatchSize,
        "--simulation-device", $Device,
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

Invoke-SearchEval "0" @("0", "16", "64", "256") "selfplay_seed0_ladder" $Games
Invoke-SearchEval "1" @("0", "64") "selfplay_seed1_budget64" $ReplicationGames
Invoke-SearchEval "2" @("0", "64") "selfplay_seed2_budget64" $ReplicationGames
Write-Host "search uplift evidence complete"
