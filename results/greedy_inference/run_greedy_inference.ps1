# Reproduce results/greedy_inference.md from the repository root.
#
# Launch outside the agent process tree, for example:
#   .\scripts\launch_training.ps1 -Name sdb_greedy_inference `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\greedy_inference\run_greedy_inference.ps1 -Python {python}"
#
# Runs the locked 15,000-game exact-seat gauntlet at temperature 0 for the
# six published v0.3 checkpoints, then one exact-seat tournament containing
# every checkpoint at temperatures 1 and 0 plus the fixed agents.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [string] $OutDir = "results\greedy_inference",
    [int] $NumWorkers = 16
)

$ErrorActionPreference = "Stop"
$checkpoints = "results\optimized_selfplay_baseline\checkpoints"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-Logged([string[]] $Arguments, [string] $Log) {
    $previous = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 turns redirected native stderr into
        # terminating errors under "Stop".
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 1> $Log 2> "$Log.err"
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "python $($Arguments -join ' ') exited with $code; see $Log.err"
    }
}

foreach ($league in @("selfplay", "fixed")) {
    foreach ($seed in 0, 1, 2) {
        $name = "${league}_seed$seed"
        $json = Join-Path $OutDir "${name}_T0.json"
        if (Test-Path -LiteralPath $json) {
            Write-Host "skip $name (complete)"
            continue
        }
        Write-Host "gauntlet $name T=0"
        Invoke-Logged @(
            "-u", "-m", "src", "eval",
            "--model", (Join-Path $checkpoints "$name.pth"),
            "--opponents", "random,heuristic,simple",
            "--games", "15000",
            "--seed", [string]$seed,
            "--num-workers", [string]$NumWorkers,
            "--temperature", "0",
            "--output-json", $json
        ) (Join-Path $OutDir "${name}_T0.log")
    }
}

$eloJson = Join-Path $OutDir "elo.json"
if (-not (Test-Path -LiteralPath $eloJson)) {
    Write-Host "tournament: six checkpoints at temperatures 1 and 0"
    Invoke-Logged @(
        "-u", "-m", "src", "elo",
        "--checkpoints",
        (Join-Path $checkpoints "fixed_seed0.pth"),
        (Join-Path $checkpoints "fixed_seed1.pth"),
        (Join-Path $checkpoints "fixed_seed2.pth"),
        (Join-Path $checkpoints "selfplay_seed0.pth"),
        (Join-Path $checkpoints "selfplay_seed1.pth"),
        (Join-Path $checkpoints "selfplay_seed2.pth"),
        "--checkpoint-temperatures", "1,0",
        "--agents", "random,heuristic,simple",
        "--games-per-pair", "1000",
        "--num-workers", [string]$NumWorkers,
        "--output-json", $eloJson
    ) (Join-Path $OutDir "elo.log")
}
Write-Host "greedy inference evidence complete"
