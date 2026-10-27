# Train one arm of decision 0013 (docs/decisions/0013-strength-yardstick.md)
# for the given seeds, and score each seed once both of its arms exist.
#
# Launch outside the agent process tree, from the repository root of a clean
# worktree pinned at the commit to run:
#   .\scripts\launch_training.ps1 -Name sdb_yardstick_control `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\strength_yardstick\run_arms.ps1 -Python {python} -Arm control -Seeds 0,1,2"
#
# Training resumes through train_reference.ps1's own manifest, a copied final
# checkpoint must match its source by hash, and a seed's score is kept only
# when its JSON holds every pairing, so re-running continues an interrupted
# launch.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [Parameter(Mandatory = $true)]
    [ValidateSet("control", "exploiter")]
    [string] $Arm,
    [string] $Seeds = "0,1,2",
    [string] $RunsDir = "D:\sdb-yardstick-runs",
    [int] $GamesPerPair = 4000,
    [int] $NumWorkers = 16
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$presets = @{
    control = "continue-selfplay-seed0"
    exploiter = "exploit-selfplay-seed0"
}
$published = Join-Path $Repo "results\optimized_selfplay_baseline\checkpoints"
$champion = Join-Path $published "selfplay_seed0.pth"
$peers = @(
    (Join-Path $published "selfplay_seed1.pth"),
    (Join-Path $published "selfplay_seed2.pth")
)
$checkpointsDir = Join-Path $RunsDir "checkpoints"
$scoresDir = Join-Path $RunsDir "scores"
New-Item -ItemType Directory -Force -Path $RunsDir, $checkpointsDir, $scoresDir | Out-Null
$seedList = @($Seeds -split "[,\s]+" | Where-Object { $_ } | ForEach-Object { [int]$_ })
$pairings = 10

$dirty = @(git -C $Repo status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "git status failed in $Repo"
}
if ($dirty.Count -gt 0) {
    throw "worktree $Repo has uncommitted changes; run from a clean commit"
}
$commit = (git -C $Repo rev-parse HEAD).Trim()

function Get-Sha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Copy-Checkpoint([string] $Source, [string] $Destination, [hashtable] $Provenance) {
    $digest = Get-Sha256 $Source
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        if ((Get-Sha256 $Destination) -ne $digest) {
            throw "$Destination exists with different contents than $Source"
        }
        return
    }
    Copy-Item -LiteralPath $Source -Destination "$Destination.tmp" -Force
    Move-Item -LiteralPath "$Destination.tmp" -Destination $Destination -Force
    $Provenance.sha256 = $digest
    $Provenance.source = $Source
    $Provenance | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath ([System.IO.Path]::ChangeExtension($Destination, ".json")) -Encoding utf8
}

function Test-Score([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $score = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        return $false
    }
    $complete = @($score.pairings | Where-Object { [int]$_.games -eq $GamesPerPair })
    return $complete.Count -eq $pairings
}

function Invoke-Python([string] $Name, [string[]] $Arguments) {
    $log = Join-Path $scoresDir "$Name.log"
    Push-Location $Repo
    $previous = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 turns redirected native stderr into
        # terminating errors under "Stop".
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 1>> $log 2>> "$log.err"
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
        Pop-Location
    }
    if ($code -ne 0) {
        throw "$Name exited with $code; see $log.err"
    }
}

Write-Host "[run_arms] arm=$Arm seeds=$($seedList -join ',') commit=$commit runs=$RunsDir"
foreach ($seed in $seedList) {
    $preset = $presets[$Arm]
    $runRoot = Join-Path $RunsDir "${preset}_seed$seed"
    Write-Host "[run_arms] $(Get-Date -Format s) train $preset seed $seed in $runRoot"
    & (Join-Path $Repo "scripts\train_reference.ps1") `
        -Foreground -Python $Python -Preset $preset -Seed $seed -RunRoot $runRoot
    $manifest = Get-Content -LiteralPath (Join-Path $runRoot "reference_manifest.json") -Raw |
        ConvertFrom-Json
    if ([string]$manifest.state -ne "complete") {
        throw "$preset seed $seed did not complete: $($manifest.failure)"
    }
    Copy-Checkpoint ([string]$manifest.final_checkpoint) `
        (Join-Path $checkpointsDir "${Arm}_seed$seed.pth") `
        @{ arm = $Arm; seed = $seed; preset = $preset; commit = $commit }

    $exploiter = Join-Path $checkpointsDir "exploiter_seed$seed.pth"
    $control = Join-Path $checkpointsDir "control_seed$seed.pth"
    $scorePath = Join-Path $scoresDir "score_seed$seed.json"
    if ((Test-Path -LiteralPath $exploiter) -and (Test-Path -LiteralPath $control) -and
            -not (Test-Score $scorePath)) {
        Write-Host "[run_arms] $(Get-Date -Format s) score seed $seed"
        $partial = Join-Path $scoresDir "score_seed$seed.partial.json"
        Remove-Item -LiteralPath $partial -ErrorAction SilentlyContinue
        Invoke-Python "score_seed$seed" @(
            "-u", "-m", "src", "elo",
            "--checkpoints", $champion, $peers[0], $peers[1], $exploiter, $control,
            "--checkpoint-temperatures", "0",
            "--games-per-pair", [string]$GamesPerPair,
            "--num-workers", [string]$NumWorkers,
            "--output-json", $partial
        )
        if (-not (Test-Score $partial)) {
            throw "score for seed $seed is incomplete: $partial"
        }
        Move-Item -LiteralPath $partial -Destination $scorePath -Force
    }
}
Write-Host "[run_arms] $(Get-Date -Format s) done"
