# Train decision 0014's exploiter from random weights against the champion's
# ladder (docs/decisions/0014-ladder-exploiter.md) for the given seeds, and
# score each seed once its checkpoint exists. With -League, play the held-out
# league for every ladder checkpoint already trained instead; run that only
# after every training lane has stopped.
#
# Launch outside the agent process tree, from the repository root of a clean
# worktree pinned at the commit to run, with the ladder checkpoints in
# results\strength_yardstick\ladder (published as release assets):
#   .\scripts\launch_training.ps1 -Name sdb_ladder_seed0 `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\strength_yardstick\run_ladder.ps1 -Python {python} -Seeds 0"
#
# Training resumes through train_reference.ps1's own manifest, a copied final
# checkpoint must match its source by hash, a seed's score is kept only when
# its JSON holds every pairing, and the league resumes from its journal, so
# re-running continues an interrupted launch.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [string] $Seeds = "0,1,2",
    [string] $RunsDir = "D:\sdb-yardstick-runs",
    [int] $GamesPerPair = 4000,
    [int] $NumWorkers = 16,
    [int] $LeagueWorkers = 6,
    [switch] $League
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$preset = "exploit-ladder-random"
$published = Join-Path $Repo "results\optimized_selfplay_baseline\checkpoints"
$champion = Join-Path $published "selfplay_seed0.pth"
$peers = @(
    (Join-Path $published "selfplay_seed1.pth"),
    (Join-Path $published "selfplay_seed2.pth")
)
$checkpointsDir = Join-Path $RunsDir "checkpoints"
$scoresDir = Join-Path $RunsDir "scores"
$leagueDir = Join-Path $RunsDir "league"
New-Item -ItemType Directory -Force -Path $RunsDir, $checkpointsDir, $scoresDir, $leagueDir | Out-Null
$seedList = @($Seeds -split "[,\s]+" | Where-Object { $_ } | ForEach-Object { [int]$_ })
$pairings = 6

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

# The ladder is git-ignored; every rung must be the recorded file.
Get-ChildItem -LiteralPath (Join-Path $Repo "results\strength_yardstick\ladder") -Filter "*.json" | ForEach-Object {
    $provenance = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
    $checkpoint = [System.IO.Path]::ChangeExtension($_.FullName, ".pth")
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "ladder checkpoint missing: $checkpoint"
    }
    if ((Get-Sha256 $checkpoint) -ne [string]$provenance.sha256) {
        throw "ladder checkpoint $checkpoint does not match its recorded SHA-256"
    }
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

function Invoke-Python([string] $Name, [string] $Directory, [string[]] $Arguments) {
    $log = Join-Path $Directory "$Name.log"
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

if ($League) {
    $models = @(Get-ChildItem -LiteralPath $checkpointsDir -Filter "ladder_seed*.pth" | Sort-Object Name |
        ForEach-Object { $_.FullName })
    if ($models.Count -eq 0) {
        throw "no ladder checkpoints in $checkpointsDir"
    }
    Write-Host "[run_ladder] $(Get-Date -Format s) commit=$commit league of $($models.Count) checkpoints"
    Invoke-Python "league_ladder" $leagueDir (
        @("-u", "-m", "src", "league", "--workers", [string]$LeagueWorkers,
          "--output-json", (Join-Path $leagueDir "league_ladder.json"), "--models") + $models
    )
    Write-Host "[run_ladder] $(Get-Date -Format s) done"
    return
}

Write-Host "[run_ladder] seeds=$($seedList -join ',') commit=$commit runs=$RunsDir"
foreach ($seed in $seedList) {
    $runRoot = Join-Path $RunsDir "${preset}_seed$seed"
    Write-Host "[run_ladder] $(Get-Date -Format s) train $preset seed $seed in $runRoot"
    & (Join-Path $Repo "scripts\train_reference.ps1") `
        -Foreground -Python $Python -Preset $preset -Seed $seed -RunRoot $runRoot
    $manifest = Get-Content -LiteralPath (Join-Path $runRoot "reference_manifest.json") -Raw |
        ConvertFrom-Json
    if ([string]$manifest.state -ne "complete") {
        throw "$preset seed $seed did not complete: $($manifest.failure)"
    }
    $ladder = Join-Path $checkpointsDir "ladder_seed$seed.pth"
    Copy-Checkpoint ([string]$manifest.final_checkpoint) $ladder `
        @{ arm = "ladder"; seed = $seed; preset = $preset; commit = $commit }

    $scorePath = Join-Path $scoresDir "ladder_score_seed$seed.json"
    if (-not (Test-Score $scorePath)) {
        Write-Host "[run_ladder] $(Get-Date -Format s) score seed $seed"
        $partial = Join-Path $scoresDir "ladder_score_seed$seed.partial.json"
        Remove-Item -LiteralPath $partial -ErrorAction SilentlyContinue
        Invoke-Python "ladder_score_seed$seed" $scoresDir @(
            "-u", "-m", "src", "elo",
            "--checkpoints", $champion, $peers[0], $peers[1], $ladder,
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
Write-Host "[run_ladder] $(Get-Date -Format s) done"
