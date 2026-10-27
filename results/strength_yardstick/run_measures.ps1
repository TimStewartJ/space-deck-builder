# Run the held-out league and the purchase audit of decision 0013
# (docs/decisions/0013-strength-yardstick.md).
#
# Launch outside the agent process tree, from the repository root of a clean
# worktree pinned at the commit to run:
#   .\scripts\launch_training.ps1 -Name sdb_yardstick_measures `
#       -Command "powershell.exe -NoProfile -ExecutionPolicy Bypass -File results\strength_yardstick\run_measures.ps1 -Python {python}"
#
# The league covers the published checkpoints, the expert-iteration round 1
# checkpoint, and every arm checkpoint already copied to <RunsDir>\checkpoints;
# the champion's counts are then checked against the prototype. The audit
# covers the three published self-play seeds. Both resume from their
# journals, so re-running after more arms finish only plays what is new.
param(
    [string] $Python = ".venv\Scripts\python.exe",
    [string] $RunsDir = "D:\sdb-yardstick-runs",
    [string] $Distilled = "D:\space-deck-builder-archive\expert-iteration-round1-20260916\work\distilled_seed0.pth",
    [int] $LeagueWorkers = 6,
    [int] $AuditThreads = 12,
    [switch] $SkipAudit
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$published = Join-Path $Repo "results\optimized_selfplay_baseline\checkpoints"
$leagueDir = Join-Path $RunsDir "league"
$auditDir = Join-Path $RunsDir "audit"
New-Item -ItemType Directory -Force -Path $leagueDir, $auditDir | Out-Null

$dirty = @(git -C $Repo status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -gt 0) {
    throw "worktree $Repo must be a clean checkout"
}
$commit = (git -C $Repo rev-parse HEAD).Trim()

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
    return $code
}

$models = @(
    (Join-Path $published "selfplay_seed0.pth"),
    (Join-Path $published "selfplay_seed1.pth"),
    (Join-Path $published "selfplay_seed2.pth"),
    (Join-Path $published "fixed_seed0.pth"),
    (Join-Path $published "fixed_seed1.pth"),
    (Join-Path $published "fixed_seed2.pth"),
    $Distilled
)
$arms = Join-Path $RunsDir "checkpoints"
if (Test-Path -LiteralPath $arms) {
    $models += @(Get-ChildItem -LiteralPath $arms -Filter "*.pth" | Sort-Object Name | ForEach-Object { $_.FullName })
}
Write-Host "[run_measures] $(Get-Date -Format s) commit=$commit league of $($models.Count) checkpoints"
$code = Invoke-Python "league" $leagueDir (
    @("-u", "-m", "src", "league", "--calibration", "--workers", [string]$LeagueWorkers,
      "--output-json", (Join-Path $leagueDir "league.json"), "--models") + $models
)
if ($code -ne 0) {
    throw "league exited with $code; see $(Join-Path $leagueDir 'league.log.err')"
}
$code = Invoke-Python "port_check" $leagueDir @(
    "results\strength_yardstick\check_port.py",
    (Join-Path $leagueDir "league.json"),
    (Join-Path $leagueDir "port_check.json")
)
Write-Host "[run_measures] $(Get-Date -Format s) port check exit code $code"

if (-not $SkipAudit) {
    foreach ($seed in 0, 1, 2) {
        $output = Join-Path $auditDir "audit_selfplay_seed$seed.json"
        Write-Host "[run_measures] $(Get-Date -Format s) purchase audit selfplay_seed$seed"
        $code = Invoke-Python "audit_selfplay_seed$seed" $auditDir @(
            "-u", "-m", "src", "purchase-audit",
            "--model", (Join-Path $published "selfplay_seed$seed.pth"),
            "--numba-threads", [string]$AuditThreads,
            "--output-json", $output
        )
        if ($code -ne 0) {
            throw "purchase audit of selfplay_seed$seed exited with $code"
        }
    }
}
Write-Host "[run_measures] $(Get-Date -Format s) done"
