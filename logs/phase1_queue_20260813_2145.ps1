# Phase 1 validation: 3 seeds under the v0.2 curriculum defaults.
#
# Directly comparable to the "mixed" arm of results/curriculum_v02.md, which
# used the same 200 updates / 16000 episodes / random,heuristic,simple
# curriculum on seeds 0,1,2 and scored 95.02% overall on the locked gauntlet.
# The only differences here are the played-cards state zones and gamma=1.0.
#
# Sequential by design: one GPU. Each seed writes its own log and a .DONE
# sentinel so a follow-up session can tell progress from completion.

$ErrorActionPreference = "Stop"
Set-Location E:\sdb-phase1

$py    = "E:\space-deck-builder\.venv\Scripts\python.exe"
$stamp = "20260813_2145"
$logs  = "E:\sdb-phase1\logs"

$env:PYTHONPATH = "E:\sdb-phase1"

foreach ($seed in 0, 1, 2) {
    $tag  = "phase1_seed${seed}_$stamp"
    $log  = Join-Path $logs "$tag.log"
    $done = Join-Path $logs "$tag.DONE"

    if (Test-Path $done) { continue }   # idempotent restart

    Set-Content -Path (Join-Path $logs "latest_phase1_phase.txt") -Value "seed$seed"
    Set-Content -Path (Join-Path $logs "latest_phase1_log.txt")   -Value $log

    & $py -u -m src train --seed $seed *> $log

    if ($LASTEXITCODE -ne 0) {
        Set-Content -Path (Join-Path $logs "$tag.FAILED") -Value "exit=$LASTEXITCODE"
        throw "seed $seed failed with exit code $LASTEXITCODE"
    }

    Set-Content -Path $done -Value "done"

    # Let ROCm fully release device memory and worker handles before the
    # next process starts.
    Start-Sleep -Seconds 90
}

Set-Content -Path (Join-Path $logs "phase1_queue_$stamp.DONE") -Value "done"
Set-Content -Path (Join-Path $logs "latest_phase1_phase.txt") -Value "complete"
