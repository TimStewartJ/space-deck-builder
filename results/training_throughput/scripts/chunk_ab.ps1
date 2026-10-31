$ErrorActionPreference = 'Stop'
Set-Location D:\sdb-perf
$interference = 'D:\sdb-bench\chunk_ab_interference.log'
Remove-Item $interference -ErrorAction SilentlyContinue
foreach ($arm in @(@{ Name = 'c25'; Chunk = 25 }, @{ Name = 'c100'; Chunk = 100 })) {
    $root = "D:\sdb-bench\queue_$($arm.Name)"
    Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force "$root\jobs" | Out-Null
    foreach ($i in 0..2) {
        $seed = 10 + $i
        $job = [ordered]@{
            preset = 'reference-selfplay'; seed = $seed; run_root = "$root\runs\seed$seed"
            parameters = [ordered]@{ Updates = 100; ChunkSize = $arm.Chunk }
        }
        $job | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$root\jobs\0${i}_seed$seed.json" -Encoding utf8NoBOM
    }
    $start = Get-Date
    $names = @()
    foreach ($lane in 1..3) {
        $name = "sdb_queue_$($arm.Name)_lane$lane"
        .\scripts\launch_training.ps1 -Name $name -Repo D:\sdb-perf -Command (
            "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_queue.ps1 " +
            "-Queue $root\jobs -Lane lane$lane -Python {python}") -ReplaceExisting *>&1 | Out-Null
        $names += $name
    }
    Start-Sleep -Seconds 20
    while (@($names | Where-Object { (Get-ScheduledTask -TaskName $_).State -eq 'Running' }).Count) {
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'pytest|sdb-rules' } |
            ForEach-Object { "$(Get-Date -Format HH:mm:ss) $($arm.Name) pid=$($_.ProcessId) $($_.CommandLine)" } |
            Add-Content -LiteralPath $interference
        Start-Sleep -Seconds 10
    }
    $end = Get-Date
    "{0}: {1:N2} min ({2:HH:mm:ss}-{3:HH:mm:ss}), jobs: {4}" -f $arm.Name, ($end - $start).TotalMinutes, $start, $end,
        ((Get-ChildItem "$root\jobs" -File | ForEach-Object Name) -join ', ')
    Start-Sleep -Seconds 5
}
