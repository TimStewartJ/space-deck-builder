param([string] $Label = "ml", [int[]] $LaneCounts = @(1, 2, 3, 4), [int] $First = 205, [int] $Last = 226)
foreach ($n in $LaneCounts) {
    $walls = @(); $rolls = @(); $ppos = @(); $rates = 0.0
    for ($lane = 0; $lane -lt $n; $lane++) {
        $log = Get-ChildItem D:\sdb-perf-run\logs -Filter "sdb_$($Label)_${n}x_${lane}_*.log" | Sort-Object LastWriteTime | Select-Object -Last 1
        if (-not $log) { continue }
        $line = & D:\sdb-perf-scratch\readlog.ps1 -Path $log.FullName -Pattern 'Metrics file:' -Width 400 | Select-Object -First 1
        if ($line -notmatch 'Metrics file: (\S+)') { continue }
        $metrics = Join-Path D:\sdb-perf-run $Matches[1]
        $rows = Get-Content $metrics | ForEach-Object { $_ | ConvertFrom-Json }
        $prev = $null; $laneWalls = @()
        foreach ($r in $rows) {
            $ts = [datetime]$r.timestamp
            if ($prev -and $r.update -ge $First -and $r.update -le $Last) {
                $laneWalls += ($ts - $prev).TotalSeconds; $rolls += $r.rollout.duration_s; $ppos += $r.ppo_duration_s
            }
            $prev = $ts
        }
        if ($laneWalls.Count) { $mean = ($laneWalls | Measure-Object -Average).Average; $walls += $mean; $rates += 1.0 / $mean }
    }
    "{0} lane(s): per-lane update {1:N2}s (rollout {2:N2}s, ppo {3:N2}s) -> {4:N2} updates/s total, {5:N2}x one lane" -f $n, ($walls | Measure-Object -Average).Average, ($rolls | Measure-Object -Average).Average, ($ppos | Measure-Object -Average).Average, $rates, $rates
}
