$ErrorActionPreference = "Stop"
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$root = "E:\attention-pooling"
$py = "$root\.venv\Scripts\python.exe"
$logs = "$root\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$common = "-u -m src train --episodes 16000 --num-workers 16 --self-play --updates 30 --eval-every 10 --eval-games 200"

$sumLog  = "$logs\cmp_sum_$ts.log"
$attnLog = "$logs\cmp_attn_$ts.log"

"=== SUM run start $(Get-Date -Format o) ===" | Out-File $sumLog
& $py $common.Split(" ") --pool-type sum *>> $sumLog
"=== SUM run done $(Get-Date -Format o) ===" | Out-File -Append $sumLog

"=== ATTN run start $(Get-Date -Format o) ===" | Out-File $attnLog
& $py $common.Split(" ") --pool-type attention *>> $attnLog
"=== ATTN run done $(Get-Date -Format o) ===" | Out-File -Append $attnLog
