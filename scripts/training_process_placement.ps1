function Get-FullTrainingAffinityMask {
    try {
        $logicalProcessors = (
            @(Get-CimInstance Win32_Processor) |
                Measure-Object NumberOfLogicalProcessors -Sum
        ).Sum
    } catch {
        Write-Warning (
            "Could not determine the full CPU affinity mask; preserving " +
            "and recording the inherited mask. " +
            $_.Exception.Message
        )
        return [int64](
            (Get-Process -Id $PID).ProcessorAffinity
        )
    }
    if ($logicalProcessors -le 0) {
        throw "Windows reported no logical processors"
    }
    if ($logicalProcessors -gt 63) {
        throw (
            "Automatic full affinity supports at most 63 logical processors " +
            "in one processor group; pass an explicit group-aware launcher."
        )
    }
    if ($logicalProcessors -eq 63) {
        return [int64]::MaxValue
    }
    return ([int64]1 -shl [int]$logicalProcessors) - 1
}


function Get-PerformanceCoreMask {
    <#
    .SYNOPSIS
        Logical processors of the performance cores on a host whose hybrid
        layout is known, or $null when it is not known.
    #>
    param(
        [object[]] $Processors = $null
    )

    if ($null -eq $Processors) {
        try {
            $Processors = @(Get-CimInstance Win32_Processor)
        } catch {
            return $null
        }
    }
    if (
        $Processors.Count -eq 1 -and
        $Processors[0].Name -match "i7-12700K" -and
        [int]$Processors[0].NumberOfLogicalProcessors -eq 20
    ) {
        # Windows numbers the eight P-core hardware threads before the four
        # E-cores on the project's i7-12700K host.
        return [int64]0xFFFF
    }
    return $null
}


function Resolve-TrainingAffinityMask {
    param(
        [string] $ProcessorAffinityMask = "Auto"
    )

    if (
        [string]::IsNullOrWhiteSpace($ProcessorAffinityMask) -or
        $ProcessorAffinityMask -eq "None" -or
        $ProcessorAffinityMask -eq "0"
    ) {
        return Get-FullTrainingAffinityMask
    }
    if ($ProcessorAffinityMask -ne "Auto") {
        $value = if ($ProcessorAffinityMask.StartsWith("0x")) {
            [Convert]::ToInt64($ProcessorAffinityMask.Substring(2), 16)
        } else {
            [Convert]::ToInt64($ProcessorAffinityMask, 10)
        }
        if ($value -le 0) {
            throw "Processor affinity mask must be positive"
        }
        return $value
    }

    try {
        $processors = @(Get-CimInstance Win32_Processor)
    } catch {
        Write-Warning (
            "Could not inspect CPU topology; preserving and recording the " +
            "inherited affinity. " +
            $_.Exception.Message
        )
        return [int64](
            (Get-Process -Id $PID).ProcessorAffinity
        )
    }
    $performanceMask = Get-PerformanceCoreMask -Processors $processors
    if ($null -ne $performanceMask) {
        return $performanceMask
    }
    return Get-FullTrainingAffinityMask
}


function Set-TrainingProcessPlacement {
    param(
        [ValidateSet("Normal", "AboveNormal", "High")]
        [string] $ProcessPriority = "Normal",
        [string] $ProcessorAffinityMask = "Auto"
    )

    $process = Get-Process -Id $PID
    $process.PriorityClass = $ProcessPriority
    $resolvedMask = Resolve-TrainingAffinityMask $ProcessorAffinityMask
    if ($null -ne $resolvedMask) {
        $process.ProcessorAffinity = [IntPtr]$resolvedMask
    }
    return [PSCustomObject]@{
        Priority = [string]$process.PriorityClass
        AffinityMask = if ($null -eq $resolvedMask) {
            $null
        } else {
            "0x{0:X}" -f $resolvedMask
        }
    }
}
