function Enter-TrainingTaskLock {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,
        [int] $TimeoutSeconds = 30
    )

    $lockRoot = Join-Path `
        ([Environment]::GetFolderPath("LocalApplicationData")) `
        "SpaceDeckBuilder\locks"
    New-Item -ItemType Directory -Force -Path $lockRoot | Out-Null
    $lockPath = Join-Path $lockRoot "$Name.lock"
    $deadline = [datetime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            return [IO.File]::Open(
                $lockPath,
                [IO.FileMode]::OpenOrCreate,
                [IO.FileAccess]::ReadWrite,
                [IO.FileShare]::None
            )
        } catch [IO.IOException] {
            if ([datetime]::UtcNow -ge $deadline) {
                throw (
                    "Timed out waiting for the launch/stop lock for '$Name'."
                )
            }
            Start-Sleep -Milliseconds 100
        }
    } while ($true)
}
