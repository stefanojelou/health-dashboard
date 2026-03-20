$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logFile = Join-Path $scriptDir "data\scheduled_pipeline.log"
$pipelineScript = Join-Path $scriptDir "update_dashboard_data.py"

function Write-Log([string]$message) {
    for ($i = 0; $i -lt 15; $i++) {
        try {
            Add-Content -Path $logFile -Value $message
            return
        }
        catch {
            Start-Sleep -Milliseconds 200
        }
    }
    throw "Could not write to log file: $logFile"
}

function Resolve-PythonExe {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    return $null
}

Set-Location $scriptDir

Write-Log ("==== [{0}] START ====" -f (Get-Date -Format "ddd MM/dd/yyyy HH:mm:ss.ff"))
Write-Log ("Script file: `"{0}`"" -f $pipelineScript)

if (-not (Test-Path -LiteralPath $pipelineScript)) {
    Write-Log "ERROR: Pipeline script not found."
    Write-Log ("==== [{0}] END (exit 2) ====" -f (Get-Date -Format "ddd MM/dd/yyyy HH:mm:ss.ff"))
    exit 2
}

$pythonExe = Resolve-PythonExe
if (-not $pythonExe) {
    Write-Log "ERROR: Python executable not found."
    Write-Log ("==== [{0}] END (exit 9009) ====" -f (Get-Date -Format "ddd MM/dd/yyyy HH:mm:ss.ff"))
    exit 9009
}

Write-Log ("Python executable: `"{0}`"" -f $pythonExe)

try {
    & $pythonExe --version 2>&1 | ForEach-Object { Write-Log $_ }

    & $pythonExe $pipelineScript --allow-billing-fallback 2>&1 | ForEach-Object { Write-Log $_ }
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) { $exitCode = 0 }
}
catch {
    Write-Log ("ERROR: {0}" -f $_.Exception.Message)
    $exitCode = 1
}

Write-Log ("==== [{0}] END (exit {1}) ====" -f (Get-Date -Format "ddd MM/dd/yyyy HH:mm:ss.ff"), $exitCode)
exit $exitCode
