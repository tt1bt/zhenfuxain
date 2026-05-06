param(
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
    [string]$LogDir = "logs_clrs_if0.05_bits32",
    [string]$OutCsv = "summary_clrs_if0.05_bits32.csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "[1/4] Run none" -ForegroundColor Cyan
& ".\run_clrs_if0.05_bits32_none.ps1" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir ("none_" + $ts + ".log"))
if ($LASTEXITCODE -ne 0) {
    throw "none case failed. ExitCode=$LASTEXITCODE"
}

Write-Host "[2/4] Run class_balanced 1-beta" -ForegroundColor Cyan
& ".\run_clrs_if0.05_bits32_cb_1beta.ps1" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir ("cb_1beta_" + $ts + ".log"))
if ($LASTEXITCODE -ne 0) {
    throw "class_balanced 1-beta case failed. ExitCode=$LASTEXITCODE"
}

Write-Host "[3/4] Run class_balanced 1" -ForegroundColor Cyan
& ".\run_clrs_if0.05_bits32_cb_1.ps1" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir ("cb_1_" + $ts + ".log"))
if ($LASTEXITCODE -ne 0) {
    throw "class_balanced 1 case failed. ExitCode=$LASTEXITCODE"
}

Write-Host "[4/4] Summarize metrics" -ForegroundColor Green
& $PythonExe "summarize_clrs_metrics.py" --log_dir $LogDir --out_csv $OutCsv
if ($LASTEXITCODE -ne 0) {
    throw "summarize_clrs_metrics.py failed. ExitCode=$LASTEXITCODE"
}

Write-Host "Done. Summary saved to $OutCsv" -ForegroundColor Magenta
