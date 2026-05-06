param(
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
    [string]$DataRoot = "data/CLRS",
    [int]$Epochs = 150,
    [int]$BatchSize = 64,
    [double]$Alpha = 0.2,
    [double]$Gamma = 1.0,
    [double]$Lr = 1e-5,
    [int]$Seed = 42,
    [string]$Device = "cuda",
    [double]$QueryRatio = 0.2,
    [int]$NumWorkers = 0,
    [bool]$Amp = $true,
    [bool]$DisableTqdm = $false,
    [switch]$NoPretrained,
    [switch]$ForceRetrain,
    [double[]]$Betas = @(0.9, 0.99, 0.999, 0.9999),
    [string]$LogDir = "logs_clrs_if0.05_bits32_cb1_beta_sweep",
    [string]$OutCsv = "summary_clrs_if0.05_bits32_cb1_beta_sweep.csv"
)

$ErrorActionPreference = "Stop"

# Force UTF-8 output to reduce mojibake in redirected Python outputs.
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$oldEnv = @{
    PYTHONUTF8 = $env:PYTHONUTF8
    PYTHONIOENCODING = $env:PYTHONIOENCODING
    TQDM_DISABLE = $env:TQDM_DISABLE
    TQDM_ASCII = $env:TQDM_ASCII
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if ($DisableTqdm) {
    # Disable dynamic progress bars to avoid unreadable bar glyphs in merged logs.
    $env:TQDM_DISABLE = "1"
    $env:TQDM_ASCII = "1"
}

$imb = 0.05
$bits = 32

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Convert-BetaToTag {
    param([double]$Beta)
    $txt = "{0:0.####}" -f $Beta
    return ($txt -replace "\.", "p")
}

function Get-MetricsFromLog {
    param([string]$LogPath)

    $metrics = @{
        mAP = [double]::NaN
        P10 = [double]::NaN
        R10 = [double]::NaN
        P50 = [double]::NaN
        R50 = [double]::NaN
        P100 = [double]::NaN
        R100 = [double]::NaN
    }

    $lines = Get-Content -Path $LogPath -ErrorAction Stop
    foreach ($line in $lines) {
        $t = $line.Trim()

        $mMap = [regex]::Match($t, "^mAP:\s*([0-9]*\.?[0-9]+)")
        if ($mMap.Success) {
            $metrics.mAP = [double]$mMap.Groups[1].Value
            continue
        }

        $mPR = [regex]::Match($t, "^P@(10|50|100):\s*([0-9]*\.?[0-9]+)\s+R@\1:\s*([0-9]*\.?[0-9]+)")
        if ($mPR.Success) {
            $k = $mPR.Groups[1].Value
            $p = [double]$mPR.Groups[2].Value
            $r = [double]$mPR.Groups[3].Value

            if ($k -eq "10") {
                $metrics.P10 = $p
                $metrics.R10 = $r
            }
            elseif ($k -eq "50") {
                $metrics.P50 = $p
                $metrics.R50 = $r
            }
            elseif ($k -eq "100") {
                $metrics.P100 = $p
                $metrics.R100 = $r
            }
        }
    }

    return $metrics
}

$summaryRows = @()
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

try {
foreach ($beta in $Betas) {
    $betaTag = Convert-BetaToTag -Beta $beta
    $weights = "model_cb1_CLRS_if0.05_bits32_b$betaTag.pth"
    $outTag = "model_cb1_CLRS_if0.05_bits32_b$betaTag"
    $logPath = Join-Path $LogDir ("cb1_beta_" + $betaTag + "_" + $ts + ".log")

    if ((Test-Path $weights) -and (-not $ForceRetrain)) {
        Write-Host "[SKIP TRAIN] Weight exists: $weights" -ForegroundColor Yellow
    }
    else {
        $trainCmd = @(
            $PythonExe,
            "train.py",
            "--root", $DataRoot,
            "--imb_factor", $imb,
            "--hash_bits", $bits,
            "--epochs", $Epochs,
            "--batch_size", $BatchSize,
            "--alpha", $Alpha,
            "--gamma", $Gamma,
            "--lr", $Lr,
            "--seed", $Seed,
            "--weights_out", $weights,
            "--device", $Device,
            "--query_ratio", $QueryRatio,
            "--split_path", "split_CLRS.json",
            "--cls_weighting", "class_balanced",
            "--cb_beta", $beta,
            "--cb_mode", "1",
            "--num_workers", $NumWorkers
        )

        if ($Amp) { $trainCmd += "--amp" }
        if ($NoPretrained) { $trainCmd += "--no_pretrained" }

        Write-Host ("=" * 90)
        Write-Host "[Train] CLRS CB-1 (if=0.05, bits=32, beta=$beta)" -ForegroundColor Cyan
        Write-Host ("[CMD ] " + ($trainCmd -join " "))

        & $trainCmd[0] $trainCmd[1..($trainCmd.Length - 1)]
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for beta=$beta. ExitCode=$LASTEXITCODE"
        }
    }

    $testCmd = @(
        $PythonExe,
        "test.py",
        "--root", $DataRoot,
        "--imb_factor", $imb,
        "--hash_bits", $bits,
        "--weights", $weights,
        "--batch_size", $BatchSize,
        "--query_ratio", $QueryRatio,
        "--split_path", "split_CLRS.json",
        "--device", $Device,
        "--topk", "0",
        "--out_tag", $outTag,
        "--num_workers", $NumWorkers
    )

    Write-Host "[Test ] CLRS CB-1 (if=0.05, bits=32, beta=$beta)" -ForegroundColor Green
    Write-Host ("[CMD ] " + ($testCmd -join " "))

    $tmpStdOut = [System.IO.Path]::GetTempFileName()
    $tmpStdErr = [System.IO.Path]::GetTempFileName()

    try {
        $proc = Start-Process `
            -FilePath $testCmd[0] `
            -ArgumentList $testCmd[1..($testCmd.Length - 1)] `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $tmpStdOut `
            -RedirectStandardError $tmpStdErr

        $testExitCode = $proc.ExitCode

        $testOutput = @()
        if (Test-Path $tmpStdOut) {
            $testOutput += Get-Content -Path $tmpStdOut -Encoding UTF8 -ErrorAction SilentlyContinue
        }
        if (Test-Path $tmpStdErr) {
            $testOutput += Get-Content -Path $tmpStdErr -Encoding UTF8 -ErrorAction SilentlyContinue
        }

        $testOutput | Tee-Object -FilePath $logPath
    }
    finally {
        if (Test-Path $tmpStdOut) { Remove-Item $tmpStdOut -Force -ErrorAction SilentlyContinue }
        if (Test-Path $tmpStdErr) { Remove-Item $tmpStdErr -Force -ErrorAction SilentlyContinue }
    }

    if ($testExitCode -ne 0) {
        throw "Testing failed for beta=$beta. ExitCode=$testExitCode"
    }

    $m = Get-MetricsFromLog -LogPath $logPath
    $summaryRows += [PSCustomObject]@{
        beta = $beta
        weights = $weights
        out_tag = $outTag
        log_file = $logPath
        mAP = $m.mAP
        "P@10" = $m.P10
        "R@10" = $m.R10
        "P@50" = $m.P50
        "R@50" = $m.R50
        "P@100" = $m.P100
        "R@100" = $m.R100
    }
}

$summaryRows |
    Sort-Object beta |
    Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8

Write-Host ("=" * 90)
Write-Host "Done. Summary saved to $OutCsv" -ForegroundColor Magenta
}
finally {
    $env:PYTHONUTF8 = $oldEnv.PYTHONUTF8
    $env:PYTHONIOENCODING = $oldEnv.PYTHONIOENCODING
    $env:TQDM_DISABLE = $oldEnv.TQDM_DISABLE
    $env:TQDM_ASCII = $oldEnv.TQDM_ASCII
}