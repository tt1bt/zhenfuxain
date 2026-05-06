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
    [double]$CbBeta = 0.999,
    [int]$NumWorkers = 0,
    [bool]$Amp = $true,
    [switch]$NoPretrained
)

$ErrorActionPreference = "Stop"

$imb = 0.05
$bits = 32
$weights = "model_none_CLRS_if0.05_bits32.pth"
$tag = "model_none_CLRS_if0.05_bits32"

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
    "--cls_weighting", "none",
    "--cb_beta", $CbBeta,
    "--cb_mode", "1",
    "--num_workers", $NumWorkers
)

if ($Amp) { $trainCmd += "--amp" }
if ($NoPretrained) { $trainCmd += "--no_pretrained" }

Write-Host "[Train] CLRS none (if=0.05, bits=32)" -ForegroundColor Cyan
Write-Host ("[CMD ] " + ($trainCmd -join " "))
& $trainCmd[0] $trainCmd[1..($trainCmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "Training failed. ExitCode=$LASTEXITCODE"
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
    "--out_tag", $tag,
    "--num_workers", $NumWorkers
)

Write-Host "[Test ] CLRS none (if=0.05, bits=32)" -ForegroundColor Green
Write-Host ("[CMD ] " + ($testCmd -join " "))
& $testCmd[0] $testCmd[1..($testCmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "Testing failed. ExitCode=$LASTEXITCODE"
}

Write-Host "Done: none case finished." -ForegroundColor Magenta
