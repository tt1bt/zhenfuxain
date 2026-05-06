param(
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
    [string]$DataRoot = "data",
    [string]$WeightsDir = ".",
    [string]$SplitDir = ".",
    [double]$ImbFactor = 0.01,
    [int]$HashBits = 32,
    [int]$Epochs = 150,
    [int]$BatchSize = 64,
    [int]$CenterBatchSize = 128,
    [double]$Alpha = 0.2,
    [double]$Gamma = 1.0,
    [double]$Lr = 1e-5,
    [int]$Seed = 42,
    [string]$Device = "cuda",
    [double]$QueryRatio = 0.2,
    [double]$CbBeta = 0.9999,
    [int]$NumWorkers = 0,
    [bool]$Amp = $true,
    [switch]$NoPretrained,
    [switch]$Force,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

# 类平衡训练（标准版）：w_i = (1-beta)/(1-beta^n_i)。
$cmd = @(
    $PythonExe,
    "train_all_datasets.py",
    "--data_root", $DataRoot,
    "--imb_factor", $ImbFactor,
    "--hash_bits", $HashBits,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--center_batch_size", $CenterBatchSize,
    "--alpha", $Alpha,
    "--gamma", $Gamma,
    "--lr", $Lr,
    "--seed", $Seed,
    "--device", $Device,
    "--query_ratio", $QueryRatio,
    "--cls_weighting", "class_balanced",
    "--cb_beta", $CbBeta,
    "--cb_mode", "1-beta",
    "--num_workers", $NumWorkers,
    "--weights_dir", $WeightsDir,
    "--weights_template", "model_cb_{dataset}.pth",
    "--split_dir", $SplitDir
)

if ($NoPretrained) { $cmd += "--no_pretrained" }
if ($Amp) { $cmd += "--amp" }
if ($Force) { $cmd += "--force" }
if ($ContinueOnError) { $cmd += "--continue_on_error" }

Write-Host "[类平衡训练-标准]" -ForegroundColor Cyan
Write-Host ("[CMD ] " + ($cmd -join " "))

& $cmd[0] $cmd[1..($cmd.Length - 1)]
exit $LASTEXITCODE
