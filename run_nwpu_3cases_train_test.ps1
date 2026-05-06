param(
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
    [string]$DataRoot = "data/NWPU-RESISC45",
    [string]$SplitPath = "split_NWPU-RESISC45.json",
    [double]$ImbFactor = 0.05,
    [int]$HashBits = 32,
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
    [switch]$NoPretrained,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path "train.py")) {
    throw "train.py not found in current directory."
}
if (-not (Test-Path "test.py")) {
    throw "test.py not found in current directory."
}
if (-not (Test-Path $DataRoot)) {
    throw "Dataset root not found: $DataRoot"
}

function Invoke-TrainThenTest {
    param(
        [string]$CaseName,
        [string]$ClsWeighting,
        [string]$CbMode,
        [string]$WeightPath,
        [string]$OutTag
    )

    if ((Test-Path $WeightPath) -and (-not $Force)) {
        throw "Weight already exists: $WeightPath. Use -Force to overwrite."
    }

    $trainCmd = @(
        $PythonExe,
        "train.py",
        "--root", $DataRoot,
        "--imb_factor", $ImbFactor,
        "--hash_bits", $HashBits,
        "--epochs", $Epochs,
        "--batch_size", $BatchSize,
        "--alpha", $Alpha,
        "--gamma", $Gamma,
        "--lr", $Lr,
        "--seed", $Seed,
        "--weights_out", $WeightPath,
        "--device", $Device,
        "--query_ratio", $QueryRatio,
        "--split_path", $SplitPath,
        "--cls_weighting", $ClsWeighting,
        "--cb_beta", $CbBeta,
        "--cb_mode", $CbMode,
        "--num_workers", $NumWorkers
    )

    if ($Amp) { $trainCmd += "--amp" }
    if ($NoPretrained) { $trainCmd += "--no_pretrained" }

    Write-Host ("=" * 90)
    Write-Host "[Train] $CaseName" -ForegroundColor Cyan
    Write-Host ("[CMD ] " + ($trainCmd -join " "))

    & $trainCmd[0] $trainCmd[1..($trainCmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $CaseName, ExitCode=$LASTEXITCODE"
    }

    $testCmd = @(
        $PythonExe,
        "test.py",
        "--root", $DataRoot,
        "--imb_factor", $ImbFactor,
        "--hash_bits", $HashBits,
        "--weights", $WeightPath,
        "--batch_size", $BatchSize,
        "--query_ratio", $QueryRatio,
        "--split_path", $SplitPath,
        "--device", $Device,
        "--topk", "0",
        "--out_tag", $OutTag,
        "--num_workers", $NumWorkers
    )

    Write-Host "[Test] $CaseName" -ForegroundColor Green
    Write-Host ("[CMD ] " + ($testCmd -join " "))

    & $testCmd[0] $testCmd[1..($testCmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Testing failed: $CaseName, ExitCode=$LASTEXITCODE"
    }
}

$cases = @(
    @{
        CaseName = "none"
        ClsWeighting = "none"
        CbMode = "1"
        WeightPath = "model_none_NWPU-RESISC45.pth"
        OutTag = "model_none_NWPU-RESISC45"
    },
    @{
        CaseName = "class_balanced_1-beta"
        ClsWeighting = "class_balanced"
        CbMode = "1-beta"
        WeightPath = "model_cb_NWPU-RESISC45.pth"
        OutTag = "model_cb_NWPU-RESISC45"
    },
    @{
        CaseName = "class_balanced_1"
        ClsWeighting = "class_balanced"
        CbMode = "1"
        WeightPath = "model_cb1_NWPU-RESISC45.pth"
        OutTag = "model_cb1_NWPU-RESISC45"
    }
)

foreach ($c in $cases) {
    Invoke-TrainThenTest `
        -CaseName $c.CaseName `
        -ClsWeighting $c.ClsWeighting `
        -CbMode $c.CbMode `
        -WeightPath $c.WeightPath `
        -OutTag $c.OutTag
}

Write-Host ("=" * 90)
Write-Host "NWPU-RESISC45: all three loss setups finished train+test (IF=$ImbFactor, bits=$HashBits, split=$SplitPath)." -ForegroundColor Magenta

