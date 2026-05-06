param(
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
    [string]$DataRoot = "data/CLRS",
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

function Invoke-TrainThenTest {
    param(
        [string]$CaseName,
        [string]$ClsWeighting,
        [string]$CbMode,
        [string]$WeightPath,
        [string]$OutTag
    )


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
        "--split_path", "split_CLRS.json",
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
        "--split_path", "split_CLRS.json",
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
        WeightPath = "model_none_CLRS.pth"
        OutTag = "model_none_CLRS"
    },
    @{
        CaseName = "class_balanced_1-beta"
        ClsWeighting = "class_balanced"
        CbMode = "1-beta"
        WeightPath = "model_cb_CLRS.pth"
        OutTag = "model_cb_CLRS"
    },
    @{
        CaseName = "class_balanced_1"
        ClsWeighting = "class_balanced"
        CbMode = "1"
        WeightPath = "model_cb1_CLRS.pth"
        OutTag = "model_cb1_CLRS"
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
Write-Host "CLRS: all three loss setups finished train+test (IF=$ImbFactor, bits=$HashBits)." -ForegroundColor Magenta
