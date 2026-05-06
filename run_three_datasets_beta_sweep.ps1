param(
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
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
    [switch]$NoPretrained,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# 验证环境
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path "train.py")) {
    throw "train.py not found in current directory."
}
if (-not (Test-Path "test.py")) {
    throw "test.py not found in current directory."
}

# 常量定义
$imb = 0.05
$bits = 32
$betas = @(0.9, 0.99, 0.999, 0.9999)

# 定义数据集配置
$datasets = @(
    @{
        Name = "NWPU"
        DataRoot = "data/NWPU-RESISC45"
        SplitPath = "split_NWPU-RESISC45.json"
    },
    @{
        Name = "PatternNet"
        DataRoot = "data/PatternNet"
        SplitPath = "split_patternnet.json"
    },
    @{
        Name = "RSSCN7"
        DataRoot = "data/RSSCN7"
        SplitPath = "split_RSSCN7.json"
    }
)

# 验证数据集存在
foreach ($ds in $datasets) {
    if (-not (Test-Path $ds.DataRoot)) {
        throw "Dataset root not found: $($ds.DataRoot)"
    }
}

# 生成 beta 字符串（0.9 -> b0p9）
function Format-BetaString {
    param([double]$Beta)
    return "b" + $Beta.ToString().Replace(".", "p")
}

# 执行训练和测试
function Invoke-TrainThenTest {
    param(
        [string]$DatasetName,
        [string]$DataRoot,
        [string]$SplitPath,
        [double]$Beta
    )

    $betaStr = Format-BetaString $Beta
    $weights = "model_cb1_${DatasetName}_if${imb}_bits${bits}_${betaStr}.pth"
    $tag = "model_cb1_${DatasetName}_if${imb}_bits${bits}_${betaStr}"

    # 检查模型是否已存在
    if ((Test-Path $weights) -and (-not $Force)) {
        Write-Host "[SKIP] Model already exists: $weights" -ForegroundColor Yellow
        return
    }

    # 构建训练命令
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
        "--split_path", $SplitPath,
        "--cls_weighting", "class_balanced",
        "--cb_beta", $Beta,
        "--cb_mode", "1",
        "--num_workers", $NumWorkers
    )

    if ($Amp) { $trainCmd += "--amp" }
    if ($NoPretrained) { $trainCmd += "--no_pretrained" }

    # 执行训练
    Write-Host ("=" * 90) -ForegroundColor Cyan
    Write-Host "[Train] $DatasetName (if=$imb, bits=$bits, beta=$Beta)" -ForegroundColor Cyan
    Write-Host ("[CMD ] " + ($trainCmd -join " "))
    Write-Host ""

    & $trainCmd[0] $trainCmd[1..($trainCmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $DatasetName (beta=$Beta), ExitCode=$LASTEXITCODE"
    }

    # 构建测试命令
    $testCmd = @(
        $PythonExe,
        "test.py",
        "--root", $DataRoot,
        "--imb_factor", $imb,
        "--hash_bits", $bits,
        "--weights", $weights,
        "--batch_size", $BatchSize,
        "--query_ratio", $QueryRatio,
        "--split_path", $SplitPath,
        "--device", $Device,
        "--topk", "0",
        "--out_tag", $tag,
        "--num_workers", $NumWorkers
    )

    # 执行测试
    Write-Host "[Test ] $DatasetName (if=$imb, bits=$bits, beta=$Beta)" -ForegroundColor Green
    Write-Host ("[CMD ] " + ($testCmd -join " "))
    Write-Host ""

    & $testCmd[0] $testCmd[1..($testCmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Testing failed for $DatasetName (beta=$Beta), ExitCode=$LASTEXITCODE"
    }

    Write-Host "[Done] $DatasetName (beta=$Beta) finished." -ForegroundColor Magenta
    Write-Host ""
}

# 主循环
$totalCases = $datasets.Count * $betas.Count
$currentCase = 0

foreach ($ds in $datasets) {
    foreach ($beta in $betas) {
        $currentCase++
        $percentage = [math]::Round(($currentCase / $totalCases) * 100, 0)
        $barLength = 30
        $filledLength = [math]::Round(($currentCase / $totalCases) * $barLength)
        $bar = ("#" * $filledLength) + ("-" * ($barLength - $filledLength))
        
        Write-Host ""
        Write-Host "[$bar] $percentage% ($currentCase/$totalCases) - $($ds.Name) β=$beta" -ForegroundColor Cyan
        Write-Host ""
        
        try {
            Invoke-TrainThenTest `
                -DatasetName $ds.Name `
                -DataRoot $ds.DataRoot `
                -SplitPath $ds.SplitPath `
                -Beta $beta
        }
        catch {
            Write-Host "ERROR: $_" -ForegroundColor Red
            throw
        }
    }
}

Write-Host ("=" * 90) -ForegroundColor Magenta
Write-Host "All done! Trained and tested:" -ForegroundColor Magenta
foreach ($ds in $datasets) {
    Write-Host "  - $($ds.Name): $($betas.Count) models (beta values: $($betas -join ', '))" -ForegroundColor Magenta
}
Write-Host "Total: $totalCases cases finished (if=$imb, bits=$bits)" -ForegroundColor Magenta
