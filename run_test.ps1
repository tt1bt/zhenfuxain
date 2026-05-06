                                                                 param(
    [switch]$All,
    [switch]$DryRun,
    [string]$DataRoot = "data",
    [string]$WeightsDir = ".",
    [string]$PythonExe = "D:\app\Anaconda\envs\DL\python.exe",
    [double]$ImbFactor = 0.01,
    [int]$HashBits = 32,
    [bool]$InferSettingsFromWeight = $true,
    [int]$BatchSize = 64,
    [double]$QueryRatio = 0.2,
    [string]$Device = "auto",
    [int]$TopK = 0,
    [int]$NumWorkers = 0,
    [switch]$Tsne,
    [int]$TsneMax = 2000,
    [string[]]$Weights
)

$ErrorActionPreference = "Stop"

if ([System.IO.Path]::IsPathRooted($PythonExe)) {
    if (-not (Test-Path $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }
    $ResolvedPythonExe = $PythonExe
} else {
    $cmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Cannot resolve Python executable '$PythonExe'. Please pass -PythonExe with an absolute path."
    }
    $ResolvedPythonExe = $cmd.Source
}

Write-Host "[ENV ] Python: $ResolvedPythonExe"

function Get-DatasetFolders {
    param([string]$Root)

    if (-not (Test-Path $Root)) {
        throw "Data root not found: $Root"
    }

    return (Get-ChildItem -Path $Root -Directory | Select-Object -ExpandProperty Name)
}

function Resolve-DatasetFromWeight {
    param(
        [string]$WeightPath,
        [string[]]$DatasetNames
    )

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($WeightPath)

    foreach ($ds in $DatasetNames) {
        if ($stem.EndsWith("_" + $ds)) {
            return $ds
        }
    }

    foreach ($ds in $DatasetNames) {
        if ($stem.Contains($ds)) {
            return $ds
        }
    }

    return $null
}

function Resolve-ImbFromWeight {
    param([string]$WeightPath)

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($WeightPath)
    $m = [regex]::Match($stem, "_if([0-9]+(?:\.[0-9]+)?)")
    if ($m.Success) {
        return [double]$m.Groups[1].Value
    }
    return $null
}

function Resolve-BitsFromWeight {
    param([string]$WeightPath)

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($WeightPath)
    $m = [regex]::Match($stem, "bits([0-9]+)")
    if ($m.Success) {
        return [int]$m.Groups[1].Value
    }
    return $null
}

function Build-TestCommand {
    param(
        [string]$Python,
        [string]$DatasetRoot,
        [double]$Imb,
        [int]$Bits,
        [string]$Weight,
        [int]$Bz,
        [double]$QueryRatioArg,
        [string]$SplitPath,
        [int]$TopKArg,
        [string]$DeviceArg,
        [int]$Workers,
        [bool]$UseTsne,
        [int]$TsneMaxArg,
        [string]$OutTag
    )

    $cmd = @(
        $Python,
        "test.py",
        "--root", $DatasetRoot,
        "--imb_factor", $Imb,
        "--hash_bits", $Bits,
        "--weights", $Weight,
        "--batch_size", $Bz,
        "--query_ratio", $QueryRatioArg,
        "--split_path", $SplitPath,
        "--topk", $TopKArg,
        "--device", $DeviceArg,
        "--out_tag", $OutTag,
        "--num_workers", $Workers
    )

    if ($UseTsne) {
        $cmd += @("--tsne", "--tsne_max", $TsneMaxArg)
    }

    return ,$cmd
}

$datasetNames = Get-DatasetFolders -Root $DataRoot
if ($datasetNames.Count -eq 0) {
    throw "No dataset folder found under: $DataRoot"
}

if ($All -or -not $Weights -or $Weights.Count -eq 0) {
    $weightFiles = Get-ChildItem -Path $WeightsDir -Filter "model_*.pth" -File | Select-Object -ExpandProperty FullName
} else {
    $weightFiles = @()
    foreach ($w in $Weights) {
        if ([System.IO.Path]::IsPathRooted($w)) {
            $weightFiles += $w
        } else {
            $weightFiles += (Join-Path $WeightsDir $w)
        }
    }
}

if ($weightFiles.Count -eq 0) {
    Write-Host "No weight files found."
    exit 0
}

$ok = 0
$skip = 0
$fail = 0

foreach ($w in $weightFiles) {
    if (-not (Test-Path $w)) {
        Write-Host "[SKIP] Weight not found: $w"
        $skip += 1
        continue
    }

    $dataset = Resolve-DatasetFromWeight -WeightPath $w -DatasetNames $datasetNames
    if (-not $dataset) {
        Write-Host "[SKIP] Cannot infer dataset from weight: $w"
        $skip += 1
        continue
    }

    $datasetRoot = Join-Path $DataRoot $dataset
    if (-not (Test-Path $datasetRoot)) {
        Write-Host "[SKIP] Dataset path not found: $datasetRoot"
        $skip += 1
        continue
    }

    $splitPath = "split_${dataset}.json"
    $outTag = [System.IO.Path]::GetFileNameWithoutExtension($w)

    $evalImb = $ImbFactor
    $evalBits = $HashBits
    if ($InferSettingsFromWeight) {
        $imbFromWeight = Resolve-ImbFromWeight -WeightPath $w
        $bitsFromWeight = Resolve-BitsFromWeight -WeightPath $w

        if ($imbFromWeight -ne $null) {
            $evalImb = [double]$imbFromWeight
        }
        if ($bitsFromWeight -ne $null) {
            $evalBits = [int]$bitsFromWeight
        }
    }

    $cmd = Build-TestCommand `
        -Python $ResolvedPythonExe `
        -DatasetRoot $datasetRoot `
        -Imb $evalImb `
        -Bits $evalBits `
        -Weight $w `
        -Bz $BatchSize `
        -QueryRatioArg $QueryRatio `
        -SplitPath $splitPath `
        -TopKArg $TopK `
        -DeviceArg $Device `
        -Workers $NumWorkers `
        -UseTsne $Tsne.IsPresent `
        -TsneMaxArg $TsneMax `
        -OutTag $outTag

    Write-Host ("=" * 80)
    Write-Host "[RUN ] Dataset: $dataset"
    Write-Host "[RUN ] Weights: $w"
    Write-Host "[RUN ] Eval settings: imb_factor=$evalImb, hash_bits=$evalBits"
    Write-Host "[CMD ] $($cmd -join ' ')"

    if ($DryRun) {
        $ok += 1
        continue
    }

    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -eq 0) {
        $ok += 1
    } else {
        Write-Host "[FAIL] ExitCode=$LASTEXITCODE"
        $fail += 1
    }
}

Write-Host ("=" * 80)
Write-Host "Batch test summary"
Write-Host "Total:   $($weightFiles.Count)"
Write-Host "OK:      $ok"
Write-Host "Skipped: $skip"
Write-Host "Failed:  $fail"

if ($fail -gt 0) {
    exit 1
}
