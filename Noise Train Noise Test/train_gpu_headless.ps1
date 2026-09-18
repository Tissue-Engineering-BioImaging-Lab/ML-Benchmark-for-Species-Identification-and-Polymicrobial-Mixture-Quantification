# Train + run a Noise Train Noise Test notebook end-to-end on the GPU with no browser,
# saving all outputs (plots, metrics, and model_predictions.csv) back into this folder.
#
# Usage:   .\train_gpu_headless.ps1 CNN_all_noisy_combined.ipynb
#          .\train_gpu_headless.ps1 transformer_mixed_noisy_combined.ipynb
#
# The whole "Analysis" folder is mounted at /workspace, so the notebooks' relative
# paths (../data/mixed_ratio) work and everything they write -- including the shared
# model_predictions.csv used by cochrans_q_analysis.py -- lands back in this folder.
# Reuses the same `raman-cnn-gpu` image the CNN/Transformer folders use
# (built from CNN/Dockerfile.gpu); no separate image is needed here.

$ErrorActionPreference = "Stop"

$modelDir    = $PSScriptRoot
$analysisDir = Split-Path $modelDir -Parent
$folderName  = Split-Path $modelDir -Leaf
$image       = "raman-cnn-gpu"

# A notebook file name is required.
if ($args.Count -lt 1) {
    Write-Host "Error: a notebook file name is required." -ForegroundColor Red
    Write-Host "Usage: .\train_gpu_headless.ps1 <notebook.ipynb>"
    exit 1
}

$notebook = $args[0]

# Make sure the notebook actually exists in this folder before launching Docker.
if (-not (Test-Path (Join-Path $modelDir $notebook))) {
    Write-Host "Error: cannot find file '$notebook' in $modelDir" -ForegroundColor Red
    exit 1
}

# The GPU image is shared with the CNN/Transformer folders. Point the user at the
# one-time build if it hasn't been created yet, instead of a cryptic Docker pull error.
$existing = docker images -q $image
if (-not $existing) {
    Write-Host "Docker image '$image' not found locally." -ForegroundColor Yellow
    Write-Host "Build it once (the same image CNN/Transformer use) from the CNN folder:"
    Write-Host "    cd ..\CNN ; docker build -t $image -f Dockerfile.gpu . ; cd '..\$folderName'"
    exit 1
}

Write-Host "Executing $notebook on the GPU..."

docker run --rm `
    --gpus all `
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -v "${analysisDir}:/workspace" `
    -w "/workspace/$folderName" `
    $image `
    jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=-1 `
        "$notebook"

Write-Host "Done. Outputs saved inside $notebook (predictions in $folderName\model_predictions.csv)"
