# Train + run a notebook end-to-end on the GPU with no browser, saving all
# outputs (plots, metrics) back into the .ipynb in place.
#
# Usage:   .\train_gpu_headless.ps1 CNN_mixed_noisy_separate.ipynb
#          .\train_gpu_headless.ps1 CNN_all_noisy_separate.ipynb

$ErrorActionPreference = "Stop"

$cnnDir      = $PSScriptRoot
$analysisDir = Split-Path $cnnDir -Parent

# A notebook file name is required.
if ($args.Count -lt 1) {
    Write-Host "Error: a notebook file name is required." -ForegroundColor Red
    Write-Host "Usage: .\train_gpu_headless.ps1 <notebook.ipynb>"
    exit 1
}

$notebook = $args[0]

# Make sure the notebook actually exists in the CNN folder before launching Docker.
if (-not (Test-Path (Join-Path $cnnDir $notebook))) {
    Write-Host "Error: cannot find file '$notebook' in $cnnDir" -ForegroundColor Red
    exit 1
}

Write-Host "Executing $notebook on the GPU..."

docker run --rm `
    --gpus all `
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -v "${analysisDir}:/workspace" `
    -w /workspace/CNN `
    raman-cnn-gpu `
    jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=-1 `
        "$notebook"

Write-Host "Done. Outputs saved inside $notebook"
