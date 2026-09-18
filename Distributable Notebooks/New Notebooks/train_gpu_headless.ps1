# Train + run a CNN or Transformer notebook end-to-end on the GPU with no browser.
# The executed notebook (plots, metrics, training logs) is written back in place and
# the trained model is saved under Distributable Notebooks\saved_models\.
#
# Usage:   .\train_gpu_headless.ps1 CNN_mixed.ipynb
#          .\train_gpu_headless.ps1 Transformer_six_single        (.ipynb is optional)
#
# The "Distributable Notebooks" folder is mounted at /workspace, so the relative
# paths in these notebooks resolve in the container exactly as they do on Windows:
#     ..\data\mixed_ratio  ->  /workspace/data/mixed_ratio
#     ..\saved_models      ->  /workspace/saved_models
# Everything the notebook writes therefore lands back in this folder tree on the host.
#
# Reuses the same `raman-cnn-gpu` image the CNN\ and Transformer\ folders use
# (built from CNN\Dockerfile.gpu); no separate image is needed here.

$ErrorActionPreference = "Stop"

$notebookDir = $PSScriptRoot                    # ...\Distributable Notebooks\New Notebooks
$mountDir    = Split-Path $notebookDir -Parent  # ...\Distributable Notebooks
$folderName  = Split-Path $notebookDir -Leaf    # "New Notebooks"
$image       = "raman-cnn-gpu"

# A notebook file name is required.
if ($args.Count -lt 1) {
    Write-Host "Error: a notebook file name is required." -ForegroundColor Red
    Write-Host "Usage: .\train_gpu_headless.ps1 <notebook.ipynb>"
    Write-Host "       e.g. .\train_gpu_headless.ps1 CNN_mixed.ipynb"
    exit 1
}

# Accept "CNN_mixed", "CNN_mixed.ipynb", or the ".\CNN_mixed.ipynb" that tab-completion produces.
$notebook = Split-Path $args[0] -Leaf
if (-not $notebook.EndsWith(".ipynb")) {
    $notebook = "$notebook.ipynb"
}

# Make sure the notebook actually exists in this folder before launching Docker.
if (-not (Test-Path (Join-Path $notebookDir $notebook))) {
    Write-Host "Error: cannot find file '$notebook' in $notebookDir" -ForegroundColor Red
    exit 1
}

# The image is built around TensorFlow and has no xgboost, so the XG notebooks cannot
# run in it. The RF notebooks do run, but they are CPU-only and gain nothing from the GPU.
if ($notebook -like "XG_*") {
    Write-Host "Error: '$notebook' needs xgboost, which is not installed in the '$image' image." -ForegroundColor Red
    Write-Host "Run the XG notebooks in your local Python environment instead (they are CPU-only)."
    exit 1
}
if ($notebook -like "RF_*") {
    Write-Host "Note: '$notebook' is CPU-only, so the GPU will sit idle. Continuing anyway." -ForegroundColor Yellow
}

# Check the daemon before the image lookup: when Docker Desktop is stopped, `docker images`
# returns nothing and would otherwise be misreported below as "image not found".
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$null = docker info --format "{{.ServerVersion}}" 2>&1
$dockerUp = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

if (-not $dockerUp) {
    Write-Host "Error: cannot reach the Docker daemon." -ForegroundColor Red
    Write-Host "Start Docker Desktop, wait for it to report 'Engine running', then retry."
    exit 1
}

# The GPU image is shared with the CNN/Transformer folders. Point the user at the
# one-time build if it hasn't been created yet, instead of a cryptic Docker pull error.
$existing = docker images -q $image
if (-not $existing) {
    Write-Host "Docker image '$image' not found locally." -ForegroundColor Yellow
    Write-Host "Build it once (the same image CNN/Transformer use) from the CNN folder:"
    Write-Host "    cd ..\..\CNN"
    Write-Host "    docker build -t $image -f Dockerfile.gpu ."
    Write-Host "    cd `"..\Distributable Notebooks\$folderName`""
    exit 1
}

Write-Host "Executing $notebook on the GPU..."

docker run --rm `
    --gpus all `
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -v "${mountDir}:/workspace" `
    -w "/workspace/$folderName" `
    $image `
    jupyter nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=-1 `
        "$notebook"

# nbconvert stops at the first failing cell; surface that instead of printing "Done".
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: nbconvert exited with code $LASTEXITCODE. '$notebook' may be partially executed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Done. Ran $notebook in place; trained model saved under Distributable Notebooks\saved_models\"
