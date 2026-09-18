# Launch JupyterLab in the GPU container and open the notebooks interactively.
# The whole "Analysis" folder is mounted at /workspace so that the notebook's
# relative paths (../data/mixed_ratio) work and any files it writes land back
# on disk. Run this from anywhere; it computes the Analysis path itself.

$ErrorActionPreference = "Stop"

# CNN folder = this script's folder; Analysis = its parent.
$cnnDir      = $PSScriptRoot
$analysisDir = Split-Path $cnnDir -Parent

Write-Host "Mounting:  $analysisDir  ->  /workspace"
Write-Host "Open http://localhost:8888/lab in your browser once it starts."
Write-Host ""

docker run --rm -it `
    --gpus all `
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -p 8888:8888 `
    -v "${analysisDir}:/workspace" `
    -w /workspace/CNN `
    raman-cnn-gpu `
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root `
        --ServerApp.token='' --ServerApp.password=''
