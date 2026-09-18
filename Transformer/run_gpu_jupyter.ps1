# Launch JupyterLab in the GPU container and open the notebooks interactively.
# The whole "Analysis" folder is mounted at /workspace so the notebook's relative
# paths (../data/single, ../data/mixed_ratio) work and files it writes land back
# on disk. Works from whichever model folder this script lives in.

$ErrorActionPreference = "Stop"

$modelDir    = $PSScriptRoot                  # ...\Analysis\Transformer
$analysisDir = Split-Path $modelDir -Parent   # ...\Analysis
$folderName  = Split-Path $modelDir -Leaf     # "Transformer"

Write-Host "Mounting:  $analysisDir  ->  /workspace"
Write-Host "Container working dir: /workspace/$folderName"
Write-Host "Open http://localhost:8888/lab in your browser once it starts."
Write-Host ""

docker run --rm -it `
    --gpus all `
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    -p 8888:8888 `
    -v "${analysisDir}:/workspace" `
    -w "/workspace/$folderName" `
    raman-cnn-gpu `
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root `
        --ServerApp.token='' --ServerApp.password=''
