# The distributable notebooks now live in subfolders, so this is a forwarder: it hands
# the notebook to the runner that sits next to it. Keeps `.\train_gpu_headless.ps1 <file name>`
# working from this folder, which is what the README documents.
#
# Usage:   .\train_gpu_headless.ps1 CNN_mixed.ipynb
#          .\train_gpu_headless.ps1 Transformer_six_single             (.ipynb is optional)
#          .\train_gpu_headless.ps1 CNN_mixed -Folder "Old Notebooks"
#
# The real script -- mount layout, image check, docker run -- lives in
# New Notebooks\train_gpu_headless.ps1. Edit it there, not here.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Notebook,

    [string] $Folder = "New Notebooks"
)

$ErrorActionPreference = "Stop"

$runner = Join-Path (Join-Path $PSScriptRoot $Folder) "train_gpu_headless.ps1"

if (-not (Test-Path $runner)) {
    Write-Host "Error: no runner script found at $runner" -ForegroundColor Red
    Write-Host "Copy 'New Notebooks\train_gpu_headless.ps1' into '$Folder' to run those notebooks."
    exit 1
}

& $runner $Notebook
exit $LASTEXITCODE
