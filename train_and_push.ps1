# Run this script to train the model and push model files + any uncommitted changes to GitHub.
# Requires: Python with pip, and: pip install -r requirements.txt

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Step 1: Training the model (this may take several minutes)..." -ForegroundColor Cyan
python main.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed. Install deps with: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "best_model.pth") -or -not (Test-Path "class_names.json")) {
    Write-Host "Model files not found after training. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "Step 2: Adding model files and pushing to GitHub..." -ForegroundColor Cyan
git add -f best_model.pth class_names.json
git add -A
git status
git commit -m "Add trained model and latest project files"
git push

Write-Host "Done. Next: Deploy the API on Render (see DEPLOY.md)." -ForegroundColor Green
