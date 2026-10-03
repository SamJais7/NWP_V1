param(
    [string]$Message = "Update Project Pratyay progress $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    [switch]$Watch = $false
)

function Push-Changes {
    param([string]$CommitMsg)
    
    # Check if git remote is configured
    $remote = git remote
    if (-not $remote) {
        Write-Host "⚠️ No Git remote found. Please set your GitHub repository URL:" -ForegroundColor Yellow
        Write-Host "  git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git" -ForegroundColor Cyan
        Write-Host "  git branch -M main" -ForegroundColor Cyan
        Write-Host "  git push -u origin main" -ForegroundColor Cyan
        return
    }

    Write-Host "==> Checking git status..." -ForegroundColor Cyan
    git add .
    $status = git status --porcelain
    if ($status) {
        Write-Host "==> Committing changes: $CommitMsg" -ForegroundColor Green
        git commit -m "$CommitMsg"
        Write-Host "==> Pushing to origin main..." -ForegroundColor Green
        git push origin main
        Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
    } else {
        Write-Host "ℹ️ Working tree clean. Nothing new to commit." -ForegroundColor Gray
    }
}

if ($Watch) {
    Write-Host "🚀 Continuous Push Watcher started. Checking every 60 seconds... (Press Ctrl+C to stop)" -ForegroundColor Magenta
    while ($true) {
        Push-Changes "Auto-backup checkpoint $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Start-Sleep -Seconds 60
    }
} else {
    Push-Changes $Message
}
