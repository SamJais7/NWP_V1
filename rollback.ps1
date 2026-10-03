param(
    [string]$Target = "HEAD~1",
    [switch]$Hard = $false,
    [switch]$List = $false
)

if ($List) {
    Write-Host "📜 Recent Git Commits:" -ForegroundColor Cyan
    git log --oneline -n 10
    return
}

Write-Host "⚠️ Rollback script initiated." -ForegroundColor Yellow
Write-Host "Current HEAD:" -ForegroundColor Gray
git log -1 --oneline

if ($Hard) {
    Write-Host "Resetting HARD to $Target (Discarding uncommitted changes)..." -ForegroundColor Red
    git reset --hard $Target
} else {
    Write-Host "Resetting SOFT to $Target (Keeping files staged for review)..." -ForegroundColor Yellow
    git reset --soft $Target
}

Write-Host "New HEAD:" -ForegroundColor Green
git log -1 --oneline
Write-Host "✅ Rollback completed!" -ForegroundColor Green
