# Squash last N commits into one

# Get current branch name
$currentBranch = git rev-parse --abbrev-ref HEAD

Write-Host "Current branch: $currentBranch" -ForegroundColor Cyan
Write-Host ""

# Prompt for number of commits to squash
$numCommits = Read-Host "How many commits to squash?"

if (-not ($numCommits -match '^\d+$') -or [int]$numCommits -lt 2) {
    Write-Host "Invalid number. Must be 2 or greater." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "This will squash the last $numCommits commits into one." -ForegroundColor Yellow
Write-Host ""

# Show commits that will be squashed
Write-Host "Commits to be squashed:" -ForegroundColor Green
git log --oneline -$numCommits

Write-Host ""
$confirm = Read-Host "Continue? (y/n)"

if ($confirm -ne 'y') {
    Write-Host "Aborted." -ForegroundColor Red
    exit 1
}

# Prompt for new commit message
Write-Host ""
$commitMessage = Read-Host "Enter new commit message"

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    Write-Host "Empty commit message. Aborted." -ForegroundColor Red
    exit 1
}

try {
    # Soft reset to N commits back
    git reset --soft "HEAD~$numCommits"

    # Create new commit with all changes
    git commit -m $commitMessage

    Write-Host ""
    Write-Host "Successfully squashed $numCommits commits!" -ForegroundColor Green
    Write-Host "New commit:" -ForegroundColor Cyan
    git log --oneline -1

    Write-Host ""
    Write-Host "To push (if already pushed before):" -ForegroundColor Yellow
    Write-Host "  git push --force-with-lease" -ForegroundColor White
}
catch {
    Write-Host "Error occurred: $_" -ForegroundColor Red
    Write-Host "Rolling back..." -ForegroundColor Yellow
    git reset --hard ORIG_HEAD
    exit 1
}
