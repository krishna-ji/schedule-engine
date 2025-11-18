
# ============================================
# Complete Logging Solution for Remote Windows
# Captures ALL console output to log file
# ============================================

# 1. In your SSH session, run:
powershell -NoExit -Command "
    cd C:\Users\khem\.0\;
    `$logDir = 'C:\Users\khem\.0\logs';
    `$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss';
    `$logFile = Join-Path `$logDir \"prod_run_`$timestamp.log\";
    New-Item -ItemType Directory -Force -Path `$logDir | Out-Null;
    Start-Transcript -Path `$logFile -Append;
    Write-Host \"=== Production run started at `$(Get-Date) ===\" | Tee-Object -FilePath `$logFile -Append;
    Write-Host \"Log file: `$logFile\" | Tee-Object -FilePath `$logFile -Append;
    Write-Host \"========================================`n\" | Tee-Object -FilePath `$logFile -Append;
    uv run prod 2>&1 | Tee-Object -FilePath `$logFile -Append
"

# 2. Detach: Just close SSH or press Ctrl+C
# The process CONTINUES running in background with FULL logging!

# 3. Reconnect via SSH anytime and monitor logs:

# Option A: Watch the latest log file
Get-ChildItem C:\Users\khem\.0\logs\prod_run_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Wait

# Option B: Watch all recent logs
Get-Content C:\Users\khem\.0\logs\prod_run_*.log -Wait

# Option C: View existing application output (if any)
Get-Content C:\Users\khem\.0\output\latest.log -Wait

