# Start TensorBoard for RL Training
# Usage: .\start_tensorboard.ps1

Write-Host "Starting TensorBoard..." -ForegroundColor Green
Write-Host "Log directory: logs/tensorboard/train" -ForegroundColor Cyan
Write-Host ""
Write-Host "TensorBoard will be available at:" -ForegroundColor Yellow
Write-Host "  http://localhost:6006/" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop TensorBoard" -ForegroundColor Gray
Write-Host ""

uv run tensorboard --logdir logs/tensorboard/train --port 6006 --bind_all
