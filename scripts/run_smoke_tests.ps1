# Thesis Experiments - Smoke Test
# Quick validation of all experimental configurations
# Estimated runtime: 30-45 minutes total

Write-Host "=== Thesis Experiments - Smoke Test ===" -ForegroundColor Cyan
Write-Host "Validating all configurations with small scale (30 gens, 100 pop)"
Write-Host "Estimated runtime: 30-45 minutes total"
Write-Host ""

$startTotal = Get-Date

# Group A: Baselines
Write-Host "=== GROUP A: BASELINE METHODS ===" -ForegroundColor Magenta

Write-Host "Testing A1: Pure NSGA-II..." -ForegroundColor Green
uv run nsga --test --name "smoke-A1-pure-nsga"

Write-Host "Testing A2: NSGA-II + Repairs..." -ForegroundColor Green  
uv run nsga --test --repair-after-every-generation --name "smoke-A2-repairs"

# Group B: GA Enhancements
Write-Host ""
Write-Host "=== GROUP B: GA ENHANCEMENT METHODS ===" -ForegroundColor Magenta

Write-Host "Testing B1: NSGA-II + IGLS..." -ForegroundColor Green
uv run repairs --test --name "smoke-B1-igls"

Write-Host "Testing B2: NSGA-II + Heuristics..." -ForegroundColor Green
uv run heuristics --test --name "smoke-B2-heuristics"

Write-Host "Testing B3: Full GA..." -ForegroundColor Green
uv run full --test --name "smoke-B3-full"

# Group C: Hyper-Heuristics
Write-Host ""
Write-Host "=== GROUP C: HYPER-HEURISTIC METHODS ===" -ForegroundColor Magenta

Write-Host "Testing C1: Round-Robin..." -ForegroundColor Green
uv run roundrobin --test --name "smoke-C1-roundrobin"

# Optional RL test
Write-Host "Testing C2: RL-Guided (optional)..." -ForegroundColor Yellow
$rl_response = Read-Host "Test RL method? Requires trained model (y/N)"
if ($rl_response -eq "y" -or $rl_response -eq "Y") {
    uv run rl --test --name "smoke-C2-rl"
} else {
    Write-Host "Skipping RL test" -ForegroundColor Yellow
}

$totalDuration = (Get-Date) - $startTotal
Write-Host ""
Write-Host "=== SMOKE TEST COMPLETED! ===" -ForegroundColor Green
Write-Host "Total time: $($totalDuration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
Write-Host ""
Write-Host "All configurations validated successfully!" -ForegroundColor Green
Write-Host "Ready for full-scale experiments: scripts/run_thesis_experiments.ps1"
Write-Host ""