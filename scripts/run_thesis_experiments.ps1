# Thesis Experiments - Batch Runner
# Executes all experimental groups in sequence
# Estimated total runtime: 40-50 hours

Write-Host "=== Thesis Experiments - Full Scale ===" -ForegroundColor Cyan
Write-Host "Estimated total runtime: 40-50 hours" -ForegroundColor Yellow
Write-Host ""

# Check system readiness
Write-Host "Checking system readiness..." -ForegroundColor Green
uv run diagnose

# Confirm execution
$response = Read-Host "Continue with full experimental run? (y/N)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "Experiment cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "=== GROUP A: BASELINE METHODS ===" -ForegroundColor Magenta
Write-Host "Purpose: Establish baseline performance without domain knowledge"
Write-Host ""

# A1: Pure NSGA-II (3-4 hours)
Write-Host "Running A1: Pure NSGA-II Baseline..." -ForegroundColor Green
$start = Get-Date
uv run nsga --prod --name "A1-pure-nsga-baseline"
$duration = (Get-Date) - $start
Write-Host "A1 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
Write-Host ""

# A2: NSGA-II + Repairs (4-5 hours)
Write-Host "Running A2: NSGA-II with Repairs..." -ForegroundColor Green
$start = Get-Date
uv run nsga --prod --repair-after-every-generation --name "A2-nsga-with-repairs"
$duration = (Get-Date) - $start
Write-Host "A2 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
Write-Host ""

Write-Host "=== GROUP B: GA ENHANCEMENT METHODS ===" -ForegroundColor Magenta
Write-Host "Purpose: Measure incremental improvements from domain knowledge"
Write-Host ""

# B1: NSGA-II + IGLS (5-6 hours)
Write-Host "Running B1: NSGA-II + IGLS Repairs..." -ForegroundColor Green
$start = Get-Date
uv run repairs --prod --name "B1-nsga-igls"
$duration = (Get-Date) - $start
Write-Host "B1 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
Write-Host ""

# B2: NSGA-II + Heuristics (6-7 hours)
Write-Host "Running B2: NSGA-II + Heuristics..." -ForegroundColor Green
$start = Get-Date
uv run heuristics --prod --name "B2-nsga-heuristics"
$duration = (Get-Date) - $start
Write-Host "B2 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
Write-Host ""

# B3: Full GA (8-10 hours)
Write-Host "Running B3: Full GA (Best Non-RL)..." -ForegroundColor Green
$start = Get-Date
uv run full --prod --name "B3-full-ga"
$duration = (Get-Date) - $start
Write-Host "B3 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
Write-Host ""

Write-Host "=== GROUP C: HYPER-HEURISTIC METHODS ===" -ForegroundColor Magenta
Write-Host "Purpose: Compare different heuristic selection strategies"
Write-Host ""

# C1: Round-Robin (7-8 hours)
Write-Host "Running C1: Round-Robin Selection..." -ForegroundColor Green
$start = Get-Date
uv run roundrobin --prod --name "C1-roundrobin"
$duration = (Get-Date) - $start
Write-Host "C1 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
Write-Host ""

# C2: RL-Guided (9-12 hours) - Optional
Write-Host "RL-Guided experiment (C2) requires trained model." -ForegroundColor Yellow
$rl_response = Read-Host "Run RL experiment? (y/N)"
if ($rl_response -eq "y" -or $rl_response -eq "Y") {
    Write-Host "Running C2: RL-Guided Selection..." -ForegroundColor Green
    $start = Get-Date
    uv run rl --prod --name "C2-rl-guided"
    $duration = (Get-Date) - $start
    Write-Host "C2 completed in: $($duration.TotalHours.ToString('F1')) hours" -ForegroundColor Green
} else {
    Write-Host "Skipping C2 (RL-Guided)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== ALL EXPERIMENTS COMPLETED! ===" -ForegroundColor Green
Write-Host "Check results in output/ directory" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run analysis: python scripts/analysis/compare_experiments.py"
Write-Host "  2. Generate plots: python scripts/analysis/generate_thesis_plots.py"
Write-Host "  3. Statistical tests: python scripts/analysis/statistical_analysis.py"
Write-Host ""