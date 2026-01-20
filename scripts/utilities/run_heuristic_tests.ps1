# Heuristic Testing Suite
# Sequential execution of 12 core heuristic tests
# Config: ngen=1000, pop_size=20

$ErrorActionPreference = "Stop"

Write-Host "`n=== HEURISTIC TESTING SUITE ===" -ForegroundColor Cyan
Write-Host "Testing 12 core heuristics individually" -ForegroundColor Cyan
Write-Host "Config: ngen=1000, pop_size=20" -ForegroundColor Cyan
Write-Host "Duration: ~1-2 hours total`n" -ForegroundColor Yellow

# Test counter
$testNum = 0
$totalTests = 12
$startTime = Get-Date

# ========================================
# TIER 1: CRITICAL BASELINE (6)
# ========================================

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: largest_degree_first (Construction)" -ForegroundColor Green
uv run heuristic-testing --test --name test-largest-degree-first --override heuristics_single_override=largest_degree_first ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: random_swap (Perturbation)" -ForegroundColor Green
uv run heuristic-testing --test --name test-random-swap --override heuristics_single_override=random_swap ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: kempe_chain (Improvement)" -ForegroundColor Green
uv run heuristic-testing --test --name test-kempe-chain --override heuristics_single_override=kempe_chain ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: distance_preserving_crossover (Diversity)" -ForegroundColor Green
uv run heuristic-testing --test --name test-distance-preserving-crossover --override heuristics_single_override=distance_preserving_crossover ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: exhaustive_repair (Repair - Brute Force)" -ForegroundColor Green
uv run heuristic-testing --test --name test-exhaustive-repair --override heuristics_single_override=exhaustive_repair ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: greedy_repair (Repair - Fast)" -ForegroundColor Green
uv run heuristic-testing --test --name test-greedy-repair --override heuristics_single_override=greedy_repair ngen=1000 pop_size=20

# ========================================
# TIER 2: IMPORTANT VARIANTS (6)
# ========================================

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: most_constrained_first (Construction)" -ForegroundColor Green
uv run heuristic-testing --test --name test-most-constrained-first --override heuristics_single_override=most_constrained_first ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: temporal_shift (Perturbation)" -ForegroundColor Green
uv run heuristic-testing --test --name test-temporal-shift --override heuristics_single_override=temporal_shift ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: ejection_chain (Improvement)" -ForegroundColor Green
uv run heuristic-testing --test --name test-ejection-chain --override heuristics_single_override=ejection_chain ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: crowding_mutation (Diversity)" -ForegroundColor Green
uv run heuristic-testing --test --name test-crowding-mutation --override heuristics_single_override=crowding_mutation ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: niching_selection (Diversity)" -ForegroundColor Green
uv run heuristic-testing --test --name test-niching-selection --override heuristics_single_override=niching_selection ngen=1000 pop_size=20

$testNum++
Write-Host "`n[$testNum/$totalTests] Testing: selective_repair (Repair - Smart)" -ForegroundColor Green
uv run heuristic-testing --test --name test-selective-repair --override heuristics_single_override=selective_repair ngen=1000 pop_size=20

# ========================================
# COMPLETION SUMMARY
# ========================================

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n=== ALL TESTS COMPLETE ===" -ForegroundColor Cyan
Write-Host "Total time: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "Results saved in: output/f-heuristic-testing/" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Run comparison script: uv run python scripts/utilities/compare_heuristics.py" -ForegroundColor White
Write-Host "  2. Review plots in output/f-heuristic-testing/" -ForegroundColor White
Write-Host "  3. Analyze heuristic_comparison.csv`n" -ForegroundColor White
