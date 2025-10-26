# Activate conda environment and run main.py
conda activate deap-env
if ($LASTEXITCODE -eq 0) {
python main.py --env test
} else {
    Write-Error "Failed to activate conda environment 'deap-env'"
}
