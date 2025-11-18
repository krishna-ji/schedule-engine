"""
CLI Entry Points for Schedule Engine Scripts

This module provides clean entry points for all utility scripts,
enabling easy execution via UV shortcuts defined in pyproject.toml.

All entry points follow the pattern:
    def entry_name():
        from scripts.category.script_name import main
        main()

This allows UV shortcuts like:
    uv run diagnose-gpu
    uv run benchmark-gpu
    uv run show-config
"""

# ==============================================================================
# DIAGNOSTICS
# ==============================================================================


def diagnose_gpu():
    """Diagnose GPU/CUDA setup and configuration."""
    from scripts.diagnostics.diagnose_gpu import main

    main()


def test_dashboard():
    """Test TensorBoard dashboard integration."""
    from scripts.diagnostics.test_dashboard_integration import main

    main()


# ==============================================================================
# BENCHMARKING
# ==============================================================================


def benchmark_gpu():
    """Benchmark GPU vs CPU training performance."""
    from scripts.benchmarking.benchmark_gpu_training import main

    main()


def benchmark_lns():
    """Benchmark Large Neighborhood Search with Constraint Programming."""
    from scripts.benchmarking.benchmark_lns_cp import main

    main()


def benchmark_constraints():
    """Benchmark constraint checking performance."""
    from scripts.benchmarking.bench_constraint_check import main

    main()


# ==============================================================================
# TRAINING
# ==============================================================================


def generate_validation():
    """Generate validation dataset for RL training."""
    from scripts.training.generate_validation_set import main

    main()


def select_checkpoint():
    """Select best checkpoint from training run."""
    from scripts.training.select_best_checkpoint import main

    main()


def promote_model():
    """Promote validated model to production."""
    from scripts.training.promote_model_to_prod import main

    main()


# ==============================================================================
# VALIDATION
# ==============================================================================


def check_data():
    """Check data quality and integrity."""
    from scripts.validation.check_data_quality import main

    main()


def verify_config():
    """Verify configuration standardization."""
    from scripts.validation.verify_config_standardization import main

    main()


def verify_enhancements():
    """Verify Phase 3 advanced enhancements."""
    from scripts.validation.verify_enhancements import main

    main()


# ==============================================================================
# UTILITIES
# ==============================================================================


def show_config():
    """Display current configuration (all constraints)."""
    from scripts.utilities.show_config import main

    main()


def show_repair():
    """Display repair system configuration."""
    from scripts.utilities.show_repair_config import main

    main()


def show_soft():
    """Display soft constraints configuration."""
    from scripts.utilities.show_soft_config import main

    main()


def show_time():
    """Display time system configuration."""
    from scripts.utilities.show_time_config import main

    main()


def tensorboard():
    """Start TensorBoard for training logs."""
    from scripts.utilities.start_tensorboard import main

    main()


def git_squash():
    """Interactive git commit squashing tool."""
    from scripts.utilities.git_squash import main

    main()


def refactor_csv():
    """Refactor CSV export functionality."""
    from scripts.utilities.refactor_csv_exports import main

    main()


# ==============================================================================
# ENTRY POINT REGISTRY
# ==============================================================================

__all__ = [
    # Diagnostics
    "diagnose_gpu",
    "test_dashboard",
    # Benchmarking
    "benchmark_gpu",
    "benchmark_lns",
    "benchmark_constraints",
    # Training
    "generate_validation",
    "select_checkpoint",
    "promote_model",
    # Validation
    "check_data",
    "verify_config",
    "verify_enhancements",
    # Utilities
    "show_config",
    "show_repair",
    "show_soft",
    "show_time",
    "tensorboard",
    "git_squash",
    "refactor_csv",
]
