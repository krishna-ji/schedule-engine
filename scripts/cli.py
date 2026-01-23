"""
CLI Entry Points for Schedule Engine Scripts

Entry points for UV shortcuts defined in pyproject.toml.
Usage: uv run show-config, uv run tensorboard, etc.
"""

# ==================
# BENCHMARKING
# ==================


def benchmark_lns():
    """Benchmark Large Neighborhood Search with Constraint Programming."""
    from scripts.benchmarking.benchmark_lns_cp import main

    main()


# ==================
# TRAINING
# ==================


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


# ==================
# VALIDATION
# ==================


def check_data():
    """Check data quality and integrity."""
    from scripts.validation.check_data_quality import main

    main()


def verify_config():
    """Verify dataclass configs work correctly."""
    from scripts.validation.test_dataclass_configs import main

    main()


# ==================
# UTILITIES
# ==================


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


def run_model():
    """Run latest trained RL model."""
    from scripts.utilities.run_latest_model import main

    main()


def compare_heuristics():
    """Compare heuristic performance results."""
    from scripts.utilities.compare_heuristics import main

    main()


def visualize_training():
    """Generate RL training visualizations."""
    from scripts.utilities.visualize_rl_training import main

    main()


# ==================
# ENTRY POINT REGISTRY
# ==================

__all__ = [
    "benchmark_lns",
    "generate_validation",
    "select_checkpoint",
    "promote_model",
    "check_data",
    "verify_config",
    "show_config",
    "show_repair",
    "show_soft",
    "show_time",
    "tensorboard",
    "run_model",
    "compare_heuristics",
    "visualize_training",
]
