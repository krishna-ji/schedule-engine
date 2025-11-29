"""Training entry point script for RL hyper-heuristic agents."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.rl.gym_env import ScheduleEnv
from src.rl.training import RLTrainer
from src.rl.training.config_loader import (
    DEFAULT_PROFILE,
    list_training_profiles,
    load_training_config,
)
from src.utils.structured_logger import StructuredLogger, setup_logging
from src.utils.system_info import get_cpu_count
from src.workflows.standard_run import load_input_data

logger = StructuredLogger.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    profiles = sorted({*list_training_profiles(), DEFAULT_PROFILE})

    parser = argparse.ArgumentParser(
        description="Train RL agent for heuristic selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Number of training timesteps",
    )

    parser.add_argument(
        "--agent",
        "--agent-type",
        dest="agent_type",
        type=str,
        default=None,
        choices=["ppo", "dqn"],
        help="RL agent type",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Model save path (default: auto-generated)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save models",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory with JSON files",
    )

    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Max generations per episode",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="GA population size",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes after training",
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Verbosity level (0=none, 1=info, 2=debug)",
    )

    parser.add_argument(
        "--debug-logging",
        dest="debug_logging",
        action="store_true",
        default=None,
        help="Enable verbose environment/debug logging",
    )

    parser.add_argument(
        "--debug-log-interval",
        type=int,
        default=None,
        metavar="N",
        help="Minimum step interval between debug log entries when enabled",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment and agent",
    )

    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help="Enable curriculum learning (progressive difficulty stages)",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        choices=profiles,
        help="Training profile defined in configs/training/",
    )

    parser.add_argument(
        "--config",
        "--config-path",
        dest="config",
        type=str,
        default=None,
        help="Optional custom training config to merge (YAML)",
    )

    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available training profiles and exit",
    )

    # Optional positional profile to allow shorthand invocation: `uv run train prod`
    parser.add_argument(
        "profile_positional",
        nargs="?",
        choices=profiles,
        help="Optional positional profile (same values as --profile).",
    )

    args = parser.parse_args()
    # Environment variable fallback: RL_DEFAULT_PROFILE to set default profile
    env_profile = None
    try:
        import os

        env_profile = os.environ.get("RL_DEFAULT_PROFILE")
    except Exception:
        env_profile = None
    if env_profile and not getattr(args, "profile_positional", None):
        # Use env var only if user did not provide positional override
        args.profile = env_profile
    # If user provided a positional profile, prefer it over --profile default
    if getattr(args, "profile_positional", None):
        args.profile = args.profile_positional
    return args


def apply_profile_defaults(args: argparse.Namespace, profile: dict[str, Any]) -> None:
    """Fill argparse values using profile defaults when not provided."""

    def pick(field: str, default: Any | None = None):
        value = getattr(args, field, None)
        if value is not None:
            return value
        return profile.get(field, default)

    args.agent_type = pick("agent_type", "ppo")
    args.timesteps = pick("timesteps")
    args.max_generations = pick("max_generations")
    args.max_steps = pick("max_steps")
    args.population_size = pick("population_size")
    args.eval_episodes = pick("eval_episodes", 0)
    args.data_dir = pick("data_dir", "data")
    args.save_dir = pick("save_dir", profile.get("save_dir", "models/rl_agents"))
    args.tensorboard_log = pick(
        "tensorboard_log",
        profile.get("tensorboard_log", "logs/tensorboard/train"),
    )
    args.verbose = pick("verbose", 1)
    args.save_prefix = profile.get("save_prefix", "rl_agent")
    args.seed = pick("seed")
    args.debug_logging = pick("debug_logging", False)
    args.debug_log_interval = pick("debug_log_interval", 25)
    args.curriculum = pick("curriculum", False)
    if args.debug_log_interval is None:
        args.debug_log_interval = 25
    args.debug_log_interval = max(1, int(args.debug_log_interval))
    args.no_eval = getattr(args, "no_eval", False) or not profile.get(
        "enable_eval", True
    )
    args.loaded_profile = profile.get("profile", args.profile)

    # Parallel training settings (NEW)
    parallel_config = profile.get("parallel", {})
    args.n_envs = pick("n_envs", parallel_config.get("n_envs", 1))
    args.use_subproc = parallel_config.get("use_subproc", False)

    # Auto-detect CPU count if n_envs is None/null
    if args.n_envs is None:
        detected_cores = get_cpu_count()  # Auto-detect all cores

        # Apply profile-specific caps for stability
        if args.profile == "test":
            args.n_envs = 1  # Test: always 1 env (fast startup)
        else:
            args.n_envs = detected_cores  # Prod/custom: use all cores

        logger.info(
            f"Auto-detected {detected_cores} CPU cores, using {args.n_envs} parallel environments (profile: {args.profile})"
        )

    # Device setting: respect user's choice, auto-detect if 'auto'
    requested_device = str(profile.get("device", "auto")).lower()

    if requested_device == "auto":
        # Auto-detect best available device
        import torch

        if torch.cuda.is_available():
            args.device = "cuda"
            logger.info(f"Auto-detected CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            args.device = "cpu"
            logger.info("No CUDA GPU detected, using CPU")
    else:
        args.device = requested_device
        logger.info(f"Using device: {args.device}")

    total_cores = get_cpu_count()
    gpu_summary = "CUDA unavailable"
    cuda_devices = 0
    try:
        import torch

        if torch.cuda.is_available():
            cuda_devices = torch.cuda.device_count()
            gpu_names = []
            for idx in range(cuda_devices):
                with torch.cuda.device(idx):
                    gpu_names.append(torch.cuda.get_device_name(idx))
            gpu_summary = "; ".join(gpu_names) if gpu_names else "Detected CUDA device"
        else:
            gpu_summary = "No CUDA device detected"
    except Exception as exc:  # pragma: no cover - defensive logging
        gpu_summary = f"GPU query failed ({exc.__class__.__name__})"

    logger.info(
        "Hardware summary: total_cores=%s | n_envs=%s | subproc=%s | device=%s | cuda_devices=%s | gpu=%s",
        total_cores,
        args.n_envs,
        args.use_subproc,
        args.device,
        cuda_devices,
        gpu_summary,
    )


def create_environment(args, context, env_rank: int = 0):
    """Create RL environment for training."""

    logger.info(f"[ENV {env_rank}] Creating environment (this takes 30-60s per env)...")

    from src.ga.evaluator.fitness import evaluate as evaluate_fitness
    from src.ga.population import generate_course_group_aware_population

    # CRITICAL: Inside SubprocVecEnv worker processes, we CANNOT use nested multiprocessing
    # This function runs inside each of the 16 parallel RL environments (daemon processes)
    # Daemon processes cannot spawn child processes in Python
    # Solution: Always use sequential population generation in RL training
    logger.info(
        f"[ENV {env_rank}] Generating initial population ({args.population_size} individuals)..."
    )
    initial_population = generate_course_group_aware_population(
        n=args.population_size,
        context=context,
        parallel=False,  # MUST be False - we're already inside 16 parallel processes
    )

    logger.info(f"[ENV {env_rank}] Evaluating {len(initial_population)} individuals...")

    logger.info(
        f"[ENV {env_rank}] Evaluating population on CPU ({len(initial_population)} individuals)..."
    )
    for idx, individual in enumerate(initial_population):
        if idx % 20 == 0:
            logger.info(
                f"[ENV {env_rank}]    ... evaluated {idx}/{len(initial_population)}"
            )
        fitness = evaluate_fitness(
            individual,
            courses=context.courses,
            instructors=context.instructors,
            groups=context.groups,
            rooms=context.rooms,
        )
        individual.fitness.values = fitness

    logger.info(
        f"[ENV {env_rank}] [OK] Population initialized with {len(initial_population)} individuals"
    )

    env = ScheduleEnv(
        initial_population=initial_population,
        context=context,
        max_generations=args.max_generations,
        max_steps_per_episode=args.max_steps,
        debug_logging=args.debug_logging,
        env_rank=env_rank,
        debug_log_interval=args.debug_log_interval,
    )

    if args.seed is not None:
        env.reset(seed=args.seed)

    logger.info(
        "Environment created: max_gen=%s, max_steps=%s",
        args.max_generations,
        args.max_steps,
    )
    logger.info("Observation space: %s", env.observation_space)
    logger.info("Action space: %s", env.action_space)

    return env


def make_parallel_envs(args, context, n_envs: int = 8, use_subproc: bool = True):
    """Create parallel environments for faster training.

    Args:
        args: Training arguments
        context: Scheduling context
        n_envs: Number of parallel environments
        use_subproc: Use SubprocVecEnv (true parallelism) vs DummyVecEnv

    Returns:
        Vectorized environment
    """
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"IMPORTANT: Creating {n_envs} parallel environments")
    logger.info("=" * 80)
    logger.info(
        f"Environment type: {'SubprocVecEnv (true parallelism)' if use_subproc else 'DummyVecEnv (sequential)'}"
    )
    logger.info(f"Population size per env: {args.population_size}")
    logger.info("")
    logger.info("EXPECTED TIME:")
    logger.info("   - Each environment needs 30-60 seconds to initialize")
    logger.info(
        f"   - With {n_envs} envs running in parallel, expect 1-2 MINUTES total"
    )
    logger.info(
        f"   - Total work: {n_envs} envs x {args.population_size} individuals = {n_envs * args.population_size} fitness evaluations"
    )
    logger.info("")
    if use_subproc:
        logger.info("WARNING: SubprocVecEnv worker logs won't appear in console!")
        logger.info("   - Workers run in separate processes (no console output)")
        logger.info("   - This will appear FROZEN for 1-2 minutes - BE PATIENT!")
        logger.info("   - Check logs/training/*.log for worker activity")
    logger.info("=" * 80)
    logger.info("")

    sys.stdout.flush()

    logger.info(f"Creating environment factories for {n_envs} workers...")

    env_profile = os.environ.get("ENVIRONMENT")
    schedule_config = os.environ.get("SCHEDULE_CONFIG")

    def make_env(rank: int):
        """Create a single environment with unique seed.

        CRITICAL: Each worker gets its own copy of context to avoid:
        - Pickling issues with shared references
        - Race conditions from shared state
        - Memory corruption in multiprocessing
        """

        def _init():
            # IMPORTANT: Create deep copy of context for this worker
            # This prevents shared state issues in SubprocVecEnv
            import copy

            worker_context = copy.deepcopy(context)

            if env_profile:
                os.environ["ENVIRONMENT"] = env_profile
            if schedule_config:
                os.environ["SCHEDULE_CONFIG"] = schedule_config
            os.environ["_GA_WORKER_PROCESS"] = "1"

            env = create_environment(args, worker_context, env_rank=rank)
            if args.seed is not None:
                env.reset(seed=args.seed + rank)
            return env

        return _init

    # Create environment factories
    env_fns = [make_env(i) for i in range(n_envs)]
    logger.info(
        f"Environment factories created. Now spawning {n_envs} worker processes..."
    )
    logger.info("THIS WILL TAKE 1-2 MINUTES WITH NO OUTPUT - PLEASE WAIT!")

    import time

    sys.stdout.flush()

    start_time = time.time()

    # Create vectorized environment
    if use_subproc:
        logger.info(
            "Calling SubprocVecEnv(start_method='spawn')... (workers initializing)"
        )
        sys.stdout.flush()
        vec_env: SubprocVecEnv | DummyVecEnv = SubprocVecEnv(
            env_fns, start_method="spawn"
        )
        elapsed = time.time() - start_time
        logger.info(f"SubprocVecEnv created in {elapsed:.1f}s")
    else:
        vec_env = DummyVecEnv(env_fns)
        elapsed = time.time() - start_time

    logger.info(
        f"[OK] Parallel environments ready ({n_envs} workers, took {elapsed:.1f}s)"
    )
    logger.info("Workers are initialized and ready for training!")
    return vec_env


def main() -> None:
    """Main training function."""

    args = parse_args()

    # Initialize structured logging (console only - training runs tracked via output/)
    console_level = "DEBUG"
    setup_logging(
        log_file=None,
        console_level=console_level,
        file_level="DEBUG",
        show_time=True,
        show_path=False,
    )

    if args.list_profiles:
        available = sorted(list_training_profiles())
        if available:
            logger.info("Available training profiles:")
            for name in available:
                logger.info("  - %s", name)
        else:
            logger.warning(
                "No training profiles found. Add YAML files to configs/training/."
            )
        return

    try:
        profile_config = load_training_config(
            profile=args.profile,
            custom_path=args.config,
        )
    except FileNotFoundError as exc:
        logger.error("Training config not found: %s", exc)
        sys.exit(2)

    apply_profile_defaults(args, profile_config)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.debug_logging:
        logger.info(
            "Debug logging ENABLED: environment steps and rollout progress will be logged every %s steps.",
            args.debug_log_interval,
        )

    logger.info("=" * 60)
    logger.info("RL AGENT TRAINING")
    logger.info("=" * 60)
    logger.info(
        "Training profile: %s (timesteps=%s)",
        args.loaded_profile,
        f"{args.timesteps:,}" if args.timesteps else "n/a",
    )
    if args.config:
        logger.info("Custom config overrides: %s", args.config)
    logger.info(f"Agent type: {args.agent_type.upper()}")
    logger.info(f"Data directory: {args.data_dir}")
    if args.seed is not None:
        logger.info("Seed: %s", args.seed)

    # Initialize variables for cleanup
    tensorboard_process = None
    tensorboard_port = 6006

    try:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Load Scheduling Data")
        logger.info("=" * 60)

        _, context = load_input_data(args.data_dir)

        # Right-align counts for consistent formatting
        max_count = max(
            len(context.courses),
            len(context.instructors),
            len(context.rooms),
            len(context.groups),
        )
        count_width = len(str(max_count))
        logger.info("Loaded %*d courses", count_width, len(context.courses))
        logger.info("Loaded %*d instructors", count_width, len(context.instructors))
        logger.info("Loaded %*d rooms", count_width, len(context.rooms))
        logger.info("Loaded %*d groups", count_width, len(context.groups))

        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Create RL Environment")
        logger.info("=" * 60)

        # Create parallel or single environment based on config
        if args.n_envs > 1:
            env = make_parallel_envs(
                args, context, n_envs=args.n_envs, use_subproc=args.use_subproc
            )
            logger.info(
                f"[OK] Using {args.n_envs} parallel environments for {args.n_envs}x speedup"
            )
        else:
            env = create_environment(args, context)
            logger.info("Using single environment (no parallelization)")

        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Initialize Trainer")
        logger.info("=" * 60)

        trainer = RLTrainer(
            env=env,
            agent_type=args.agent_type,
            save_dir=args.save_dir,
            tensorboard_log=args.tensorboard_log,
            verbose=0,  # Silence SB3's output to prevent duplicate logging
            seed=args.seed,
            n_envs=args.n_envs,
            use_subproc=args.use_subproc,
            device=args.device,
            debug_logging=args.debug_logging,
        )

        # Start TensorBoard in background
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3.5: Start TensorBoard")
        logger.info("=" * 60)

        import socket
        import subprocess

        # Check if TensorBoard is already running
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        if is_port_in_use(tensorboard_port):
            logger.info(
                f"TensorBoard already running at http://localhost:{tensorboard_port}"
            )
        else:
            try:
                # Start TensorBoard in background
                tensorboard_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tensorboard.main",
                        "--logdir",
                        args.tensorboard_log,
                        "--port",
                        str(tensorboard_port),
                        "--bind_all",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    ),
                )
                logger.info(
                    f"Started TensorBoard at http://localhost:{tensorboard_port}"
                )
                logger.info("Open in browser to monitor training in real-time!")
            except Exception as e:
                logger.warning(f"Could not start TensorBoard: {e}")
                logger.info(
                    "You can start it manually: uv run tensorboard --logdir %s",
                    args.tensorboard_log,
                )

        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Train Agent")
        logger.info("=" * 60)

        if args.curriculum:
            logger.info("[INFO] Curriculum learning: ENABLED")
            logger.info("[WARN] Curriculum learning not yet fully implemented!")
            logger.info("[INFO] Training will use standard approach for now.")

        logger.info(
            "[DEBUG] About to call trainer.train() - this will start rollout collection"
        )

        sys.stdout.flush()

        trainer.train(
            total_timesteps=args.timesteps,
            progress_bar=False,
        )

        logger.info("\n[OK] Training completed successfully!")

        if not args.no_eval and args.eval_episodes > 0:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 5: Evaluate Agent")
            logger.info("=" * 60)

            metrics = trainer.evaluate(n_eval_episodes=args.eval_episodes)
            logger.info("\nEvaluation Results:")
            logger.info(
                "  Mean Reward: %.2f ± %.2f",
                metrics["mean_reward"],
                metrics["std_reward"],
            )
            logger.info(
                "  Min/Max Reward: %.2f / %.2f",
                metrics["min_reward"],
                metrics["max_reward"],
            )
            logger.info("  Mean Episode Length: %.1f", metrics["mean_length"])

        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Save Model")
        logger.info("=" * 60)

        if args.save_path:
            save_path = args.save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                f"{args.save_prefix}_{args.agent_type}_{args.timesteps}_{timestamp}"
            )

        model_path = trainer.save_model(
            filename=save_path,
            metadata={
                "timesteps": args.timesteps,
                "agent_type": args.agent_type,
                "max_generations": args.max_generations,
                "max_steps": args.max_steps,
                "population_size": args.population_size,
            },
        )

        logger.info("\n[OK] Model saved to: %s", model_path)

        stats = trainer.get_training_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING STATISTICS")
        logger.info("=" * 60)
        logger.info("Total timesteps: %s", f"{stats['total_timesteps']:,}")
        logger.info(
            "Total training time: %.1fs (%.1f min)",
            stats["total_training_time"],
            stats["total_training_time"] / 60,
        )
        logger.info("Training runs: %s", stats["num_training_runs"])
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)

        logger.info("\n" + "=" * 60)
        logger.info("VIEW TRAINING IN TENSORBOARD")
        logger.info("=" * 60)
        logger.info("TensorBoard: http://localhost:%d", tensorboard_port)
        if tensorboard_process:
            logger.info(
                "(TensorBoard will keep running - press Ctrl+C in terminal to stop)"
            )
        else:
            logger.info("\nOr start manually: .\\start_tensorboard.ps1")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        if tensorboard_process:
            logger.info("Stopping TensorBoard...")
            tensorboard_process.terminate()
        sys.exit(1)
    except Exception as exc:
        logger.error("\nTraining failed: %s", exc, exc_info=True)
        if tensorboard_process:
            logger.info("Stopping TensorBoard...")
            tensorboard_process.terminate()
        sys.exit(1)
    finally:
        # Keep TensorBoard running if training completed successfully
        if tensorboard_process and tensorboard_process.poll() is None:
            logger.info(
                "\nTensorBoard is still running at http://localhost:%d",
                tensorboard_port,
            )
            logger.info("Press Ctrl+C to stop it when done viewing.")


if __name__ == "__main__":
    main()
