"""Training entry point script for RL hyper-heuristic agents."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from gymnasium import Env as GymEnv

EnvFactory = Callable[[], GymEnv[Any, Any]]

from stable_baselines3.common.vec_env import VecEnv  # noqa: E402

from src.domain.types import SchedulingContext  # noqa: E402
from src.io.data_store import load_input_data
from src.rl.gym_env import ScheduleEnv  # noqa: E402
from src.rl.training import RLTrainer
from src.rl.training.config_loader import (
    DEFAULT_PROFILE,
    list_training_profiles,
    load_training_config,
)
from src.utils.logging_config import get_logger, setup_logging
from src.utils.system_info import get_cpu_count

logger = get_logger(__name__)


class _FitnessVector(Protocol):
    values: tuple[float, float]


class _FitnessAssignable(Protocol):
    fitness: _FitnessVector


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
        help="Training profile defined in src.rl.training.presets",
    )

    parser.add_argument(
        "--config",
        "--config-path",
        dest="config",
        type=str,
        default=None,
        help="Optional custom training override (JSON)",
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

    def pick(field: str, default: Any | None = None) -> Any | None:
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


def create_environment(
    args: argparse.Namespace,
    context: SchedulingContext,
    env_rank: int = 0,
) -> ScheduleEnv:
    """Create RL environment for training."""

    logger.info(f"[ENV {env_rank}] Creating environment (this takes 30-60s per env)...")

    from src.ga.core.evaluator import evaluate as evaluate_fitness
    from src.ga.core.population import (
        generate_course_group_aware_population,
    )

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
        cast(_FitnessAssignable, individual).fitness.values = fitness

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


def make_parallel_envs(
    args: argparse.Namespace,
    context: SchedulingContext,
    n_envs: int = 8,
    use_subproc: bool = True,
) -> VecEnv:
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

    def make_env(rank: int) -> EnvFactory:
        """Create a single environment with unique seed.

        CRITICAL: Each worker gets its own copy of context to avoid:
        - Pickling issues with shared references
        - Race conditions from shared state
        - Memory corruption in multiprocessing
        """

        def _init() -> GymEnv[Any, Any]:
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
    env_fns: list[EnvFactory] = [make_env(i) for i in range(n_envs)]
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

    # Initialize logging (console only - training runs tracked via output/)
    setup_logging(level="DEBUG")

    if args.list_profiles:
        available = sorted(list_training_profiles())
        if available:
            logger.info("Available training profiles:")
            for name in available:
                logger.info("  - %s", name)
        else:
            logger.warning(
                "No training profiles registered. Update src.rl.training.presets to add more."
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
    except ValueError as exc:
        logger.error("Invalid training config override: %s", exc)
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
        logger.info("STEP 1: Setup Output Directory")
        logger.info("=" * 60)

        # Create timestamped output directory
        from pathlib import Path

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            f"rl_training_{args.loaded_profile}_{args.agent_type}_{args.timesteps}"
        )

        output_dir = (
            Path("output")
            / "ga_05_repair_qlearning"
            / f"{timestamp_str}_{experiment_name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        logger.info("Models will be saved to: models/rl_agents/")

        # Configure paths for this run
        run_tensorboard_log = output_dir / "logs" / "tensorboard"

        run_tensorboard_log.parent.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Load Scheduling Data")
        logger.info("=" * 60)

        # Load global config for cohort pairs
        from src.config import get_config

        config = get_config()
        _, context = load_input_data(args.data_dir, config=config)

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
        logger.info("STEP 3: Create RL Environment")
        logger.info("=" * 60)

        # Create parallel or single environment based on config
        env: VecEnv | ScheduleEnv
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
        logger.info("STEP 4: Initialize RL Trainer")
        logger.info("=" * 60)

        trainer = RLTrainer(
            env=env,  # Use the env created above (parallel or single)
            agent_type=args.agent_type,
            save_dir="models/rl_agents",  # Models always go to models/
            tensorboard_log=str(run_tensorboard_log),  # TensorBoard in run output/
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
        def is_port_in_use(port: int) -> bool:
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
        logger.info("STEP 5: Train Agent")
        logger.info("=" * 60)

        if args.curriculum:
            logger.info("[INFO] Curriculum learning: ENABLED")
            logger.info("[INFO] Starting multi-stage curriculum training...")

            # Import curriculum manager
            from src.rl.training.curriculum import (
                CurriculumManager,
                create_default_curriculum,
            )

            # Initialize curriculum manager
            curriculum_stages = create_default_curriculum()
            curriculum_mgr = CurriculumManager(
                context=context,
                stages=curriculum_stages,
                random_seed=args.seed,
            )

            logger.info(f"Curriculum stages: {len(curriculum_stages)}")

            # Train through curriculum stages
            total_trained_steps = 0
            for stage_idx, stage in enumerate(curriculum_stages):
                logger.info(
                    f"\n{'='*60}\nStage {stage_idx+1}/{len(curriculum_stages)}: {stage['name'].upper()}\n{'='*60}"
                )
                logger.info(f"Episodes: {stage['num_episodes']}")
                logger.info(f"Max generations: {stage['max_generations']}")

                # Calculate timesteps for this stage
                stage_timesteps = stage["num_episodes"] * args.max_steps

                # Update environment parameters for stage difficulty
                # (This would require env reconfiguration - simplified for now)

                # Train for stage episodes
                trainer.train(
                    total_timesteps=stage_timesteps,
                    progress_bar=False,
                    reset_num_timesteps=False,  # Accumulate across stages
                )

                total_trained_steps += stage_timesteps

                # Validation check
                if stage_idx < len(curriculum_stages) - 1:  # Not final stage
                    logger.info(f"\nValidating stage '{stage['name']}'...")
                    val_metrics = trainer.evaluate(
                        n_eval_episodes=stage["validation_episodes"]
                    )
                    val_score = val_metrics["mean_reward"]

                    logger.info(f"Validation mean reward: {val_score:.4f}")

                    if curriculum_mgr.should_advance(val_score):
                        logger.info(
                            f"[OK] Stage '{stage['name']}' completed! Advancing..."
                        )
                        curriculum_mgr.advance_stage()
                    else:
                        logger.warning(
                            f"[WARN] Validation score below threshold ({stage.get('threshold', 0.0)})"
                        )
                        logger.info(
                            "Continuing to next stage anyway (linear curriculum)"
                        )
                        curriculum_mgr.advance_stage()

                # Save curriculum progress
                progress_path = trainer.save_dir / "curriculum_progress.json"
                curriculum_mgr.save_progress(str(progress_path))

            logger.info(
                f"\n[OK] Curriculum training completed! Total steps: {total_trained_steps:,}"
            )

        else:
            # Standard training (no curriculum)
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
            logger.info("STEP 6: Evaluate Trained Agent")
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
        logger.info("STEP 7: Save Model")
        logger.info("=" * 60)

        if args.save_path:
            save_path = args.save_path
        else:
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                f"{args.save_prefix}_{args.agent_type}_{args.timesteps}_{datetime_str}"
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

        # Generate visualizations
        logger.info("\n" + "=" * 60)
        logger.info("STEP 8: Generate Training Visualizations")
        logger.info("=" * 60)

        try:
            from pathlib import Path as PathLib

            from src.rl.training.visualizer import generate_visualizations

            # Use run output directory (not model directory) for plots/csv
            # Convert trainer.tensorboard_log to Path if it's a string
            tb_logdir = (
                PathLib(trainer.tensorboard_log)
                if isinstance(trainer.tensorboard_log, str)
                else trainer.tensorboard_log
            )

            generate_visualizations(
                tensorboard_logdir=tb_logdir,
                output_dir=output_dir,  # Changed: use run output dir
                experiment_name=experiment_name,
            )
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
            logger.info(
                "You can manually generate plots later using: uv run visualize-rl"
            )

        # Register experiment in manifest
        exp_manager.register_run(
            runtime_mode="e5",
            config_reference=args.loaded_profile,
            output_path=output_dir,
            experiment_name=experiment_name,
            notes=f"RL training: {args.agent_type.upper()}, {args.timesteps:,} steps, model: {model_path.name}",
        )

        logger.info("\n" + "=" * 60)
        logger.info("OUTPUT LOCATIONS")
        logger.info("=" * 60)
        logger.info("Trained model:   %s", model_path)
        logger.info("Run artifacts:   %s", output_dir)
        logger.info("  - Plots:       %s", output_dir / "plots")
        logger.info("  - Metrics CSV: %s", output_dir / "csv")
        logger.info("  - Logs:        %s", output_dir / "logs")
        logger.info("  - TensorBoard: %s", run_tensorboard_log)

        logger.info("\n" + "=" * 60)
        logger.info("VIEW TRAINING IN TENSORBOARD")
        logger.info("=" * 60)
        logger.info("TensorBoard: http://localhost:%d", tensorboard_port)
        if tensorboard_process:
            logger.info(
                "(TensorBoard will keep running - press Ctrl+C in terminal to stop)"
            )
        else:
            logger.info("\\nOr start manually: .\\\\start_tensorboard.ps1")

        logger.info("\n" + "=" * 60)
        logger.info("OUTPUT LOCATIONS")
        logger.info("=" * 60)
        logger.info("Trained model:   %s", model_path)
        logger.info("Run artifacts:   %s", output_dir)
        logger.info("  - Plots:       %s", output_dir / "plots")
        logger.info("  - Metrics CSV: %s", output_dir / "csv")
        logger.info("  - Logs:        %s", output_dir / "logs")
        logger.info("  - TensorBoard: %s", run_tensorboard_log)

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
