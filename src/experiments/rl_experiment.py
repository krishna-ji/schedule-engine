"""RL experiment runners — wrappers around ``src.rl.helpers``.

Each experiment loads the scheduling context, builds ScheduleEnv,
and delegates training / evaluation to the shared helper functions.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np

from .base import BaseExperiment

# =====================================================================
#  RL helpers (imported lazily so torch isn't loaded until needed)
# =====================================================================


def _rl_imports():
    """Lazy import of RL stack — avoids torch load if only GA is used."""
    from src.rl.helpers import (
        build_notebook_config,
        create_env,
        evaluate_agent,
        load_context,
        run_ablation,
        set_global_seed,
        train_agent,
    )

    return {
        "build_notebook_config": build_notebook_config,
        "create_env": create_env,
        "evaluate_agent": evaluate_agent,
        "load_context": load_context,
        "set_global_seed": set_global_seed,
        "train_agent": train_agent,
        "run_ablation": run_ablation,
    }


# =====================================================================
#  Base RL experiment
# =====================================================================


class _BaseRLExperiment(BaseExperiment):
    """Shared plumbing for RL experiments."""

    def __init__(
        self,
        *,
        pop_size: int = 20,
        max_generations: int = 50,
        max_steps: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_steps = max_steps

    def _get_helpers(self):
        return _rl_imports()

    def _load_context(self, helpers):
        """Seed, build config, load context."""
        helpers["set_global_seed"](self.seed)
        config = helpers["build_notebook_config"](
            seed=self.seed, overrides={"pop_size": self.pop_size}
        )
        _, context = helpers["load_context"](self.data_dir, config)
        return config, context


# =====================================================================
#  Train (PPO / DQN)
# =====================================================================


class RLTrainExperiment(_BaseRLExperiment):
    """Train a single RL agent (PPO or DQN) and evaluate.

    Parameters
    ----------
    agent_type : str
        ``"ppo"`` or ``"dqn"``.
    timesteps : int
        Training timesteps.
    """

    def __init__(
        self,
        *,
        agent_type: str = "ppo",
        timesteps: int = 5000,
        **kwargs,
    ):
        tag = kwargs.pop("tag", f"rl_train_{agent_type}")
        name = kwargs.pop("name", f"RL Train {agent_type.upper()}")
        super().__init__(tag=tag, name=name, **kwargs)
        self.agent_type = agent_type
        self.timesteps = timesteps

    def _execute(self) -> dict[str, Any]:
        h = self._get_helpers()
        _, context = self._load_context(h)

        env = h["create_env"](
            context=context,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            max_steps=self.max_steps,
        )
        self.logger.info(
            f"Env created: obs={env.observation_space.shape}, "
            f"actions={env.action_space.n}"
        )

        self.logger.info(
            f"Training {self.agent_type.upper()} for {self.timesteps} steps..."
        )
        agent, train_time = h["train_agent"](
            agent_type=self.agent_type,
            env=env,
            timesteps=self.timesteps,
            seed=self.seed,
        )
        self.logger.info(f"Trained in {train_time:.2f}s")

        result = h["evaluate_agent"](agent, env, max_generations=self.max_generations)
        self.logger.info(
            f"Best fitness: {result.best_fitness}  "
            f"Convergence: gen {result.convergence_gen}/{self.max_generations}"
        )

        return {
            "config": self._config_dict(),
            "results": {
                "train_time_seconds": train_time,
                "best_fitness": result.best_fitness,
                "convergence_gen": result.convergence_gen,
            },
        }


# =====================================================================
#  Curriculum
# =====================================================================


class RLCurriculumExperiment(_BaseRLExperiment):
    """Curriculum learning — progressive stage training.

    Parameters
    ----------
    stages : list[dict]
        Each dict has keys ``name``, ``max_generations``, ``max_steps``,
        ``timesteps``.
    """

    def __init__(
        self,
        *,
        stages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("tag", "rl_curriculum")
        kwargs.setdefault("name", "RL Curriculum Learning")
        super().__init__(**kwargs)
        self.stages = stages or [
            {"name": "easy", "max_generations": 30, "max_steps": 10, "timesteps": 3000},
            {
                "name": "medium",
                "max_generations": 50,
                "max_steps": 15,
                "timesteps": 4000,
            },
            {"name": "hard", "max_generations": 80, "max_steps": 20, "timesteps": 5000},
        ]

    def _execute(self) -> dict[str, Any]:
        h = self._get_helpers()
        _, context = self._load_context(h)

        agent = None
        stage_results: list[dict[str, Any]] = []
        total_train_time = 0.0

        for i, stage in enumerate(self.stages):
            self.logger.info(
                f"Stage {i + 1}/{len(self.stages)}: {stage['name'].upper()}"
            )

            env = h["create_env"](
                context=context,
                pop_size=self.pop_size,
                max_generations=stage["max_generations"],
                max_steps=stage["max_steps"],
            )

            if agent is None:
                agent, train_time = h["train_agent"](
                    agent_type="ppo",
                    env=env,
                    timesteps=stage["timesteps"],
                    seed=self.seed,
                )
            else:
                agent.set_env(env)
                t0 = time.time()
                agent.learn(total_timesteps=stage["timesteps"], progress_bar=False)
                train_time = time.time() - t0

            total_train_time += train_time
            result = h["evaluate_agent"](
                agent, env, max_generations=stage["max_generations"]
            )
            stage_results.append(
                {
                    "stage": stage["name"],
                    "train_time": train_time,
                    "best_fitness": result.best_fitness,
                    "convergence_gen": result.convergence_gen,
                }
            )
            self.logger.info(
                f"  {stage['name']}: fitness={result.best_fitness}, "
                f"conv={result.convergence_gen}, time={train_time:.2f}s"
            )

        return {
            "config": self._config_dict(),
            "results": {
                "total_train_time_seconds": total_train_time,
                "stage_results": stage_results,
                "final_best_fitness": stage_results[-1]["best_fitness"],
            },
        }


# =====================================================================
#  Specialist
# =====================================================================


class RLSpecialistExperiment(_BaseRLExperiment):
    """Multi-agent specialist selection analysis.

    Parameters
    ----------
    num_episodes : int
        Episodes to run for selection pattern analysis.
    strategy : str
        AgentCoordinator strategy (``"state_based"``).
    """

    def __init__(
        self,
        *,
        num_episodes: int = 5,
        strategy: str = "state_based",
        **kwargs,
    ):
        kwargs.setdefault("tag", "rl_specialist")
        kwargs.setdefault("name", "RL Specialist Agents")
        super().__init__(**kwargs)
        self.num_episodes = num_episodes
        self.strategy = strategy

    def _execute(self) -> dict[str, Any]:
        from src.rl.multi_agent.agent_coordinator import AgentCoordinator

        h = self._get_helpers()
        _, context = self._load_context(h)

        env = h["create_env"](
            context=context,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            max_steps=self.max_steps,
        )

        coordinator = AgentCoordinator(strategy=self.strategy)
        selection_history: list[list[str]] = []

        for episode in range(self.num_episodes):
            obs, info = env.reset()
            sels: list[str] = []
            for _ in range(self.max_steps):
                state = {
                    "generation": info.get("generation", 0),
                    "generations_without_improvement": info.get(
                        "generations_without_improvement", 0
                    ),
                }
                agent = coordinator.select_agent(env.population, state, obs)
                sels.append(agent.name)
                action = env.action_space.sample()
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            selection_history.append(sels)
            self.logger.info(f"Episode {episode + 1}: {sels[:5]}...")

        all_sels = [s for ep in selection_history for s in ep]
        counts = Counter(all_sels)
        self.logger.info(f"Selection distribution: {dict(counts)}")

        return {
            "config": self._config_dict(),
            "results": {
                "total_selections": len(all_sels),
                "selection_distribution": dict(counts),
                "selection_history": selection_history,
            },
        }


# =====================================================================
#  Reward Compare
# =====================================================================


class RLRewardCompareExperiment(_BaseRLExperiment):
    """Compare scalar vs hypervolume reward formulations.

    Parameters
    ----------
    num_transitions : int
        Number of transitions to sample for comparison.
    """

    def __init__(self, *, num_transitions: int = 10, **kwargs):
        kwargs.setdefault("tag", "rl_reward_compare")
        kwargs.setdefault("name", "RL Reward Shaping Comparison")
        super().__init__(**kwargs)
        self.num_transitions = num_transitions

    def _execute(self) -> dict[str, Any]:
        from src.rl.gym_env.reward_calculator import RewardCalculator

        h = self._get_helpers()
        _, context = self._load_context(h)

        env = h["create_env"](
            context=context,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            max_steps=self.max_steps,
        )

        scalar_calc = RewardCalculator(use_hypervolume=False)
        hv_calc = RewardCalculator(use_hypervolume=True)
        population = env.population

        comparison: list[dict[str, Any]] = []
        for i in range(min(self.num_transitions, len(population) - 1)):
            sr, sc = scalar_calc.calculate_reward(
                prev_individual=population[i],
                new_individual=population[i + 1],
                population_diversity=0.1,
                generation=i,
                population=population,
            )
            hr, hc = hv_calc.calculate_reward(
                prev_individual=population[i],
                new_individual=population[i + 1],
                population_diversity=0.1,
                generation=i,
                population=population,
            )
            comparison.append(
                {
                    "transition": i,
                    "scalar_reward": sr,
                    "scalar_components": sc,
                    "hv_reward": hr,
                    "hv_components": hc,
                }
            )
            self.logger.info(f"Transition {i}: scalar={sr:.4f}, hv={hr:.4f}")

        s_vals = [c["scalar_reward"] for c in comparison]
        h_vals = [c["hv_reward"] for c in comparison]
        corr = float(np.corrcoef(s_vals, h_vals)[0, 1]) if len(s_vals) > 1 else 0.0

        return {
            "config": self._config_dict(),
            "results": {
                "scalar": {
                    "mean": float(np.mean(s_vals)),
                    "std": float(np.std(s_vals)),
                },
                "hypervolume": {
                    "mean": float(np.mean(h_vals)),
                    "std": float(np.std(h_vals)),
                },
                "correlation": corr,
                "transition_details": comparison,
            },
        }


# =====================================================================
#  Adaptive Params
# =====================================================================


class RLAdaptiveParamsExperiment(_BaseRLExperiment):
    """Compare fixed vs adaptive GA parameter configurations."""

    def __init__(self, **kwargs):
        kwargs.setdefault("tag", "rl_adaptive_params")
        kwargs.setdefault("name", "RL Adaptive Probabilities")
        super().__init__(**kwargs)

    def _execute(self) -> dict[str, Any]:
        h = self._get_helpers()

        fixed = h["build_notebook_config"](
            seed=self.seed, overrides={"use_adaptive_probabilities": False}
        )
        adaptive = h["build_notebook_config"](
            seed=self.seed, overrides={"use_adaptive_probabilities": True}
        )

        self.logger.info(f"Fixed:    cxpb={fixed.ga.cxpb}, mutpb={fixed.ga.mutpb}")
        self.logger.info(
            f"Adaptive: cxpb={adaptive.ga.cxpb}, mutpb={adaptive.ga.mutpb}"
        )

        return {
            "config": self._config_dict(),
            "results": {
                "fixed": {
                    "cxpb": fixed.ga.cxpb,
                    "mutpb": fixed.ga.mutpb,
                    "adaptive": fixed.ga.use_adaptive_probabilities,
                },
                "adaptive": {
                    "cxpb": adaptive.ga.cxpb,
                    "mutpb": adaptive.ga.mutpb,
                    "adaptive": adaptive.ga.use_adaptive_probabilities,
                },
            },
        }


# =====================================================================
#  Ablation
# =====================================================================


class RLAblationExperiment(_BaseRLExperiment):
    """Systematic ablation study across random / PPO / DQN.

    Parameters
    ----------
    methods : dict
        ``{name: {"agent_type": ...}}`` mapping.
    trials : int
        Repetitions per method.
    timesteps : int
        Training timesteps per trial.
    """

    def __init__(
        self,
        *,
        methods: dict[str, dict[str, Any]] | None = None,
        trials: int = 5,
        timesteps: int = 3000,
        **kwargs,
    ):
        kwargs.setdefault("tag", "rl_ablation")
        kwargs.setdefault("name", "RL Ablation Study")
        super().__init__(**kwargs)
        self.methods = methods or {
            "random": {"agent_type": "random"},
            "ppo": {"agent_type": "ppo"},
            "dqn": {"agent_type": "dqn"},
        }
        self.trials = trials
        self.timesteps = timesteps

    def _execute(self) -> dict[str, Any]:
        h = self._get_helpers()

        self.logger.info(
            f"Ablation: {len(self.methods)} methods x {self.trials} trials"
        )
        results = h["run_ablation"](
            methods=self.methods,
            data_dir=self.data_dir,
            trials=self.trials,
            timesteps=self.timesteps,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            max_steps=self.max_steps,
            seed=self.seed,
        )

        method_stats: dict[str, dict[str, Any]] = {}
        for mk, runs in results.items():
            bf = [r.best_fitness for r in runs]
            cg = [r.convergence_gen for r in runs]
            method_stats[mk] = {
                "best_fitness_mean": float(np.mean(bf)),
                "best_fitness_std": float(np.std(bf)),
                "convergence_mean": float(np.mean(cg)),
                "convergence_std": float(np.std(cg)),
                "raw_fitness": bf,
                "raw_convergence": cg,
            }
            self.logger.info(
                f"{mk.upper()}: fitness={np.mean(bf):.2f}±{np.std(bf):.2f}, "
                f"conv={np.mean(cg):.1f}±{np.std(cg):.1f}"
            )

        return {
            "config": self._config_dict(),
            "results": method_stats,
        }


# =====================================================================
#  Hyperparameter sweep
# =====================================================================


class RLHyperparamSweepExperiment(_BaseRLExperiment):
    """Learning-rate sensitivity sweep.

    Parameters
    ----------
    learning_rates : list[float]
        Learning rates to test.
    timesteps : int
        Training timesteps per LR.
    """

    def __init__(
        self,
        *,
        learning_rates: list[float] | None = None,
        timesteps: int = 3000,
        **kwargs,
    ):
        kwargs.setdefault("tag", "rl_hyperparam_sweep")
        kwargs.setdefault("name", "RL Hyperparameter Sweep")
        super().__init__(**kwargs)
        self.learning_rates = learning_rates or [1e-4, 3e-4, 1e-3]
        self.timesteps = timesteps

    def _execute(self) -> dict[str, Any]:
        h = self._get_helpers()
        _, context = self._load_context(h)

        sweep: list[dict[str, Any]] = []
        for lr in self.learning_rates:
            self.logger.info(f"Testing lr={lr:.0e}...")
            env = h["create_env"](
                context=context,
                pop_size=self.pop_size,
                max_generations=self.max_generations,
                max_steps=self.max_steps,
            )
            agent, train_time = h["train_agent"](
                agent_type="ppo",
                env=env,
                timesteps=self.timesteps,
                seed=self.seed,
                learning_rate=lr,
            )
            result = h["evaluate_agent"](
                agent, env, max_generations=self.max_generations
            )
            sweep.append(
                {
                    "learning_rate": lr,
                    "train_time": train_time,
                    "best_fitness": result.best_fitness,
                    "convergence_gen": result.convergence_gen,
                }
            )
            self.logger.info(
                f"  lr={lr:.0e}: fitness={result.best_fitness}, conv={result.convergence_gen}"
            )

        best = min(sweep, key=lambda x: x["best_fitness"])
        self.logger.info(f"Best LR: {best['learning_rate']:.0e}")

        return {
            "config": self._config_dict(),
            "results": {
                "sweep": sweep,
                "best_lr": best["learning_rate"],
                "best_fitness": best["best_fitness"],
            },
        }


# =====================================================================
#  Multi-agent
# =====================================================================


class RLMultiAgentExperiment(_BaseRLExperiment):
    """Multi-episode agent coordination analysis.

    Parameters
    ----------
    num_episodes : int
        Episodes for analysis.
    strategy : str
        AgentCoordinator strategy.
    """

    def __init__(
        self,
        *,
        num_episodes: int = 10,
        strategy: str = "state_based",
        **kwargs,
    ):
        kwargs.setdefault("tag", "rl_multi_agent")
        kwargs.setdefault("name", "RL Multi-Agent Coordination")
        super().__init__(**kwargs)
        self.num_episodes = num_episodes
        self.strategy = strategy

    def _execute(self) -> dict[str, Any]:
        from src.rl.multi_agent.agent_coordinator import AgentCoordinator

        h = self._get_helpers()
        _, context = self._load_context(h)

        env = h["create_env"](
            context=context,
            pop_size=self.pop_size,
            max_generations=self.max_generations,
            max_steps=self.max_steps,
        )

        coordinator = AgentCoordinator(strategy=self.strategy)
        all_sels: list[str] = []
        episode_data: list[dict[str, Any]] = []

        for ep in range(self.num_episodes):
            obs, info = env.reset()
            sels: list[str] = []
            for _ in range(self.max_steps):
                state = {
                    "generation": info.get("generation", 0),
                    "generations_without_improvement": info.get(
                        "generations_without_improvement", 0
                    ),
                }
                agent = coordinator.select_agent(env.population, state, obs)
                sels.append(agent.name)
                action = env.action_space.sample()
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            all_sels.extend(sels)
            episode_data.append({"episode": ep, "selections": sels})
            self.logger.info(f"Episode {ep + 1}/{self.num_episodes}: {len(sels)} steps")

        counts = Counter(all_sels)
        steps = [len(ed["selections"]) for ed in episode_data]

        return {
            "config": self._config_dict(),
            "results": {
                "total_selections": len(all_sels),
                "selection_distribution": dict(counts),
                "steps_per_episode": steps,
                "episode_selections": [ed["selections"] for ed in episode_data],
            },
        }


# =====================================================================
#  Verify
# =====================================================================


class RLVerifyExperiment(BaseExperiment):
    """Component availability check — no training required."""

    def __init__(self, **kwargs):
        kwargs.setdefault("tag", "rl_verify")
        kwargs.setdefault("name", "RL Component Verification")
        kwargs.setdefault("seed", 0)
        super().__init__(**kwargs)

    def _execute(self) -> dict[str, Any]:
        from src.rl.helpers import build_notebook_config

        config = build_notebook_config()

        checks: dict[str, str] = {}
        for label, import_path in [
            ("ScheduleEnv", "src.rl:ScheduleEnv"),
            ("PPO Agent", "src.rl.agents:create_ppo_agent"),
            ("DQN Agent", "src.rl.agents:create_dqn_agent"),
            ("Random Agent", "src.rl.agents:RandomAgent"),
            (
                "AgentCoordinator",
                "src.rl.multi_agent.agent_coordinator:AgentCoordinator",
            ),
            ("RewardCalculator", "src.rl.gym_env.reward_calculator:RewardCalculator"),
        ]:
            mod_path, attr = import_path.split(":")
            try:
                import importlib

                mod = importlib.import_module(mod_path)
                getattr(mod, attr)
                checks[label] = "available"
            except Exception as e:
                checks[label] = f"error: {e}"

        available = sum(1 for v in checks.values() if v == "available")
        self.logger.info(f"Components: {available}/{len(checks)} available")
        for k, v in checks.items():
            self.logger.info(f"  {k:20s}: {v}")

        return {
            "config": {
                "rl_enabled": config.rl.enabled,
                "default_agent": config.rl.default_agent,
            },
            "results": {
                "components_available": available,
                "components_total": len(checks),
                "component_status": checks,
            },
        }
