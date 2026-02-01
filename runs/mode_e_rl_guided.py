#!/usr/bin/env python3
"""
Mode E: RL-Guided NSGA-II

Full deployment with RL-guided heuristic selection using Q-learning.

Usage:
    python runs/mode_e_rl_guided.py
"""
from __future__ import annotations

import copy
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.io.decoder import decode_individual
from schedule_engine.notebooks.core import (
    EvolutionStats,
    course_aware_crossover,
    create_evaluator,
    create_random_individual,
    get_best_individual,
    get_constraint_breakdown,
    load_data,
    print_constraint_details,
    setup_deap,
    smart_mutation,
    stats_to_ga_metrics,
    track_nsga_metrics,
)
from schedule_engine.notebooks.strategies import SimpleRLSelector
from schedule_engine.notebooks.viz import print_summary
from schedule_engine.utils.json_utils import to_jsonable
from schedule_engine.workflows.reporting import generate_reports


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_e_rl_guided.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("mode_e_rl_guided")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    """Run Mode E: RL-Guided NSGA-II."""

    # CONFIGURATION

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters
    POP_SIZE = 50
    NGEN = 100
    CXPB = 0.9
    MUTPB = 0.2
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # MODE E: RL-guided selection (Q-learning)
    REPAIR_PROB = 0.3
    LEARNING_RATE = 0.2
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    LOG_INTERVAL = 10

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_e_rl_guided" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE E: RL-GUIDED NSGA-II")
    logger.info("=" * 60)
    logger.info(
        f"Config: pop={POP_SIZE}, ngen={NGEN}, lr={LEARNING_RATE}, eps={EPSILON_START}->{EPSILON_END}"
    )
    logger.info(f"Output: {OUTPUT_DIR}")

    # LOAD DATA

    logger.info("Loading data...")
    data = load_data(
        data_dir=DATA_DIR,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )
    logger.info(f"Data loaded: {data.summary()}")

    evaluate = create_evaluator(data)

    # TEST RL SELECTOR

    logger.info("Testing RL selector...")
    selector = SimpleRLSelector(
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=EPSILON_END,
    )
    test_ind = create_random_individual(data)

    logger.info(f"Initial Q-table: {selector.q_table}")
    logger.info(f"Initial epsilon: {selector.epsilon:.3f}")

    for _ in range(10):
        name, fixes, _reward = selector.apply(test_ind, data, evaluate)
        selector.decay_epsilon()

    logger.info(f"After 10 applications: Q-table={selector.q_table}")
    logger.info(f"Epsilon after decay: {selector.epsilon:.3f}")

    # RUN RL-GUIDED NSGA-II

    logger.info("Starting RL-Guided NSGA-II evolution...")

    start = time.time()
    setup_deap(FITNESS_WEIGHTS)
    selector = SimpleRLSelector(
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=EPSILON_END,
    )

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", lambda: creator.Individual(create_random_individual(data))
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", course_aware_crossover)
    toolbox.register("mutate", lambda ind: smart_mutation(ind, data))
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    stats = EvolutionStats()
    total_repairs = 0
    epsilon_history: list[float] = []
    q_table_history: list[dict[str, float]] = []
    rewards_history: list[float] = []

    for gen in range(NGEN):
        offspring = [copy.deepcopy(ind) for ind in toolbox.select(pop, len(pop))]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # MODE E: RL-Guided Repair
        gen_rewards: list[float] = []
        for ind in offspring:
            if random.random() < REPAIR_PROB:
                genes = list(ind)
                _, fixes, reward = selector.apply(genes, data, evaluate)
                total_repairs += fixes
                ind[:] = genes
                del ind.fitness.values
                gen_rewards.append(reward)

        if gen_rewards:
            rewards_history.append(float(np.mean(gen_rewards)))
        else:
            rewards_history.append(0.0)

        # Decay epsilon
        selector.decay_epsilon()

        # Evaluate
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, POP_SIZE)

        # Stats
        hard_vals = [ind.fitness.values[0] for ind in pop]
        soft_vals = [ind.fitness.values[1] for ind in pop]
        stats.generations.append(gen)
        stats.min_hard.append(float(min(hard_vals)))
        stats.avg_hard.append(float(np.mean(hard_vals)))
        stats.max_hard.append(float(max(hard_vals)))
        stats.feasible_count.append(sum(1 for h in hard_vals if h == 0))
        stats.min_soft.append(float(min(soft_vals)))
        stats.avg_soft.append(float(np.mean(soft_vals)))
        epsilon_history.append(selector.epsilon)
        if selector.q_table:
            action_qs = {action: 0.0 for action in selector.actions}
            for state_qs in selector.q_table.values():
                for action, q_value in state_qs.items():
                    action_qs[action] += q_value
            action_qs = {
                action: q_value / len(selector.q_table)
                for action, q_value in action_qs.items()
            }
        else:
            action_qs = {action: 0.0 for action in selector.actions}
        q_table_history.append(action_qs)
        track_nsga_metrics(pop, stats, data)

        if gen % LOG_INTERVAL == 0 or gen == NGEN - 1:
            best_ind = min(
                pop, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
            )
            breakdown = get_constraint_breakdown(list(best_ind), data)
            hard_names = {
                "student_group_exclusivity",
                "instructor_exclusivity",
                "instructor_qualifications",
                "room_suitability",
                "room_exclusivity",
                "instructor_time_availability",
                "room_time_availability",
                "course_completeness",
            }
            hard_bd = {k: v for k, v in breakdown.items() if k in hard_names}
            soft_bd = {k: v for k, v in breakdown.items() if k not in hard_names}
            print_constraint_details(hard_bd, soft_bd, gen)
            q_str = ", ".join(f"{k}:{v:.2f}" for k, v in action_qs.items())
            logger.debug(f"Gen {gen}: epsilon={selector.epsilon:.3f}, Q=[{q_str}]")

    stats.elapsed_time = time.time() - start
    logger.info(
        f"Evolution completed in {stats.elapsed_time:.1f}s (total repairs: {total_repairs})"
    )
    logger.info(f"Final Q-table: {selector.q_table}")
    logger.info(f"Final epsilon: {selector.epsilon:.4f}")

    final_pop = pop

    # RESULTS & VISUALIZATION

    logger.info("Generating results and visualizations...")

    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)
    print_summary(final_pop, stats, breakdown)

    # Plot Q-value evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Q-values over time
    ax1 = axes[0, 0]
    heuristic_names = list(q_table_history[0].keys())
    for name in heuristic_names:
        q_vals = [q[name] for q in q_table_history]
        ax1.plot(q_vals, label=name)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Q-Value")
    ax1.set_title("Q-Value Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(epsilon_history, color="red", linewidth=2)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay (Exploration → Exploitation)")
    ax2.grid(True, alpha=0.3)

    # Rewards over time
    ax3 = axes[1, 0]
    ax3.plot(rewards_history, color="green", alpha=0.7)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Mean Reward")
    ax3.set_title("RL Rewards (Fitness Improvement)")
    ax3.grid(True, alpha=0.3)

    # Final Q-values bar chart
    ax4 = axes[1, 1]
    final_q = q_table_history[-1] if q_table_history else {}
    names = list(final_q.keys())
    values = list(final_q.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))  # type: ignore[attr-defined]
    ax4.barh(names, values, color=colors)
    ax4.set_xlabel("Q-Value")
    ax4.set_title("Final Q-Values (Learned Preferences)")
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mode_e_rl_learning.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_e_rl_learning.png'}")

    # EXPORT RESULTS

    logger.info("Exporting full results...")
    best_schedule = decode_individual(
        best, data.courses, data.instructors, data.groups, data.rooms
    )
    ga_metrics = stats_to_ga_metrics(stats)
    generate_reports(
        decoded_schedule=best_schedule,
        metrics=ga_metrics,
        population=final_pop,
        qts=data.qts,
        output_dir=str(OUTPUT_DIR),
        course_map=data.courses,
    )

    metadata = {
        "experiment": "mode_e_rl_guided",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
            "repair_prob": REPAIR_PROB,
            "learning_rate": LEARNING_RATE,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
        },
        "results": {
            "elapsed_time": stats.elapsed_time,
            "total_repairs": total_repairs,
            "final_min_hard": stats.min_hard[-1] if stats.min_hard else None,
            "final_min_soft": stats.min_soft[-1] if stats.min_soft else None,
            "final_feasible_count": (
                stats.feasible_count[-1] if stats.feasible_count else 0
            ),
            "final_epsilon": epsilon_history[-1] if epsilon_history else None,
            "final_q_table": q_table_history[-1] if q_table_history else {},
        },
        "constraint_breakdown": breakdown,
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(to_jsonable(metadata), f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE E COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
