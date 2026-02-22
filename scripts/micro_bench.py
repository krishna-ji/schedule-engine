#!/usr/bin/env python3
"""Quick micro-benchmark for pipeline components."""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import quick_setup

logger = quick_setup()

import pickle

import numpy as np

pkl_path = "events_with_domains.pkl"
with open(pkl_path, "rb") as f:
    pkl_data = pickle.load(f)

from src.pipeline.fast_evaluator_batch import (
    fast_evaluate_hard_batch,
    prepare_batch_data,
)
from src.pipeline.fast_evaluator_vectorized import (
    fast_evaluate_hard_vectorized,
    prepare_vectorized_data,
)
from src.pipeline.pymoo_operators import ConstructiveSampling
from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair
from src.pipeline.scheduling_problem import SchedulingProblem

vec_data = prepare_vectorized_data(pkl_data)
batch_data = prepare_batch_data(pkl_data)
prob = SchedulingProblem(pkl_path)

# Sampling
sampling = ConstructiveSampling(pkl_path)
t0 = time.perf_counter()
X = sampling._do(prob, 50)
t_sampling = time.perf_counter() - t0
logger.info("Sampling 50: %.2fs (%.3fs each)", t_sampling, t_sampling / 50)

# Hard eval — batch (5 reps)
t0 = time.perf_counter()
for _ in range(10):
    G_batch = fast_evaluate_hard_batch(X, batch_data)
t_batch = (time.perf_counter() - t0) / 10
logger.info("Hard eval batch (50 inds): %.4fs", t_batch)

# Hard eval — vectorized (5 reps)
t0 = time.perf_counter()
for _ in range(10):
    G_vec = fast_evaluate_hard_vectorized(X, vec_data)
t_vec = (time.perf_counter() - t0) / 10
logger.info("Hard eval vectorized (50 inds): %.4fs", t_vec)
logger.info("Speedup vec/batch: %.1fx", t_batch / t_vec)
logger.info("Equivalence: %s", np.array_equal(G_batch, G_vec))

# Scale test at 200
X200 = np.tile(X, (4, 1))  # 200 inds
t0 = time.perf_counter()
for _ in range(5):
    fast_evaluate_hard_batch(X200, batch_data)
t_batch200 = (time.perf_counter() - t0) / 5

t0 = time.perf_counter()
for _ in range(5):
    fast_evaluate_hard_vectorized(X200, vec_data)
t_vec200 = (time.perf_counter() - t0) / 5
logger.info("Hard eval batch (200 inds): %.4fs", t_batch200)
logger.info("Hard eval vectorized (200 inds): %.4fs", t_vec200)
logger.info("Speedup vec/batch @200: %.1fx", t_batch200 / t_vec200)

# Repair
repairer = BitsetSchedulingRepair(pkl_path)
n_repair = 10
t0 = time.perf_counter()
for i in range(n_repair):
    repairer.repair(X[i].copy())
t_repair = (time.perf_counter() - t0) / n_repair
logger.info("Repair per individual: %.4fs", t_repair)
logger.info("Repair est 200 individuals: %.1fs", t_repair * 200)

# Full _evaluate
out = {}
t0 = time.perf_counter()
prob._evaluate(X, out)
t_eval = time.perf_counter() - t0
logger.info("Full _evaluate (50 inds): %.4fs", t_eval)
logger.info("F shape: %s, G shape: %s", out["F"].shape, out["G"].shape)

# Crossover and mutation timing
from src.pipeline.pymoo_operators import EventBlockCrossover, EventLocalMutation

cx = EventBlockCrossover(pkl_path)
mut = EventLocalMutation(pkl_path)

# Crossover — need parents pairs
n_matings = 25
parents = np.stack([X[:n_matings], X[n_matings : 2 * n_matings]], axis=1)
t0 = time.perf_counter()
for _ in range(10):
    cx._do(prob, parents)
t_cx = (time.perf_counter() - t0) / 10
logger.info("Crossover (%d matings): %.4fs", n_matings, t_cx)

# Mutation
t0 = time.perf_counter()
for _ in range(10):
    mut._do(prob, X.copy())
t_mut = (time.perf_counter() - t0) / 10
logger.info("Mutation (50 inds): %.4fs", t_mut)

logger.info("\n=== SUMMARY ===")
logger.info("Constructive sampling: %.1fms/ind", t_sampling / 50 * 1000)
logger.info(
    "Hard eval vectorized:  %.2fms/ind (%.4fs for 200)", t_vec200 / 200 * 1000, t_vec200
)
logger.info(
    "Hard eval batch:       %.2fms/ind (%.4fs for 200)",
    t_batch200 / 200 * 1000,
    t_batch200,
)
logger.info(
    "Repair:                %.1fms/ind (%.1fs for 200)", t_repair * 1000, t_repair * 200
)
logger.info("Crossover:             %.1fms for %d matings", t_cx * 1000, n_matings)
logger.info("Mutation:              %.1fms for 50 inds", t_mut * 1000)
logger.info("Full _evaluate:        %.1fms for 50 inds", t_eval * 1000)
