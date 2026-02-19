#!/usr/bin/env python3
"""Quick micro-benchmark for pipeline components."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
print(f"Sampling 50: {t_sampling:.2f}s ({t_sampling / 50:.3f}s each)")

# Hard eval — batch (5 reps)
t0 = time.perf_counter()
for _ in range(10):
    G_batch = fast_evaluate_hard_batch(X, batch_data)
t_batch = (time.perf_counter() - t0) / 10
print(f"Hard eval batch (50 inds): {t_batch:.4f}s")

# Hard eval — vectorized (5 reps)
t0 = time.perf_counter()
for _ in range(10):
    G_vec = fast_evaluate_hard_vectorized(X, vec_data)
t_vec = (time.perf_counter() - t0) / 10
print(f"Hard eval vectorized (50 inds): {t_vec:.4f}s")
print(f"Speedup vec/batch: {t_batch / t_vec:.1f}x")
print(f"Equivalence: {np.array_equal(G_batch, G_vec)}")

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
print(f"Hard eval batch (200 inds): {t_batch200:.4f}s")
print(f"Hard eval vectorized (200 inds): {t_vec200:.4f}s")
print(f"Speedup vec/batch @200: {t_batch200 / t_vec200:.1f}x")

# Repair
repairer = BitsetSchedulingRepair(pkl_path)
n_repair = 10
t0 = time.perf_counter()
for i in range(n_repair):
    repairer.repair(X[i].copy())
t_repair = (time.perf_counter() - t0) / n_repair
print(f"Repair per individual: {t_repair:.4f}s")
print(f"Repair est 200 individuals: {t_repair * 200:.1f}s")

# Full _evaluate
out = {}
t0 = time.perf_counter()
prob._evaluate(X, out)
t_eval = time.perf_counter() - t0
print(f"Full _evaluate (50 inds): {t_eval:.4f}s")
print(f"F shape: {out['F'].shape}, G shape: {out['G'].shape}")

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
print(f"Crossover ({n_matings} matings): {t_cx:.4f}s")

# Mutation
t0 = time.perf_counter()
for _ in range(10):
    mut._do(prob, X.copy())
t_mut = (time.perf_counter() - t0) / 10
print(f"Mutation (50 inds): {t_mut:.4f}s")

print("\n=== SUMMARY ===")
print(f"Constructive sampling: {t_sampling / 50 * 1000:.1f}ms/ind")
print(
    f"Hard eval vectorized:  {t_vec200 / 200 * 1000:.2f}ms/ind ({t_vec200:.4f}s for 200)"
)
print(
    f"Hard eval batch:       {t_batch200 / 200 * 1000:.2f}ms/ind ({t_batch200:.4f}s for 200)"
)
print(
    f"Repair:                {t_repair * 1000:.1f}ms/ind ({t_repair * 200:.1f}s for 200)"
)
print(f"Crossover:             {t_cx * 1000:.1f}ms for {n_matings} matings")
print(f"Mutation:              {t_mut * 1000:.1f}ms for 50 inds")
print(f"Full _evaluate:        {t_eval * 1000:.1f}ms for 50 inds")
