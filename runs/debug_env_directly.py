#!/usr/bin/env python3
"""Debug Environment Directly - Bypass SB3 to test raw environment."""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv


def test_environment_directly():
    """Test the environment directly without SB3."""

    print("🔬 ENVIRONMENT DIRECT TEST")
    print("=" * 50)

    # Create environment with fast settings
    env = PymooHyperHeuristicEnv(
        pkl_path=".cache/events_with_domains.pkl",
        max_generations=5,  # Very short for quick test
        pop_size=20,  # Small population
        seed=42,
        acceptance_tolerance=0.0,
    )

    print("✅ Environment created successfully")
    print(f"   Max generations: {env.max_generations}")
    print(f"   Population size: {env.pop_size}")
    print(f"   Action space: {env.action_space.n} actions")
    print(f"   Observation shape: {env.observation_space.shape}")

    # Test reset
    print("\\n🔄 Testing reset()...")
    t0 = time.perf_counter()
    obs, info = env.reset()
    reset_time = time.perf_counter() - t0

    print(f"✅ Reset completed in {reset_time:.2f}s")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Best hard: {info.get('best_hard', 'N/A')}")
    print(f"   Best soft: {info.get('best_soft', 'N/A')}")

    # Test action masking
    print("\\n🎭 Testing action_masks()...")
    masks = env.action_masks()
    print(f"✅ Action masks: {masks}")
    print(f"   Actions blocked: {np.where(~masks)[0].tolist()}")
    print(f"   Actions available: {np.where(masks)[0].tolist()}")

    # Test a few steps
    print("\\n👟 Testing step() function...")
    total_time = 0
    for step_num in range(3):  # Just 3 steps
        # Pick a random valid action
        valid_actions = np.where(masks)[0]
        action = np.random.choice(valid_actions)

        print(f"\\n  Step {step_num + 1}: Action {action}")
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.perf_counter() - t0
        total_time += step_time

        print(f"    Time: {step_time:.2f}s, Reward: {reward:.3f}")
        print(f"    Terminated: {terminated}, Truncated: {truncated}")
        print(f"    Best hard: {info.get('best_hard', 'N/A')}")
        print(f"    Generation: {env._gen}/{env.max_generations}")

        # Update masks for next step
        masks = env.action_masks()

        if terminated or truncated:
            print(f"    Episode ended: terminated={terminated}, truncated={truncated}")
            break

    print("\\n📊 SUMMARY:")
    print(f"   Total step time: {total_time:.2f}s")
    print(f"   Average step time: {total_time/(step_num+1):.2f}s")
    print(f"   Final generation: {env._gen}/{env.max_generations}")

    env.close()

    # Test multiple resets to check for memory leaks
    print("\\n🔄 Testing multiple resets...")
    times = []
    for i in range(3):
        env = PymooHyperHeuristicEnv(
            pkl_path=".cache/events_with_domains.pkl",
            max_generations=3,
            pop_size=10,
            seed=42 + i,
        )
        t0 = time.perf_counter()
        obs, info = env.reset()
        times.append(time.perf_counter() - t0)
        env.close()
        print(f"   Reset {i+1}: {times[-1]:.2f}s")

    print("\\n✅ ENVIRONMENT DIRECT TEST COMPLETE")
    print(f"   Average reset time: {np.mean(times):.2f}s")
    print("   Environment appears to be working correctly")


if __name__ == "__main__":
    test_environment_directly()
