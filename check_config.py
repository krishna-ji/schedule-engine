"""Quick script to check what dev config actually loads"""

from src.config import init_config
import sys


def check_config():
    try:
        # Load dev config
        config = init_config("configs/dev.yaml")

        print("\n" + "=" * 60)
        print("DEV CONFIGURATION CHECK")
        print("=" * 60)

        print("\n📊 GA Parameters:")
        print(f"  - Generations (ngen): {config.ga.ngen}")
        print(f"  - Population Size: {config.ga.pop_size}")
        print(f"  - Crossover Prob: {config.ga.cxpb}")
        print(f"  - Mutation Prob: {config.ga.mutpb}")
        print(f"  - Population Strategy: {config.ga.population_strategy}")

        print("\n🔧 Repair Settings:")
        print(f"  - Enabled: {config.repair.enabled}")
        print(f"  - Max Iterations: {config.repair.max_iterations}")
        print(f"  - Selective Mode: {config.repair.selective_mode}")
        print(f"  - Memetic Mode: {config.repair.memetic_mode}")

        print("\n⚡ Enhancements:")
        print(f"  - Memetic Mode: {config.enhancements.memetic_mode}")
        print(f"  - Hypermutation: {config.enhancements.hypermutation.enabled}")
        print(f"  - Greedy Init %: {config.enhancements.greedy_initialization_percent}")

        print("\n🔍 Parallel:")
        print(f"  - Multiprocessing: {config.parallel.use_multiprocessing}")
        print(f"  - Workers: {config.parallel.num_workers}")

        print("\n🎯 Hard Constraints:")
        for name, info in config.hard_constraints.items():
            if info["enabled"]:
                print(f"  - {name}: weight={info['weight']}")

        print("\n✨ Soft Constraints:")
        for name, info in config.soft_constraints.items():
            if info["enabled"]:
                print(f"  - {name}: weight={info['weight']}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    check_config()
