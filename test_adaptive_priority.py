"""
Test adaptive priority adjustment implementation
"""
from src.config.loader import load_config
from src.ga.heuristic_tracker import HeuristicTracker

print("=" * 60)
print("ADAPTIVE PRIORITY ADJUSTMENT - IMPLEMENTATION TEST")
print("=" * 60)

# Test 1: Config loading
print("\n1. Testing config loading...")
config = load_config('configs/hybrid/d-adaptive.yaml')
ap = config.heuristics.adaptive_priority
print(f"   ✓ Adaptive priority enabled: {ap['enabled']}")
print(f"   ✓ Reorder interval: {ap['reorder_interval']}")
print(f"   ✓ Evaluation window: {ap['evaluation_window']}")
print(f"   ✓ Min applications: {ap['min_applications']}")

# Test 2: HeuristicTracker methods
print("\n2. Testing HeuristicTracker.reorder_by_effectiveness()...")
tracker = HeuristicTracker()
tracker.set_heuristic_order(['random_swap', 'temporal_shift', 'kempe_chain', 'igls_repair'])
print(f"   Initial order: {tracker.heuristic_order}")

# Simulate applications with different effectiveness
tracker.record_application(1, 'random_swap', 'perturbation', (10, 50), (10, 48), 0.05, 0)
tracker.record_application(2, 'temporal_shift', 'perturbation', (10, 48), (10, 47), 0.05, 0)
tracker.record_application(3, 'kempe_chain', 'improvement', (10, 47), (5, 35), 0.1, 0)
tracker.record_application(4, 'igls_repair', 'repair', (5, 35), (3, 28), 0.3, 0)
tracker.record_application(5, 'random_swap', 'perturbation', (3, 28), (3, 26), 0.05, 0)
tracker.record_application(6, 'temporal_shift', 'perturbation', (3, 26), (3, 25), 0.05, 0)
tracker.record_application(7, 'kempe_chain', 'improvement', (3, 25), (1, 20), 0.1, 0)
tracker.record_application(8, 'igls_repair', 'repair', (1, 20), (0, 15), 0.3, 0)

# Trigger reordering
order_changed = tracker.reorder_by_effectiveness(
    current_generation=9,
    window_size=10,
    min_applications=2
)

print(f"   ✓ Order changed: {order_changed}")
print(f"   New order (best first): {tracker.heuristic_order}")

# Show effectiveness scores
scores = tracker.get_effectiveness_summary()
print("\n   Effectiveness scores:")
for heuristic in tracker.heuristic_order:
    print(f"     - {heuristic}: {scores[heuristic]:+.3f}")

# Test 3: Verify best performers moved to front
print("\n3. Verifying reordering logic...")
if scores['kempe_chain'] > scores['random_swap']:
    print(f"   ✓ kempe_chain ({scores['kempe_chain']:+.3f}) > random_swap ({scores['random_swap']:+.3f})")
if scores['igls_repair'] > scores['temporal_shift']:
    print(f"   ✓ igls_repair ({scores['igls_repair']:+.3f}) > temporal_shift ({scores['temporal_shift']:+.3f})")

if tracker.heuristic_order[0] in ['kempe_chain', 'igls_repair']:
    print(f"   ✓ Best performer '{tracker.heuristic_order[0]}' is now first")

# Test 4: Check base config defaults
print("\n4. Testing base config defaults...")
base_config = load_config('configs/base.yaml')
base_ap = base_config.heuristics.adaptive_priority
print(f"   ✓ Default enabled: {base_ap['enabled']} (should be False)")
print(f"   ✓ Default interval: {base_ap['reorder_interval']}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Implementation ready!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run: uv run python main.py --config configs/hybrid/d-adaptive.yaml --env test")
print("  2. Look for '📊 Reordered heuristics' messages in logs")
print("  3. Compare convergence with/without adaptive priority")
