"""
Integration tests for Phase 1 & 2 production readiness.

Tests core functionality without requiring heavy dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loader():
    """Test hierarchical config loader."""
    from src.config.loader import load_config, _check_hierarchical_structure
    
    # Check hierarchical structure exists
    assert _check_hierarchical_structure(), "Hierarchical structure should exist"
    
    # Load config
    config = load_config()
    assert config is not None, "Config should load"
    assert hasattr(config, 'ga'), "Config should have GA section"
    assert hasattr(config, 'rl'), "Config should have RL section"
    assert hasattr(config, 'time'), "Config should have time section"
    
    print("✓ Config loader works with hierarchical structure")


def test_heuristics_registry():
    """Test heuristics registry and registration."""
    from src.heuristics.registry import (
        get_registry,
        get_all_heuristics,
        HeuristicCategory,
        get_heuristics_by_category,
    )
    
    # Import heuristics to register them
    import src.heuristics.construction
    import src.heuristics.perturbation
    import src.heuristics.improvement
    import src.heuristics.diversity
    import src.heuristics.meta
    
    # Get all heuristics
    registry = get_registry()
    assert len(registry) > 0, "Should have registered heuristics"
    assert len(registry) == 19, f"Should have 19 heuristics, got {len(registry)}"
    
    # Check categories
    categories = set(h.category for h in registry.values())
    assert len(categories) == 5, f"Should have 5 categories, got {len(categories)}"
    
    # Check each category
    for category in HeuristicCategory:
        cat_heuristics = get_heuristics_by_category(category)
        assert len(cat_heuristics) > 0, f"Category {category} should have heuristics"
    
    print(f"✓ Heuristics registry works ({len(registry)} operators)")


def test_ga_core_imports():
    """Test that core GA modules can be imported."""
    from src.config import init_config, get_config
    from src.core.ga_scheduler import GAScheduler
    from src.ga.population import generate_course_group_aware_population
    
    # Initialize config
    init_config()
    config = get_config()
    
    # Check GA config
    assert hasattr(config.ga, 'ngen'), "GA config should have ngen"
    assert hasattr(config.ga, 'pop_size'), "GA config should have pop_size"
    assert hasattr(config.ga, 'cxpb'), "GA config should have cxpb"
    assert hasattr(config.ga, 'mutpb'), "GA config should have mutpb"
    
    print("✓ Core GA modules import successfully")


def test_config_domains():
    """Test that all config domains are properly loaded."""
    from src.config import init_config, get_config
    
    init_config()
    config = get_config()
    
    # Check common domain
    assert hasattr(config, 'time'), "Should have time config"
    assert hasattr(config, 'io'), "Should have I/O config"
    assert hasattr(config, 'parallel'), "Should have parallel config"
    
    # Check GA domain
    assert hasattr(config, 'ga'), "Should have GA config"
    assert hasattr(config, 'hard_constraints'), "Should have hard constraints"
    assert hasattr(config, 'soft_constraints'), "Should have soft constraints"
    assert hasattr(config, 'repair'), "Should have repair config"
    assert hasattr(config, 'heuristics'), "Should have heuristics config"
    
    # Check RL domain
    assert hasattr(config, 'rl'), "Should have RL config"
    assert hasattr(config.rl, 'enabled'), "RL should have enabled flag"
    assert hasattr(config.rl, 'agent'), "RL should have agent config"
    assert hasattr(config.rl, 'training'), "RL should have training config"
    
    print("✓ All config domains loaded correctly")


def test_heuristics_config_integration():
    """Test that heuristics config integrates with registry."""
    from src.config import init_config, get_config
    from src.heuristics.registry import get_enabled_heuristics
    
    # Import heuristics first
    import src.heuristics.construction
    import src.heuristics.perturbation
    import src.heuristics.improvement
    import src.heuristics.diversity
    import src.heuristics.meta
    
    init_config()
    config = get_config()
    
    # Check heuristics config exists
    assert hasattr(config, 'heuristics'), "Config should have heuristics section"
    
    # Get enabled heuristics
    enabled = get_enabled_heuristics()
    assert len(enabled) > 0, "Should have enabled heuristics"
    
    # Check priorities are applied
    priorities = [h.priority for h in enabled.values()]
    assert priorities == sorted(priorities), "Heuristics should be sorted by priority"
    
    print(f"✓ Heuristics config integration works ({len(enabled)} enabled)")


def test_environment_profiles():
    """Test that different environment profiles load correctly."""
    import os
    from src.config.loader import load_config
    
    # Test with different environments
    for env in ['test', 'prod', 'med']:
        os.environ['ENVIRONMENT'] = env
        config = load_config()
        
        assert config is not None, f"Config should load for {env}"
        assert hasattr(config, 'environment'), f"Config should have environment for {env}"
        assert config.environment == env, f"Environment should be {env}"
        
        # Check GA settings differ per environment
        if env == 'test':
            assert config.ga.ngen == 30, "Test should have 30 generations"
            assert config.ga.pop_size == 10, "Test should have 10 population"
        elif env == 'prod':
            assert config.ga.ngen == 1000, "Prod should have 1000 generations"
            assert config.ga.pop_size == 100, "Prod should have 100 population"
        elif env == 'med':
            assert config.ga.ngen == 200, "Med should have 200 generations"
            assert config.ga.pop_size == 50, "Med should have 50 population"
    
    # Reset to test
    os.environ['ENVIRONMENT'] = 'test'
    
    print("✓ All environment profiles load correctly")


def main():
    """Run all tests."""
    print("="*60)
    print("Phase 1 & 2 Integration Tests")
    print("="*60)
    
    tests = [
        test_config_loader,
        test_heuristics_registry,
        test_ga_core_imports,
        test_config_domains,
        test_heuristics_config_integration,
        test_environment_profiles,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
