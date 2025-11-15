from src.utils.experiment import sanitize_experiment_name


def test_sanitize_basic():
    assert sanitize_experiment_name("My Experiment") == "my-experiment"
    assert sanitize_experiment_name("  Lead__Test  ") == "lead__test"
    assert sanitize_experiment_name("") is None
    assert sanitize_experiment_name(None) is None


def test_sanitize_special_chars():
    name = "Run:Test/Complex*Name?"
    assert sanitize_experiment_name(name) == "run-test-complex-name"


def test_sanitize_truncate():
    long_name = "a" * 100
    assert len(sanitize_experiment_name(long_name)) <= 50
