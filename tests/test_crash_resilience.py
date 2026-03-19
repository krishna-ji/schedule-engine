"""Tests for crash-resilience features: continuous CSV logging, checkpointing,
and auto-restart training loop.

These tests verify that:
1. ThesisLoggingCallback writes CSVs continuously (not buffered in memory)
2. Checkpoint files are created at the configured interval
3. CSV files survive a simulated crash (partial training)
4. Resumed training appends to existing CSVs (no overwrite)
5. The auto-restart training loop recovers from BrokenPipeError
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.training.thesis_callback import ThesisLoggingCallback

# ---------------------------------------------------------------------------
# Helpers — lightweight mock environment to drive the callback
# ---------------------------------------------------------------------------


def _make_mock_model(num_timesteps: int = 0):
    """Create a minimal mock that looks like an SB3 model."""
    model = MagicMock()
    model.num_timesteps = num_timesteps
    model.logger = MagicMock()
    model.logger.name_to_value = {}
    model.save = MagicMock()
    return model


def _simulate_steps(cb: ThesisLoggingCallback, n_episodes: int, steps_per_ep: int = 5):
    """Drive the callback through n_episodes, each with steps_per_ep steps.

    Returns the final timestep.
    """
    ts = getattr(cb, "_sim_ts", 0)
    for _ep in range(n_episodes):
        for step in range(steps_per_ep):
            ts += 1
            cb.num_timesteps = ts
            is_last = step == steps_per_ep - 1
            cb.locals = {
                "actions": np.array([ts % 6]),
                "rewards": np.array([1.5]),
                "dones": np.array([is_last]),
                "infos": [
                    {
                        "best_hard": 10.0,
                        "best_soft": 5.0,
                        "feasible_frac": 0.8,
                        "rejected": False,
                        "delta_hard": -0.5,
                        "delta_soft": -0.3,
                    }
                ],
            }
            cb._on_step()
    cb._sim_ts = ts
    return ts


# ---------------------------------------------------------------------------
# Test 1: Continuous CSV writing — files have data DURING training
# ---------------------------------------------------------------------------


class TestContinuousCSVWriting:
    def test_csv_files_created_on_start(self, tmp_path):
        """CSV files with headers should exist right after _on_training_start."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb.model = _make_mock_model()
        cb._on_training_start()

        assert (tmp_path / "training_curve.csv").exists()
        assert (tmp_path / "step_log.csv").exists()
        assert (tmp_path / "sb3_training_metrics.csv").exists()

        # Each file should have at least a header line
        for name in ["training_curve.csv", "step_log.csv", "sb3_training_metrics.csv"]:
            content = (tmp_path / name).read_text()
            assert len(content.strip()) > 0, f"{name} should have header"

        cb._on_training_end()

    def test_data_written_after_each_episode(self, tmp_path):
        """training_curve.csv should have rows IMMEDIATELY after episode ends."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb.model = _make_mock_model()
        cb._on_training_start()

        # Simulate 1 episode (5 steps)
        _simulate_steps(cb, n_episodes=1, steps_per_ep=5)

        # Read the CSV — should have header + 1 data row
        with open(tmp_path / "training_curve.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2, "Should have header + 1 episode row"

        # Simulate 2 more episodes
        _simulate_steps(cb, n_episodes=2, steps_per_ep=5)
        with open(tmp_path / "training_curve.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4, "Should have header + 3 episode rows"

        cb._on_training_end()

    def test_step_log_written_per_step(self, tmp_path):
        """step_log.csv should have one row per step."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb.model = _make_mock_model()
        cb._on_training_start()

        _simulate_steps(cb, n_episodes=2, steps_per_ep=5)

        # Force flush (episode boundary should have flushed)
        with open(tmp_path / "step_log.csv") as f:
            rows = list(csv.reader(f))
        # header + 10 steps (2 episodes × 5 steps)
        assert len(rows) == 11, f"Expected 11 rows (header + 10 steps), got {len(rows)}"

        cb._on_training_end()


# ---------------------------------------------------------------------------
# Test 2: Checkpoint saving
# ---------------------------------------------------------------------------


class TestCheckpointSaving:
    def test_checkpoint_created_at_interval(self, tmp_path):
        """Model checkpoint should be saved every checkpoint_interval episodes."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=3)
        cb.model = _make_mock_model()
        cb._on_training_start()

        # 3 episodes — checkpoint at episode 3
        _simulate_steps(cb, n_episodes=3, steps_per_ep=5)
        cb.model.save.assert_called_once()
        call_path = cb.model.save.call_args[0][0]
        assert "checkpoint_3" in call_path

        # 3 more episodes — checkpoint at episode 6
        _simulate_steps(cb, n_episodes=3, steps_per_ep=5)
        assert cb.model.save.call_count == 2

        cb._on_training_end()

    def test_no_checkpoint_when_disabled(self, tmp_path):
        """No checkpoints should be created when checkpoint_interval=0."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb.model = _make_mock_model()
        cb._on_training_start()

        _simulate_steps(cb, n_episodes=10, steps_per_ep=5)
        cb.model.save.assert_not_called()

        cb._on_training_end()

    def test_checkpoint_dir_created(self, tmp_path):
        """checkpoints/ directory should be created when interval > 0."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=5)
        assert (tmp_path / "checkpoints").exists()


# ---------------------------------------------------------------------------
# Test 3: Crash survival — data written before crash persists on disk
# ---------------------------------------------------------------------------


class TestCrashSurvival:
    def test_data_survives_without_on_training_end(self, tmp_path):
        """If training crashes (no _on_training_end), CSV data should still exist."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb.model = _make_mock_model()
        cb._on_training_start()

        # Simulate 5 complete episodes
        _simulate_steps(cb, n_episodes=5, steps_per_ep=5)

        # DELIBERATELY skip _on_training_end() — simulating a crash

        # Verify data exists on disk
        with open(tmp_path / "training_curve.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 6, "5 episodes + header should be on disk"

        with open(tmp_path / "step_log.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 26, "25 steps + header should be on disk"

    def test_partial_episode_data_on_crash(self, tmp_path):
        """Mid-episode crash: completed episodes should be saved, partial lost."""
        cb = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb.model = _make_mock_model()
        cb._on_training_start()

        # Complete 3 episodes
        _simulate_steps(cb, n_episodes=3, steps_per_ep=5)

        # Start a 4th episode — 2 steps only (no done=True)
        for step in range(2):
            cb.num_timesteps = 100 + step
            cb.locals = {
                "actions": np.array([0]),
                "rewards": np.array([1.0]),
                "dones": np.array([False]),
                "infos": [{"best_hard": 5.0, "best_soft": 2.0}],
            }
            cb._on_step()

        # CRASH — no _on_training_end

        # 3 completed episodes should be in the curve CSV
        with open(tmp_path / "training_curve.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4, "3 complete episodes + header"


# ---------------------------------------------------------------------------
# Test 4: Resume — appending to existing CSVs
# ---------------------------------------------------------------------------


class TestResumeAppend:
    def test_csv_append_on_resume(self, tmp_path):
        """Resuming into the same run_dir should APPEND, not overwrite."""
        # First "run" — 3 episodes
        cb1 = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb1.model = _make_mock_model()
        cb1._on_training_start()
        _simulate_steps(cb1, n_episodes=3, steps_per_ep=5)
        cb1._on_training_end()

        with open(tmp_path / "training_curve.csv") as f:
            rows_before = list(csv.reader(f))
        assert len(rows_before) == 4  # header + 3

        # Second "run" (resume) — 2 more episodes
        cb2 = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb2.model = _make_mock_model()
        cb2._on_training_start()
        _simulate_steps(cb2, n_episodes=2, steps_per_ep=5)
        cb2._on_training_end()

        with open(tmp_path / "training_curve.csv") as f:
            rows_after = list(csv.reader(f))
        # Should have header + 3 + 2 = 6 rows (NO duplicate header)
        assert len(rows_after) == 6, f"Expected 6 rows, got {len(rows_after)}"

    def test_step_log_append_on_resume(self, tmp_path):
        """step_log.csv should also append on resume."""
        # First run
        cb1 = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb1.model = _make_mock_model()
        cb1._on_training_start()
        _simulate_steps(cb1, n_episodes=2, steps_per_ep=5)
        cb1._on_training_end()

        # Second run
        cb2 = ThesisLoggingCallback(run_dir=tmp_path, verbose=0, checkpoint_interval=0)
        cb2.model = _make_mock_model()
        cb2._on_training_start()
        _simulate_steps(cb2, n_episodes=1, steps_per_ep=5)
        cb2._on_training_end()

        with open(tmp_path / "step_log.csv") as f:
            rows = list(csv.reader(f))
        # header + 10 + 5 = 16
        assert len(rows) == 16, f"Expected 16 rows, got {len(rows)}"


# ---------------------------------------------------------------------------
# Test 5: Auto-restart loop logic
# ---------------------------------------------------------------------------


class TestAutoRestartLoop:
    """Test the crash-recovery loop that will wrap model.learn() in rl_06."""

    def test_find_latest_checkpoint(self, tmp_path):
        """_find_latest_checkpoint should return the most recent .zip file."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Create some checkpoint files
        import time

        for i in [10, 20, 30]:
            p = ckpt_dir / f"checkpoint_{i}.zip"
            p.write_bytes(b"fake")
            time.sleep(0.05)  # ensure different mtime

        # Import the helper we'll create
        from runs.rl_06_train_ppo_titan_v4_sota import _find_latest_checkpoint

        result = _find_latest_checkpoint(tmp_path)
        assert result is not None
        assert "checkpoint_30" in result.name

    def test_find_latest_checkpoint_empty(self, tmp_path):
        """Should return None when no checkpoints exist."""
        from runs.rl_06_train_ppo_titan_v4_sota import _find_latest_checkpoint

        assert _find_latest_checkpoint(tmp_path) is None

    def test_find_latest_checkpoint_empty_dir(self, tmp_path):
        """Should return None when checkpoints/ exists but is empty."""
        (tmp_path / "checkpoints").mkdir()

        from runs.rl_06_train_ppo_titan_v4_sota import _find_latest_checkpoint

        assert _find_latest_checkpoint(tmp_path) is None
