"""Unit tests for utility functions and helpers."""

import pytest
from src.utils.console_helpers import (
    print_success,
    print_warning,
    print_error,
    print_info,
)
from src.utils.logging_config import setup_logging, get_logger


class TestConsoleHelpers:
    """Test suite for console output helpers."""

    def test_print_success_no_exception(self, capsys):
        """Test print_success doesn't raise exceptions."""
        try:
            print_success("Test message")
            captured = capsys.readouterr()
            assert "[!ok]" in captured.out or "Test message" in captured.out
        except Exception as e:
            pytest.fail(f"print_success raised exception: {e}")

    def test_print_warning_no_exception(self, capsys):
        """Test print_warning doesn't raise exceptions."""
        try:
            print_warning("Test warning")
            captured = capsys.readouterr()
            assert "[!warn]" in captured.out or "Test warning" in captured.out
        except Exception as e:
            pytest.fail(f"print_warning raised exception: {e}")

    def test_print_error_no_exception(self, capsys):
        """Test print_error doesn't raise exceptions."""
        try:
            print_error("Test error")
            captured = capsys.readouterr()
            assert "[!err]" in captured.out or "Test error" in captured.out
        except Exception as e:
            pytest.fail(f"print_error raised exception: {e}")

    def test_print_info_no_exception(self, capsys):
        """Test print_info doesn't raise exceptions."""
        try:
            print_info("Test info")
            captured = capsys.readouterr()
            assert "[!info]" in captured.out or "Test info" in captured.out
        except Exception as e:
            pytest.fail(f"print_info raised exception: {e}")

    def test_console_helpers_with_detail(self, capsys):
        """Test console helpers accept detail parameter."""
        print_success("Main message", "Detail text")
        captured = capsys.readouterr()
        # Just verify no exception raised
        assert len(captured.out) > 0


class TestLoggingConfig:
    """Test suite for logging configuration."""

    def test_setup_logging_returns_logger(self):
        """Test setup_logging returns logger instance."""
        logger = setup_logging(level="INFO")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_get_logger_returns_logger(self):
        """Test get_logger returns logger instance."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_logging_levels_work(self, tmp_path):
        """Test different logging levels."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(level="DEBUG", log_file=log_file)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify log file created
        assert log_file.exists()

        # Verify messages logged
        content = log_file.read_text()
        assert "Info message" in content
        assert "Error message" in content

    def test_verbose_mode_enables_debug(self, tmp_path):
        """Test verbose mode enables DEBUG level."""
        logger = setup_logging(verbose=True)
        assert logger.level <= 10  # DEBUG is 10


class TestTimeHelpers:
    """Test suite for time utility functions."""

    def test_time_helpers_module_exists(self):
        """Test time_helpers module can be imported."""
        try:
            from src.utils import time_helpers

            assert time_helpers is not None
        except ImportError:
            pytest.fail("time_helpers module not found")
