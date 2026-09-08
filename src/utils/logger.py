"""
Logging configuration module for the Fraud Detection system.
Provides formatted console logging with ANSI colors and persistent file logging to artifacts/app.log.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI colors based on log levels for console output."""

    # ANSI color escape codes
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    BASE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{log_color}{self.BASE_FORMAT}{self.RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def get_logger(
    name: str = "fraud_detection",
    log_file: Optional[Union[str, Path]] = "artifacts/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get or create a configured logger instance with colored console handler
    and file handler logging to artifacts/app.log by default.

    Args:
        name: Name of the logger (typically module __name__ or component name).
        log_file: Path to the log file (defaults to 'artifacts/app.log').
        level: Logging severity level (defaults to logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If handlers are already attached, avoid duplicate handlers
    if logger.handlers:
        return logger

    # Prevent propagation to root logger to avoid double printing
    logger.propagate = False

    # Console Handler with Colored Output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # File Handler
    if log_file is not None:
        try:
            file_path = Path(log_file)
            if not file_path.is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent
                file_path = project_root / file_path

            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as err:
            logger.warning(f"Unable to setup file logger at '{log_file}': {err}")

    return logger
