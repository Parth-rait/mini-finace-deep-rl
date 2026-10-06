"""
Central logging setup.

Every script and library module writes to both the console and a rotating
file under results/logs/, so a long-running training or backtest sweep
leaves a record you can grep after the fact instead of terminal scrollback
that's gone once the window closes.

Usage: any module just does

    from minifinrl.core.configs.logging_config import get_logger
    log = get_logger(__name__)
    log.info("...")

`get_logger` configures handlers on first call (idempotent - safe to call
from many modules without producing duplicate log lines) so there's no
separate "call setup_logging() in main()" step to forget.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from minifinrl.core.configs.settings import LOG_DIR

_CONFIGURED = False


def setup_logging(console_level: int = logging.INFO, *, log_file: str = "mini_finrl.log") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)

    # 5 x 5MB rotating files keep recent history without growing unbounded
    # across many training/backtest runs.
    file_handler = logging.handlers.RotatingFileHandler(
        Path(LOG_DIR) / log_file, maxBytes=5_000_000, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger("mini_finrl")
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"mini_finrl.{name}")
