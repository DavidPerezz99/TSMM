import logging
import os


def setup_logger(log_file: str) -> logging.Logger:
    """Setup logger with file and console handlers.

    This function is idempotent per-module: existing handlers are cleared so
    repeated calls (e.g. across many bulk_search runs) do not duplicate
    log output.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers when called multiple times in the same process
    logger.handlers = []

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger