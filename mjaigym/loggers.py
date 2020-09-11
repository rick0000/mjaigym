import logging
from logging.handlers import RotatingFileHandler
from logging import StreamHandler


from pathlib import Path
import os

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=10)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


run_folder = "log"

logger_main = setup_logger("logger_main", Path(run_folder) / "logs/logger_main.log")
logger_main.disabled = False

logger_server = setup_logger("logger_server", Path(run_folder) / "logs/logger_server.log")
logger_server.disabled = False
