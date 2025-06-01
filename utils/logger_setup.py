import os
import sys
from loguru import logger

def configure_logger(log_filename="logs/default.log", level="INFO"):
    """
    Set up Loguru logger with file and styled output.
    """
    log_dir = os.path.dirname(log_filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger.remove()

    # Set to write 'w' mode for ease of showing logs for assessment
    logger.add(log_filename, level=level.upper(), mode='w', format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.add(sys.stdout, level=level.upper(), format=(
        "\033[1m<green>{time:YYYY-MM-DD HH:mm:ss}</green>\033[0m | "
        "\033[1m<level>{level}</level>\033[0m | "
        "\033[1m<white>{message}</white>\033[0m"
    ))

    return logger
