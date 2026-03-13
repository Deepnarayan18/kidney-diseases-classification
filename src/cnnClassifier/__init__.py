# src/cnnClassifier/__init__.py 
# Logging setup for entire cnnClassifier package (Krish Naik style) [web:215]

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Logs directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file name with timestamp
log_filename = f"running_logs_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILEPATH = LOG_DIR / log_filename

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:',
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("cnnClassifier")

# Test log
if __name__ == "__main__":
    logger.info("Logging initialized successfully!")
    logger.info(f"Log file: {LOG_FILEPATH}")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
