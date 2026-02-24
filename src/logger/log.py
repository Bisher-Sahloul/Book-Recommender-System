import logging
import os
from datetime import datetime

def setup_logging():
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    CURRENT_TIME_STAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"log_{CURRENT_TIME_STAMP}.log"
    log_file_path = os.path.join(LOG_DIR, file_name)

    # Clear previous handlers
    logging.root.handlers.clear()

    logging.basicConfig(
        filename=log_file_path,
        filemode="w",
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG
    )

    # Named logger
    logger = logging.getLogger("book_recommender")
    logger.propagate = True

    logger.info("Logging initialized successfully")
    return logger

# Export a global logger instance
logger = logging.getLogger("book_recommender")
