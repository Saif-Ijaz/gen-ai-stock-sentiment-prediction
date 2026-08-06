from loguru import logger

logger.add(
    "logs/system.log",
    rotation="1 MB",
    level="INFO",
    format="{time} | {level} | {message}"
)

def get_logger():
    return logger
