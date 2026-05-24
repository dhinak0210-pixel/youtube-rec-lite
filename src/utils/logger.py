import sys
from loguru import logger
from src.config import BASE_DIR

# Configure logger
logger.remove()  # Remove default logger

# Log to stdout with a clean and professional format
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Also log to a file for persistence
logger.add(
    BASE_DIR / "logs" / "recosystem.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

logger.info("Structured logging system initialized successfully.")
