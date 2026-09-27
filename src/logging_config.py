import logging
import os
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/bds35.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("BDS35")
