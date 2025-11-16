from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Data file paths (adjust names to match real files)
SALES_AND_DELIVERIES_PATH = DATA_DIR / "sales_and_deliveries.csv"
REPLACEMENT_ORDERS_PATH = DATA_DIR / "replacement_orders.csv"
PURCHASES_PATH = DATA_DIR / "purchases.csv"
PRODUCT_DATA_PATH = DATA_DIR / "product_data.json"

# LM Studio / LLM config
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen3-v1-8b")

# API / CORS config
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")
