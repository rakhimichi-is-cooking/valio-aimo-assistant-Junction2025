from functools import lru_cache
from pathlib import Path
import json
import pandas as pd

# --- File Locations ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SALES_AND_DELIVERIES_PATH = DATA_DIR / "valio_aimo_sales_and_deliveries_junction_2025.csv"
PURCHASES_PATH = DATA_DIR / "valio_aimo_purchases_junction_2025.csv"
REPLACEMENT_ORDERS_PATH = DATA_DIR / "valio_aimo_replacement_orders_junction_2025.csv"
PRODUCT_DATA_PATH = DATA_DIR / "valio_aimo_product_data_junction_2025.json"


# --- Loaders ----------------------------------------------------------------

@lru_cache(maxsize=1)
def load_sales_and_deliveries() -> pd.DataFrame:
    """Load Valio Aimo sales + deliveries CSV."""
    df = pd.read_csv(SALES_AND_DELIVERIES_PATH)
    return df


@lru_cache(maxsize=1)
def load_replacement_orders() -> pd.DataFrame:
    """Load product replacement orders."""
    df = pd.read_csv(REPLACEMENT_ORDERS_PATH)
    return df


@lru_cache(maxsize=1)
def load_purchases() -> pd.DataFrame:
    """Load customer purchase orders."""
    df = pd.read_csv(PURCHASES_PATH)
    return df


@lru_cache(maxsize=1)
def load_product_data() -> list[dict]:
    """Load product metadata JSON."""
    with open(PRODUCT_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format cases supported:
    # - {"products": [...]}
    # - [...]
    if isinstance(data, dict) and "products" in data:
        return data["products"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON format in {PRODUCT_DATA_PATH}")


@lru_cache(maxsize=1)
def load_product_dict() -> dict:
    """
    Load product data as a dictionary mapping product_code -> product info.

    Returns:
        dict: {product_code: {name, category, etc}}
    """
    products_list = load_product_data()
    product_dict = {}

    for product in products_list:
        # Try different possible keys for product code
        product_code = product.get('product_code') or product.get('code') or product.get('id')
        if product_code:
            product_dict[str(product_code)] = product

    return product_dict
