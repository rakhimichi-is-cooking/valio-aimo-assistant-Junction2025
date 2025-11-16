from __future__ import annotations

from typing import List, Dict, Optional
from functools import lru_cache

import pandas as pd

from backend.forecasting_engine import ForecastingEngine
from backend.pattern_analysis import PatternAnalyzer
from backend.risk_scorer import RiskScorer


# We map the real CSV columns here
REQUIRED_SALES_COLUMNS = [
    "order_number",
    "requested_delivery_date",
    "customer_number",
    "product_code",
    "order_qty",
    "delivered_qty",
]

# Global instances for caching
_pattern_analyzer = None
_risk_scorer = None
_forecasting_engine = None


def _get_pattern_analyzer() -> PatternAnalyzer:
    """Get or create global pattern analyzer instance."""
    global _pattern_analyzer
    if _pattern_analyzer is None:
        _pattern_analyzer = PatternAnalyzer()
    return _pattern_analyzer


def _get_risk_scorer() -> RiskScorer:
    """Get or create global risk scorer instance."""
    global _risk_scorer
    if _risk_scorer is None:
        _risk_scorer = RiskScorer()
    return _risk_scorer


def _get_forecasting_engine(method: str = "prophet") -> ForecastingEngine:
    """Get or create global forecasting engine instance."""
    global _forecasting_engine
    if _forecasting_engine is None:
        try:
            _forecasting_engine = ForecastingEngine(method=method)
        except ImportError:
            # Fallback to simpler methods if preferred method unavailable
            for fallback in ["arima", "xgboost"]:
                try:
                    _forecasting_engine = ForecastingEngine(method=fallback)
                    break
                except ImportError:
                    continue
    return _forecasting_engine


def _validate_sales_columns(sales: pd.DataFrame) -> None:
    """Ensure the sales + deliveries DataFrame has the columns we expect."""
    missing = [c for c in REQUIRED_SALES_COLUMNS if c not in sales.columns]
    if missing:
        raise KeyError(
            f"Missing columns in valio_aimo_sales_and_deliveries_junction_2025.csv: {missing}. "
            "Update shortage_engine.py mappings if the dataset schema changes."
        )


def compute_shortage_events(
    sales: pd.DataFrame,
    replacement_orders: pd.DataFrame | None = None,
    purchases: pd.DataFrame | None = None,
    use_advanced_scoring: bool = True,
    enable_forecasting: bool = False,
) -> List[Dict]:
    """
    Core shortage logic with optional advanced risk scoring.

    Basic mode (use_advanced_scoring=False):
    - A row is a shortage if delivered_qty < order_qty
    - Risk score = (order_qty - delivered_qty) / order_qty   (clipped to [0, 1])

    Advanced mode (use_advanced_scoring=True):
    - Multi-factor risk scoring using historical patterns, customer importance, etc.
    - Historical pattern analysis
    - Optional forecasting (if enable_forecasting=True)

    Args:
        sales: Sales and deliveries dataframe
        replacement_orders: Optional replacement orders data
        purchases: Optional purchases data
        use_advanced_scoring: Enable multi-factor risk scoring
        enable_forecasting: Enable demand forecasting (slower)

    Returns a list of plain dicts; FastAPI will coerce them into ShortageEvent.
    """

    _validate_sales_columns(sales)

    df = sales.copy()

    # Ensure numeric
    df["order_qty"] = pd.to_numeric(df["order_qty"], errors="coerce")
    df["delivered_qty"] = pd.to_numeric(df["delivered_qty"], errors="coerce")

    # Shortfall & risk
    df["shortfall"] = df["order_qty"] - df["delivered_qty"]
    df = df[df["shortfall"] > 0]  # keep only true shortages

    if df.empty:
        return []

    # Basic risk score (always calculated)
    df["risk_score"] = (df["shortfall"] / df["order_qty"]).clip(lower=0.0, upper=1.0)
    df["risk_score"] = df["risk_score"].fillna(0.0)

    events: List[Dict] = []

    for _, row in df.iterrows():
        customer_number = str(row["customer_number"])

        # Basic synthetic name from the ID.
        customer_name = f"Customer {customer_number}"

        event: Dict = {
            "customer_name": customer_name,
            "customer_id": customer_number,
            "sku": str(row["product_code"]),
            "product_name": "",  # can be filled using product_data JSON later
            "ordered_qty": float(row["order_qty"]),
            "delivered_qty": float(row["delivered_qty"]),
            "delivery_date": str(row["requested_delivery_date"]),
            "risk_score": float(row["risk_score"]),
            "suggested_substitutes": [],  # filled in attach_substitutes
        }
        events.append(event)

    # Apply advanced risk scoring if enabled
    if use_advanced_scoring:
        try:
            events = apply_advanced_risk_scoring(
                events=events,
                sales_df=sales,
                enable_forecasting=enable_forecasting
            )
        except Exception as e:
            # If advanced scoring fails, fall back to basic scores
            print(f"Advanced scoring failed, using basic scores: {e}")

    return events


def apply_advanced_risk_scoring(
    events: List[Dict],
    sales_df: pd.DataFrame,
    enable_forecasting: bool = False
) -> List[Dict]:
    """
    Apply advanced multi-factor risk scoring to shortage events.

    Args:
        events: List of shortage events
        sales_df: Full sales dataframe for pattern analysis
        enable_forecasting: Enable forecasting (slower)

    Returns:
        Events with updated risk scores
    """
    if not events:
        return events

    # Get analyzer instances
    pattern_analyzer = _get_pattern_analyzer()
    risk_scorer = _get_risk_scorer()

    # Analyze historical patterns
    product_stats = pattern_analyzer.analyze_shortage_frequency(sales_df)
    customer_stats = pattern_analyzer.analyze_customer_patterns(sales_df)
    seasonal_patterns = pattern_analyzer.analyze_seasonal_patterns(sales_df)

    # Optional: Generate forecasts
    forecast_data = {}
    if enable_forecasting:
        try:
            forecast_engine = _get_forecasting_engine()
            if forecast_engine:
                # Analyze trends for each product
                for event in events:
                    sku = event['sku']
                    if sku not in forecast_data:
                        product_df = sales_df[sales_df['product_code'] == sku]
                        if len(product_df) >= 14:  # Need minimum data
                            trends = pattern_analyzer.detect_trends(product_df)
                            forecast_data[sku] = trends
        except Exception as e:
            print(f"Forecasting failed: {e}")

    # Apply multi-factor risk scoring
    scored_events = risk_scorer.batch_score(
        shortage_events=events,
        product_stats=product_stats,
        customer_stats=customer_stats,
        forecast_data=forecast_data if forecast_data else None,
        seasonal_patterns=seasonal_patterns
    )

    return scored_events


def attach_substitutes(
    product_data: list[dict],
    events: List[Dict],
) -> List[Dict]:
    """
    Enrich shortage events with suggested substitutes.

    MVP implementation: no-op that simply returns the events unchanged.
    Later you can implement:
    - find products with similar category / brand / fat% etc.
    - ensure they are available in stock
    """
    return events
