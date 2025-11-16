"""
Advanced multi-factor risk scoring system for shortage events.

Combines:
- Historical shortage patterns
- Forecasting predictions
- Customer importance
- Product criticality
- Seasonal factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class RiskScorer:
    """
    Multi-factor risk scoring engine.

    Risk score is a composite of:
    1. Shortage severity (how much is missing)
    2. Historical reliability (product's past performance)
    3. Customer importance (business impact)
    4. Forecast trend (predicted demand)
    5. Seasonal impact (time-based factors)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize risk scorer.

        Args:
            weights: Optional custom weights for risk factors
                    Default: {
                        'severity': 0.30,
                        'historical': 0.25,
                        'customer': 0.20,
                        'forecast': 0.15,
                        'seasonal': 0.10
                    }
        """
        self.weights = weights or {
            'severity': 0.30,      # Current shortage severity
            'historical': 0.25,    # Historical reliability
            'customer': 0.20,      # Customer importance
            'forecast': 0.15,      # Forecast trend
            'seasonal': 0.10       # Seasonal factors
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def calculate_severity_score(
        self,
        ordered_qty: float,
        delivered_qty: float
    ) -> float:
        """
        Calculate shortage severity score.

        Args:
            ordered_qty: Quantity ordered
            delivered_qty: Quantity delivered

        Returns:
            Severity score [0-1], where 1 is worst
        """
        if ordered_qty <= 0:
            return 0.0

        shortage_qty = max(0, ordered_qty - delivered_qty)
        shortage_pct = shortage_qty / ordered_qty

        # Non-linear scaling: severe shortages get higher scores
        if shortage_pct >= 0.9:
            return 1.0
        elif shortage_pct >= 0.5:
            return 0.7 + (shortage_pct - 0.5) * 0.75
        else:
            return shortage_pct * 1.4  # Scale up smaller shortages

    def calculate_historical_score(
        self,
        product_stats: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate historical reliability score.

        Args:
            product_stats: Historical statistics for the product

        Returns:
            Historical risk score [0-1], where 1 is worst
        """
        if not product_stats:
            return 0.5  # Neutral if no data

        # Shortage rate from historical data
        shortage_rate = product_stats.get('shortage_rate', 0.0)

        # Average severity when shortages occur
        avg_severity = product_stats.get('avg_shortage_severity', 0.0)

        # Combine rate and severity
        # Products with frequent and severe shortages get high scores
        historical_score = (shortage_rate * 0.6) + (avg_severity * 0.4)

        return min(1.0, historical_score)

    def calculate_customer_score(
        self,
        customer_stats: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate customer importance score.

        Args:
            customer_stats: Customer behavior statistics

        Returns:
            Customer importance score [0-1], where 1 is most important
        """
        if not customer_stats:
            return 0.5  # Neutral if no data

        # Normalize factors to 0-1 scale
        importance = customer_stats.get('importance_score', 0)
        order_frequency = customer_stats.get('order_frequency_per_day', 0)
        total_volume = customer_stats.get('total_volume', 0)

        # Combine factors (higher values = more important customer)
        # We want important customers to have HIGH risk scores (so we prioritize them)
        customer_score = min(1.0, (
            (importance / 10000) * 0.4 +
            (order_frequency * 10) * 0.3 +
            (total_volume / 1000) * 0.3
        ))

        return customer_score

    def calculate_forecast_score(
        self,
        forecast_data: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate forecast-based risk score.

        Args:
            forecast_data: Forecast predictions for the product

        Returns:
            Forecast risk score [0-1], where 1 is worst
        """
        if not forecast_data:
            return 0.5  # Neutral if no forecast

        # Trend direction
        trend = forecast_data.get('trend', 'stable')
        trend_score = {
            'increasing': 0.8,  # High demand predicted = higher risk
            'stable': 0.5,
            'decreasing': 0.3
        }.get(trend, 0.5)

        # Volatility (unpredictable demand = higher risk)
        volatility = forecast_data.get('volatility', 0.0)
        volatility_score = min(1.0, volatility * 2)

        # Combine
        forecast_score = (trend_score * 0.6) + (volatility_score * 0.4)

        return forecast_score

    def calculate_seasonal_score(
        self,
        date: str,
        seasonal_patterns: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate seasonal risk score.

        Args:
            date: Delivery date
            seasonal_patterns: Seasonal pattern data

        Returns:
            Seasonal risk score [0-1]
        """
        if not seasonal_patterns:
            return 0.5  # Neutral if no data

        try:
            dt = pd.to_datetime(date)
            month = dt.month
            day_of_week = dt.dayofweek

            # Check if this is a peak period
            monthly_data = seasonal_patterns.get('monthly', {})
            peak_month = monthly_data.get('peak_month', 0)

            weekly_data = seasonal_patterns.get('weekly', {})
            peak_day = weekly_data.get('peak_day', 0)

            # Higher score if in peak periods
            month_score = 0.8 if month == peak_month else 0.5
            day_score = 0.7 if day_of_week == peak_day else 0.5

            # Combine
            seasonal_score = (month_score * 0.6) + (day_score * 0.4)

            return seasonal_score

        except:
            return 0.5

    def calculate_composite_risk(
        self,
        ordered_qty: float,
        delivered_qty: float,
        product_stats: Optional[Dict[str, Any]] = None,
        customer_stats: Optional[Dict[str, Any]] = None,
        forecast_data: Optional[Dict[str, Any]] = None,
        seasonal_patterns: Optional[Dict[str, Any]] = None,
        delivery_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate composite risk score from all factors.

        Args:
            ordered_qty: Quantity ordered
            delivered_qty: Quantity delivered
            product_stats: Historical product statistics
            customer_stats: Customer behavior statistics
            forecast_data: Forecast predictions
            seasonal_patterns: Seasonal pattern data
            delivery_date: Delivery date for seasonal analysis

        Returns:
            Dictionary with composite score and individual factor scores
        """
        # Calculate individual scores
        severity = self.calculate_severity_score(ordered_qty, delivered_qty)
        historical = self.calculate_historical_score(product_stats)
        customer = self.calculate_customer_score(customer_stats)
        forecast = self.calculate_forecast_score(forecast_data)
        seasonal = self.calculate_seasonal_score(
            delivery_date or datetime.now().isoformat(),
            seasonal_patterns
        )

        # Calculate weighted composite score
        composite = (
            severity * self.weights['severity'] +
            historical * self.weights['historical'] +
            customer * self.weights['customer'] +
            forecast * self.weights['forecast'] +
            seasonal * self.weights['seasonal']
        )

        return {
            'composite_risk_score': round(composite, 3),
            'severity_score': round(severity, 3),
            'historical_score': round(historical, 3),
            'customer_score': round(customer, 3),
            'forecast_score': round(forecast, 3),
            'seasonal_score': round(seasonal, 3),
            'risk_level': self._classify_risk_level(composite)
        }

    def _classify_risk_level(self, score: float) -> str:
        """
        Classify risk score into categories.

        Args:
            score: Composite risk score

        Returns:
            Risk level: critical, high, medium, low
        """
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'

    def batch_score(
        self,
        shortage_events: List[Dict[str, Any]],
        product_stats: Dict[str, Dict[str, Any]],
        customer_stats: Dict[str, Dict[str, Any]],
        forecast_data: Optional[Dict[str, Dict[str, Any]]] = None,
        seasonal_patterns: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Score multiple shortage events.

        Args:
            shortage_events: List of shortage events
            product_stats: Product statistics by SKU
            customer_stats: Customer statistics by ID
            forecast_data: Forecast data by product
            seasonal_patterns: Seasonal patterns

        Returns:
            Shortage events with risk scores added
        """
        scored_events = []

        for event in shortage_events:
            sku = event.get('sku')
            customer_id = event.get('customer_id')

            # Get relevant stats
            prod_stats = product_stats.get(sku)
            cust_stats = customer_stats.get(customer_id)
            forecast = forecast_data.get(sku) if forecast_data else None

            # Calculate risk scores
            risk_scores = self.calculate_composite_risk(
                ordered_qty=event.get('ordered_qty', 0),
                delivered_qty=event.get('delivered_qty', 0),
                product_stats=prod_stats,
                customer_stats=cust_stats,
                forecast_data=forecast,
                seasonal_patterns=seasonal_patterns,
                delivery_date=event.get('delivery_date')
            )

            # Add scores to event
            event_with_scores = event.copy()
            event_with_scores.update(risk_scores)

            scored_events.append(event_with_scores)

        # Sort by composite risk score
        scored_events.sort(key=lambda x: x['composite_risk_score'], reverse=True)

        return scored_events
