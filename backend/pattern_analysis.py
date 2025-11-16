"""
Historical pattern analysis for shortage prediction.

Analyzes:
- Shortage frequency patterns
- Seasonal trends
- Customer behavior patterns
- Product delivery reliability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class PatternAnalyzer:
    """
    Analyzes historical patterns in order fulfillment and shortages.
    """

    def __init__(self):
        """Initialize pattern analyzer."""
        self.patterns = {}

    def analyze_shortage_frequency(
        self,
        df: pd.DataFrame,
        product_col: str = "product_code",
        date_col: str = "requested_delivery_date",
        order_col: str = "order_qty",
        delivered_col: str = "delivered_qty",
        window_days: int = 90
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how frequently each product experiences shortages.

        Args:
            df: Sales/delivery data
            product_col: Product identifier column
            date_col: Date column
            order_col: Ordered quantity column
            delivered_col: Delivered quantity column
            window_days: Analysis window in days

        Returns:
            Dictionary mapping product -> shortage statistics
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter to recent history
        cutoff_date = df[date_col].max() - timedelta(days=window_days)
        recent_df = df[df[date_col] >= cutoff_date]

        # Calculate shortage events
        recent_df['shortage'] = (
            recent_df[order_col] > recent_df[delivered_col]
        ).astype(int)
        recent_df['shortage_qty'] = (
            recent_df[order_col] - recent_df[delivered_col]
        ).clip(lower=0)

        # Aggregate by product
        product_stats = {}

        for product in recent_df[product_col].unique():
            product_df = recent_df[recent_df[product_col] == product]

            total_orders = len(product_df)
            shortage_count = product_df['shortage'].sum()
            shortage_rate = shortage_count / total_orders if total_orders > 0 else 0

            avg_shortage_qty = product_df[product_df['shortage'] == 1]['shortage_qty'].mean()
            avg_shortage_qty = 0 if pd.isna(avg_shortage_qty) else avg_shortage_qty

            # Calculate severity (percentage of order not fulfilled)
            product_df['shortage_pct'] = (
                product_df['shortage_qty'] / product_df[order_col]
            ).fillna(0)
            avg_severity = product_df[product_df['shortage'] == 1]['shortage_pct'].mean()
            avg_severity = 0 if pd.isna(avg_severity) else avg_severity

            product_stats[product] = {
                'total_orders': total_orders,
                'shortage_count': int(shortage_count),
                'shortage_rate': float(shortage_rate),
                'avg_shortage_qty': float(avg_shortage_qty),
                'avg_shortage_severity': float(avg_severity),
                'reliability_score': float(1 - shortage_rate)
            }

        return product_stats

    def analyze_seasonal_patterns(
        self,
        df: pd.DataFrame,
        date_col: str = "requested_delivery_date",
        value_col: str = "order_qty",
        product_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns in demand.

        Args:
            df: Sales/order data
            date_col: Date column
            value_col: Value to analyze
            product_col: Optional product column for per-product analysis

        Returns:
            Seasonal pattern statistics
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Extract time features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter

        patterns = {}

        # Weekly patterns
        weekly = df.groupby('day_of_week')[value_col].agg(['mean', 'std', 'count'])
        patterns['weekly'] = {
            'data': weekly.to_dict('index'),
            'peak_day': int(weekly['mean'].idxmax()),
            'low_day': int(weekly['mean'].idxmin()),
            'variation_coefficient': float(weekly['mean'].std() / weekly['mean'].mean())
        }

        # Monthly patterns
        monthly = df.groupby('month')[value_col].agg(['mean', 'std', 'count'])
        patterns['monthly'] = {
            'data': monthly.to_dict('index'),
            'peak_month': int(monthly['mean'].idxmax()),
            'low_month': int(monthly['mean'].idxmin()),
            'variation_coefficient': float(monthly['mean'].std() / monthly['mean'].mean())
        }

        # Quarterly patterns
        quarterly = df.groupby('quarter')[value_col].agg(['mean', 'std', 'count'])
        patterns['quarterly'] = {
            'data': quarterly.to_dict('index'),
            'peak_quarter': int(quarterly['mean'].idxmax()),
            'low_quarter': int(quarterly['mean'].idxmin())
        }

        # Product-specific patterns (if requested)
        if product_col:
            product_patterns = {}
            for product in df[product_col].unique():
                product_df = df[df[product_col] == product]
                if len(product_df) >= 30:  # Need sufficient data
                    product_monthly = product_df.groupby('month')[value_col].mean()
                    product_patterns[product] = {
                        'peak_month': int(product_monthly.idxmax()),
                        'low_month': int(product_monthly.idxmin()),
                        'seasonality_strength': float(
                            product_monthly.std() / product_monthly.mean()
                        )
                    }
            patterns['by_product'] = product_patterns

        return patterns

    def analyze_customer_patterns(
        self,
        df: pd.DataFrame,
        customer_col: str = "customer_number",
        product_col: str = "product_code",
        date_col: str = "requested_delivery_date",
        order_col: str = "order_qty",
        delivered_col: str = "delivered_qty"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze customer ordering and shortage patterns.

        Args:
            df: Sales/delivery data
            customer_col: Customer identifier column
            product_col: Product identifier column
            date_col: Date column
            order_col: Ordered quantity column
            delivered_col: Delivered quantity column

        Returns:
            Dictionary mapping customer -> behavior statistics
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        customer_stats = {}

        for customer in df[customer_col].unique():
            customer_df = df[df[customer_col] == customer]

            # Calculate shortage impact
            customer_df['shortage'] = (
                customer_df[order_col] > customer_df[delivered_col]
            ).astype(int)

            total_orders = len(customer_df)
            shortage_count = customer_df['shortage'].sum()
            unique_products = customer_df[product_col].nunique()

            # Order frequency
            date_range = (customer_df[date_col].max() - customer_df[date_col].min()).days
            order_frequency = total_orders / max(date_range, 1) if date_range > 0 else 0

            # Average order size
            avg_order_size = customer_df[order_col].mean()

            # Calculate customer importance score
            total_volume = customer_df[order_col].sum()
            shortage_impact = shortage_count * avg_order_size

            customer_stats[customer] = {
                'total_orders': int(total_orders),
                'shortage_count': int(shortage_count),
                'shortage_rate': float(shortage_count / total_orders if total_orders > 0 else 0),
                'unique_products': int(unique_products),
                'order_frequency_per_day': float(order_frequency),
                'avg_order_size': float(avg_order_size),
                'total_volume': float(total_volume),
                'shortage_impact': float(shortage_impact),
                'importance_score': float(total_volume * order_frequency)
            }

        return customer_stats

    def detect_trends(
        self,
        df: pd.DataFrame,
        date_col: str = "requested_delivery_date",
        value_col: str = "order_qty",
        window: int = 7
    ) -> Dict[str, Any]:
        """
        Detect trends in demand over time.

        Args:
            df: Sales/order data
            date_col: Date column
            value_col: Value to analyze
            window: Moving average window

        Returns:
            Trend analysis results
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Aggregate by date
        daily = df.groupby(date_col)[value_col].sum().reset_index()
        daily = daily.sort_values(date_col)

        # Calculate moving average
        daily['ma'] = daily[value_col].rolling(window=window, min_periods=1).mean()

        # Calculate trend direction
        recent_ma = daily['ma'].tail(window).mean()
        older_ma = daily['ma'].head(window).mean()

        if recent_ma > older_ma * 1.1:
            trend = "increasing"
        elif recent_ma < older_ma * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        # Calculate volatility
        daily['daily_change'] = daily[value_col].pct_change()
        volatility = daily['daily_change'].std()

        return {
            'trend': trend,
            'recent_avg': float(recent_ma),
            'historical_avg': float(older_ma),
            'change_pct': float((recent_ma - older_ma) / older_ma * 100) if older_ma > 0 else 0,
            'volatility': float(volatility),
            'data_points': len(daily)
        }

    def identify_high_risk_combinations(
        self,
        df: pd.DataFrame,
        product_col: str = "product_code",
        customer_col: str = "customer_number",
        date_col: str = "requested_delivery_date",
        order_col: str = "order_qty",
        delivered_col: str = "delivered_qty",
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Identify product-customer combinations with high shortage risk.

        Args:
            df: Sales/delivery data
            product_col: Product identifier column
            customer_col: Customer identifier column
            date_col: Date column
            order_col: Ordered quantity column
            delivered_col: Delivered quantity column
            threshold: Shortage rate threshold

        Returns:
            List of high-risk combinations
        """
        df = df.copy()
        df['shortage'] = (df[order_col] > df[delivered_col]).astype(int)

        # Group by product-customer combination
        grouped = df.groupby([product_col, customer_col]).agg({
            'shortage': ['sum', 'count'],
            order_col: 'sum'
        }).reset_index()

        grouped.columns = [product_col, customer_col, 'shortage_count', 'total_orders', 'total_volume']

        # Calculate shortage rate
        grouped['shortage_rate'] = grouped['shortage_count'] / grouped['total_orders']

        # Filter high-risk combinations
        high_risk = grouped[
            (grouped['shortage_rate'] >= threshold) &
            (grouped['total_orders'] >= 5)  # Minimum sample size
        ].sort_values('shortage_rate', ascending=False)

        results = []
        for _, row in high_risk.iterrows():
            results.append({
                'product': row[product_col],
                'customer': row[customer_col],
                'shortage_rate': float(row['shortage_rate']),
                'shortage_count': int(row['shortage_count']),
                'total_orders': int(row['total_orders']),
                'total_volume': float(row['total_volume']),
                'risk_level': 'high' if row['shortage_rate'] > 0.5 else 'medium'
            })

        return results

    def generate_insights(
        self,
        df: pd.DataFrame,
        product_col: str = "product_code",
        customer_col: str = "customer_number",
        date_col: str = "requested_delivery_date",
        order_col: str = "order_qty",
        delivered_col: str = "delivered_qty"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights from historical data.

        Args:
            df: Sales/delivery data
            All other columns specify field names

        Returns:
            Comprehensive insights dictionary
        """
        insights = {
            'shortage_frequency': self.analyze_shortage_frequency(
                df, product_col, date_col, order_col, delivered_col
            ),
            'seasonal_patterns': self.analyze_seasonal_patterns(
                df, date_col, order_col, product_col
            ),
            'customer_patterns': self.analyze_customer_patterns(
                df, customer_col, product_col, date_col, order_col, delivered_col
            ),
            'trends': self.detect_trends(df, date_col, order_col),
            'high_risk_combinations': self.identify_high_risk_combinations(
                df, product_col, customer_col, date_col, order_col, delivered_col
            )
        }

        return insights
