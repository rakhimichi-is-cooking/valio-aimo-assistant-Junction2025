"""
Time-series forecasting engine for predicting product shortages.

Implements multiple forecasting algorithms:
- Prophet (Facebook's time-series forecasting)
- ARIMA (Auto-Regressive Integrated Moving Average)
- XGBoost (Gradient Boosting)
- LSTM/GRU (Neural network hybrid - MCDFN-inspired)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time-series libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Neural forecaster
try:
    from backend.neural_forecaster import LSTMGRUForecaster, train_and_forecast_neural
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False


class ForecastingEngine:
    """
    Multi-algorithm time-series forecasting engine.

    Supports:
    - Prophet: Best for seasonal patterns and holidays
    - ARIMA: Classical statistical approach
    - XGBoost: ML-based approach with feature engineering
    - LSTM: Neural network hybrid (MCDFN-inspired)
    """

    def __init__(self, method: str = "prophet"):
        """
        Initialize forecasting engine.

        Args:
            method: Forecasting method ("prophet", "arima", "xgboost", "lstm", "ensemble")
        """
        self.method = method.lower()
        self._validate_method()

    def _validate_method(self):
        """Validate that required libraries are available for the chosen method."""
        if self.method == "prophet" and not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        elif self.method == "arima" and not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels not installed. Run: pip install statsmodels")
        elif self.method == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost scikit-learn")
        elif self.method == "lstm" and not NEURAL_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_col: str = "requested_delivery_date",
        value_col: str = "order_qty",
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Prepare time-series data for forecasting.

        Args:
            df: Input dataframe
            date_col: Date column name
            value_col: Value to forecast
            freq: Frequency (D=daily, W=weekly, M=monthly)

        Returns:
            Prepared time-series dataframe
        """
        # Convert date column to datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Aggregate by date
        ts_data = df.groupby(date_col)[value_col].sum().reset_index()
        ts_data.columns = ['ds', 'y']  # Prophet naming convention

        # Sort by date
        ts_data = ts_data.sort_values('ds').reset_index(drop=True)

        # Fill missing dates
        date_range = pd.date_range(
            start=ts_data['ds'].min(),
            end=ts_data['ds'].max(),
            freq=freq
        )
        ts_data = ts_data.set_index('ds').reindex(date_range, fill_value=0).reset_index()
        ts_data.columns = ['ds', 'y']

        return ts_data

    def forecast_prophet(
        self,
        ts_data: pd.DataFrame,
        periods: int = 30,
        seasonality: bool = True
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Forecast using Facebook Prophet.

        Args:
            ts_data: Time-series data with 'ds' and 'y' columns
            periods: Number of periods to forecast
            seasonality: Enable seasonal components

        Returns:
            Tuple of (forecast dataframe, fitted model)
        """
        # Initialize Prophet model
        model = Prophet(
            daily_seasonality=seasonality,
            weekly_seasonality=seasonality,
            yearly_seasonality=seasonality,
            changepoint_prior_scale=0.05,  # Flexibility of trend
            seasonality_prior_scale=10.0,   # Strength of seasonality
        )

        # Fit model
        model.fit(ts_data)

        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)

        # Predict
        forecast = model.predict(future)

        return forecast, model

    def forecast_arima(
        self,
        ts_data: pd.DataFrame,
        periods: int = 30,
        order: Tuple[int, int, int] = (2, 1, 2)
    ) -> pd.DataFrame:
        """
        Forecast using ARIMA.

        Args:
            ts_data: Time-series data with 'ds' and 'y' columns
            periods: Number of periods to forecast
            order: ARIMA order (p, d, q)

        Returns:
            Forecast dataframe
        """
        # Prepare data
        y = ts_data['y'].values

        # Fit ARIMA model
        model = ARIMA(y, order=order)
        fitted = model.fit()

        # Forecast
        forecast_values = fitted.forecast(steps=periods)

        # Create forecast dataframe
        last_date = ts_data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': forecast_values * 0.9,  # Simple confidence interval
            'yhat_upper': forecast_values * 1.1
        })

        return forecast_df

    def forecast_xgboost(
        self,
        ts_data: pd.DataFrame,
        periods: int = 30,
        lags: int = 7
    ) -> pd.DataFrame:
        """
        Forecast using XGBoost with lag features.

        Args:
            ts_data: Time-series data with 'ds' and 'y' columns
            periods: Number of periods to forecast
            lags: Number of lag features to create

        Returns:
            Forecast dataframe
        """
        # Create lag features
        df = ts_data.copy()
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)

        # Add time-based features
        df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['ds']).dt.day
        df['month'] = pd.to_datetime(df['ds']).dt.month

        # Drop NaN rows
        df = df.dropna()

        # Split features and target
        feature_cols = [f'lag_{i}' for i in range(1, lags + 1)] + \
                      ['day_of_week', 'day_of_month', 'month']
        X = df[feature_cols].values
        y = df['y'].values

        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)

        # Recursive forecasting
        last_values = ts_data['y'].tail(lags).values.tolist()
        last_date = ts_data['ds'].max()

        forecast_values = []
        for i in range(periods):
            # Create features for next prediction
            current_date = last_date + timedelta(days=i + 1)
            features = last_values[-lags:] + [
                current_date.dayofweek,
                current_date.day,
                current_date.month
            ]

            # Predict
            pred = model.predict(np.array([features]))[0]
            forecast_values.append(max(0, pred))  # Ensure non-negative

            # Update last values
            last_values.append(pred)

        # Create forecast dataframe
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_values,
            'yhat_lower': np.array(forecast_values) * 0.85,
            'yhat_upper': np.array(forecast_values) * 1.15
        })

        return forecast_df

    def forecast(
        self,
        ts_data: pd.DataFrame,
        periods: int = 30,
        **kwargs
    ) -> pd.DataFrame:
        """
        Main forecast method that routes to appropriate algorithm.

        Args:
            ts_data: Time-series data with 'ds' and 'y' columns
            periods: Number of periods to forecast
            **kwargs: Method-specific parameters

        Returns:
            Forecast dataframe with columns: ds, yhat, yhat_lower, yhat_upper
        """
        if self.method == "prophet":
            forecast, _ = self.forecast_prophet(ts_data, periods, **kwargs)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        elif self.method == "arima":
            return self.forecast_arima(ts_data, periods, **kwargs)

        elif self.method == "xgboost":
            return self.forecast_xgboost(ts_data, periods, **kwargs)

        elif self.method == "lstm":
            # Neural LSTM/GRU forecasting
            return train_and_forecast_neural(ts_data, periods, **kwargs)

        elif self.method == "ensemble":
            # Ensemble of all methods
            forecasts = []

            if PROPHET_AVAILABLE:
                try:
                    prophet_fc, _ = self.forecast_prophet(ts_data, periods)
                    forecasts.append(prophet_fc[['ds', 'yhat']])
                except:
                    pass

            if STATSMODELS_AVAILABLE:
                try:
                    arima_fc = self.forecast_arima(ts_data, periods)
                    forecasts.append(arima_fc[['ds', 'yhat']])
                except:
                    pass

            if XGBOOST_AVAILABLE:
                try:
                    xgb_fc = self.forecast_xgboost(ts_data, periods)
                    forecasts.append(xgb_fc[['ds', 'yhat']])
                except:
                    pass

            if not forecasts:
                raise RuntimeError("No forecasting methods available")

            # Average predictions
            ensemble_df = forecasts[0].copy()
            for fc in forecasts[1:]:
                ensemble_df = ensemble_df.merge(fc, on='ds', suffixes=('', '_temp'))
                ensemble_df['yhat'] = (ensemble_df['yhat'] + ensemble_df['yhat_temp']) / 2
                ensemble_df = ensemble_df.drop('yhat_temp', axis=1)

            # Add confidence intervals
            ensemble_df['yhat_lower'] = ensemble_df['yhat'] * 0.85
            ensemble_df['yhat_upper'] = ensemble_df['yhat'] * 1.15

            return ensemble_df

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forecast_by_product(
        self,
        df: pd.DataFrame,
        product_col: str = "product_code",
        date_col: str = "requested_delivery_date",
        value_col: str = "order_qty",
        periods: int = 30,
        top_n: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Forecast demand for each product.

        Args:
            df: Sales/order data
            product_col: Product identifier column
            date_col: Date column
            value_col: Value to forecast
            periods: Forecast horizon
            top_n: Only forecast top N products by volume

        Returns:
            Dictionary mapping product_code -> forecast dataframe
        """
        # Find top N products by total volume
        top_products = df.groupby(product_col)[value_col].sum() \
            .nlargest(top_n).index.tolist()

        forecasts = {}

        for product in top_products:
            try:
                # Filter data for this product
                product_df = df[df[product_col] == product]

                # Prepare time series
                ts_data = self.prepare_time_series(
                    product_df,
                    date_col=date_col,
                    value_col=value_col
                )

                # Skip if insufficient data
                if len(ts_data) < 14:  # Need at least 2 weeks
                    continue

                # Forecast
                forecast = self.forecast(ts_data, periods=periods)
                forecasts[product] = forecast

            except Exception as e:
                # Skip products that fail to forecast
                print(f"Failed to forecast {product}: {e}")
                continue

        return forecasts
