"""
Neural LSTM/GRU Demand Forecasting Engine
MCDFN-inspired hybrid architecture for supply chain forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os

try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMGRUForecaster:
    """
    Hybrid LSTM+GRU neural forecaster inspired by MCDFN architecture.

    Architecture:
    - Input: Time-series sequences with lag features
    - LSTM layer: Captures long-term dependencies
    - GRU layer: Captures short-term patterns
    - Dense layer: Final prediction
    """

    def __init__(
        self,
        sequence_length: int = 14,
        lstm_units: int = 64,
        gru_units: int = 32,
        dropout_rate: float = 0.2,
        use_differencing: bool = True
    ):
        """
        Initialize neural forecaster.

        Args:
            sequence_length: Number of historical timesteps
            lstm_units: LSTM hidden units
            gru_units: GRU hidden units
            dropout_rate: Dropout for regularization
            use_differencing: Use first-order differencing to remove trends
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required. Run: pip install tensorflow")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.use_differencing = use_differencing
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.last_value = None  # Store last value for inverse differencing

    def build_model(self, n_features: int = 1):
        """
        Build hybrid LSTM+GRU model.

        Args:
            n_features: Number of input features per timestep
        """
        model = keras.Sequential([
            # LSTM layer for long-term patterns
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features)
            ),
            layers.Dropout(self.dropout_rate),

            # GRU layer for short-term patterns
            layers.GRU(self.gru_units, return_sequences=False),
            layers.Dropout(self.dropout_rate),

            # Output layer
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def create_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create supervised learning sequences from time-series.

        Args:
            data: 1D array of time-series values

        Returns:
            X, y arrays for training
        """
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)

        return X, y

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score."""
        self.scaler_mean = np.mean(data)
        self.scaler_std = np.std(data) + 1e-8  # Avoid division by zero
        return (data - self.scaler_mean) / self.scaler_std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.scaler_mean is None or self.scaler_std is None:
            return data
        return (data * self.scaler_std) + self.scaler_mean

    def apply_differencing(self, data: np.ndarray) -> np.ndarray:
        """
        Apply first-order differencing to remove trends.

        Y_t = X_t - X_{t-1}

        Args:
            data: Original time series values

        Returns:
            Differenced series (length = len(data) - 1)
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for differencing")

        # Store last value for inverse transform
        self.last_value = float(data[-1])

        # Compute differences: Y_t = X_t - X_{t-1}
        diff = np.diff(data)
        return diff

    def inverse_differencing(self, diff_predictions: np.ndarray) -> np.ndarray:
        """
        Inverse first-order differencing to recover original scale.

        X_t = X_{t-1} + Y_t

        Args:
            diff_predictions: Predicted differences

        Returns:
            Predictions in original scale
        """
        if self.last_value is None:
            raise ValueError("No last value stored. Must train with differencing first.")

        # Cumulative sum starting from last known value
        # X_t = X_{t-1} + diff_t
        predictions = np.zeros(len(diff_predictions))
        prev_value = self.last_value

        for i, diff_val in enumerate(diff_predictions):
            current_value = prev_value + diff_val
            predictions[i] = current_value
            prev_value = current_value

        return predictions

    def train(
        self,
        ts_data: pd.DataFrame,
        value_col: str = 'y',
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2
    ) -> dict:
        """
        Train the neural forecaster.

        Args:
            ts_data: DataFrame with time-series data
            value_col: Column name with values to forecast
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio

        Returns:
            Training history
        """
        # Extract values
        values = ts_data[value_col].values.astype(np.float32)

        # Apply differencing if enabled (removes trend)
        if self.use_differencing:
            values = self.apply_differencing(values)

        # Normalize (now on stationary differences if differencing enabled)
        values_norm = self.normalize(values)

        # Create sequences
        X, y = self.create_sequences(values_norm)

        if len(X) < 10:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length + 10} points")

        # Build model if not exists
        if self.model is None:
            self.build_model(n_features=1)

        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )

        return {
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
            'mae': float(history.history['mae'][-1])
        }

    def forecast(
        self,
        ts_data: pd.DataFrame,
        periods: int = 30,
        value_col: str = 'y',
        num_samples: int = 100
    ) -> pd.DataFrame:
        """
        Generate forecast for future periods with proper uncertainty quantification.

        Uses Monte Carlo dropout for real confidence intervals based on model uncertainty.

        Args:
            ts_data: Historical time-series data
            periods: Number of periods to forecast
            value_col: Column with values
            num_samples: Number of Monte Carlo samples for uncertainty estimation

        Returns:
            DataFrame with forecasted values and real confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first")

        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Model not trained with normalization parameters")

        # Extract values
        values = ts_data[value_col].values.astype(np.float32)

        # Apply differencing if enabled (to match training preprocessing)
        if self.use_differencing:
            values_diff = self.apply_differencing(values)
            # Normalize differences using TRAINING scalers
            values_norm = (values_diff - self.scaler_mean) / self.scaler_std
        else:
            # Normalize raw values using TRAINING scalers
            values_norm = (values - self.scaler_mean) / self.scaler_std

        # Monte Carlo dropout for uncertainty estimation
        all_predictions = []

        for _ in range(num_samples):
            # Recursive forecasting with dropout enabled
            last_sequence = values_norm[-self.sequence_length:].tolist()
            predictions = []

            for _ in range(periods):
                # Prepare input
                x_input = np.array(last_sequence[-self.sequence_length:]).reshape(1, self.sequence_length, 1)

                # Predict with dropout enabled (training=True for Monte Carlo)
                y_pred = self.model(x_input, training=True).numpy()[0, 0]
                predictions.append(y_pred)

                # Update sequence
                last_sequence.append(y_pred)

            all_predictions.append(predictions)

        # Convert to array: shape (num_samples, periods)
        all_predictions = np.array(all_predictions)

        # Denormalize all predictions (get back to difference scale or raw scale)
        all_predictions_denorm = self.denormalize(all_predictions)

        # If using differencing, apply inverse differencing to convert back to original scale
        if self.use_differencing:
            # Apply inverse differencing to each Monte Carlo sample
            all_predictions_original = np.zeros_like(all_predictions_denorm)
            for i in range(num_samples):
                all_predictions_original[i] = self.inverse_differencing(all_predictions_denorm[i])
            all_predictions_denorm = all_predictions_original

        # Calculate statistics
        mean_predictions = np.mean(all_predictions_denorm, axis=0)
        lower_bound = np.percentile(all_predictions_denorm, 2.5, axis=0)  # 95% CI
        upper_bound = np.percentile(all_predictions_denorm, 97.5, axis=0)

        # Create forecast DataFrame
        last_date = pd.to_datetime(ts_data['ds'].max())
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )

        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': mean_predictions,
            'yhat_lower': lower_bound,
            'yhat_upper': upper_bound
        })

        return forecast_df

    def save_model(self, filepath: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        # Save model
        self.model.save(filepath)

        # Save scaler params and differencing state
        scaler_path = filepath.replace('.h5', '_scaler.npz')
        np.savez(
            scaler_path,
            mean=self.scaler_mean,
            std=self.scaler_std,
            sequence_length=self.sequence_length,
            use_differencing=self.use_differencing,
            last_value=self.last_value if self.last_value is not None else 0.0
        )

    def load_model(self, filepath: str):
        """Load trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        # Load model
        self.model = keras.models.load_model(filepath)

        # Load scaler params and differencing state
        scaler_path = filepath.replace('.h5', '_scaler.npz')
        if os.path.exists(scaler_path):
            scaler_data = np.load(scaler_path)
            self.scaler_mean = scaler_data['mean']
            self.scaler_std = scaler_data['std']
            self.sequence_length = int(scaler_data['sequence_length'])
            self.use_differencing = bool(scaler_data.get('use_differencing', False))
            last_val = scaler_data.get('last_value', None)
            self.last_value = float(last_val) if last_val is not None else None


def train_and_forecast_neural(
    ts_data: pd.DataFrame,
    periods: int = 30,
    sequence_length: int = 14,
    epochs: int = 30,
    num_samples: int = 100,
    use_differencing: bool = True
) -> pd.DataFrame:
    """
    Quick helper to train and forecast with neural model.

    Args:
        ts_data: Time-series with 'ds' and 'y' columns
        periods: Forecast horizon
        sequence_length: Lookback window
        epochs: Training epochs
        num_samples: Monte Carlo samples for confidence intervals
        use_differencing: Use first-order differencing to remove trends

    Returns:
        Forecast DataFrame with real confidence intervals
    """
    forecaster = LSTMGRUForecaster(
        sequence_length=sequence_length,
        use_differencing=use_differencing
    )

    # Train
    forecaster.train(ts_data, epochs=epochs)

    # Forecast with proper uncertainty quantification
    forecast = forecaster.forecast(ts_data, periods=periods, num_samples=num_samples)

    return forecast
