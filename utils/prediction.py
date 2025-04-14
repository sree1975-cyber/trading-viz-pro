import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_features(data, target_col='Close', lag_days=5):
    """
    Create technical features for models.
    Returns DataFrame with additional features.
    """
    df = data.copy()
    # Moving averages
    df['MA_5'] = df[target_col].rolling(window=5).mean()
    df['MA_20'] = df[target_col].rolling(window=20).mean()
    # Momentum
    df['Return_1'] = df[target_col].pct_change(periods=1)
    # Lagged features
    for lag in range(1, lag_days + 1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

def train_simple_ma_model(data, column='Close', forecast_periods=30, window=20):
    """Simple Moving Average model - FIXED naming"""
    try:
        last_value = data[column].iloc[-1]
        ma_value = data[column].rolling(window=window).mean().iloc[-1]
        ratio = last_value / ma_value if ma_value != 0 else 1.0
        
        forecast_values = [last_value * ratio for _ in range(forecast_periods)]
        dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        return {
            'predictions': pd.Series(forecast_values, index=dates),
            'model': 'Moving Average (Simple)',  # Exact UI name
            'window': window
        }
    except Exception as e:
        logger.error(f"MA model error: {str(e)}")
        return {
            'predictions': None,
            'model': 'Moving Average (Simple)',  # Consistent even in errors
            'error': str(e)
        }

def train_linear_trend_model(data, column='Close', forecast_periods=30):
    """Linear Trend model - FIXED naming"""
    try:
        lookback = min(30, len(data))
        y = data[column].values[-lookback:]
        x = np.arange(lookback)
        
        if len(y) > 1:
            m, b = np.polyfit(x, y, 1)
        else:
            m, b = 0, y[0]
        
        forecast = [m*(lookback+i) + b for i in range(1, forecast_periods+1)]
        dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        return {
            'predictions': pd.Series(forecast, index=dates),
            'model': 'Linear Trend (Simple)',  # Exact UI name
            'slope': m
        }
    except Exception as e:
        logger.error(f"Trend model error: {str(e)}")
        return {
            'predictions': None,
            'model': 'Linear Trend (Simple)',  # Consistent in errors
            'error': str(e)
        }

def get_price_predictions(data, forecast_periods=30, target_col='Close'):
    """
    Main prediction function - FULLY FIXED VERSION
    Returns predictions with guaranteed name matching
    """
    # Validate input
    if not isinstance(data, pd.DataFrame) or len(data) < 30:
        return {'error': 'Need at least 30 days of DataFrame data'}
    
    # Initialize results with correct model names
    results = {
        'Moving Average (Simple)': None,
        'Linear Trend (Simple)': None
    }
    
    # Get MA predictions
    try:
        ma_results = train_simple_ma_model(
            data, column=target_col,
            forecast_periods=forecast_periods
        )
        results['Moving Average (Simple)'] = ma_results
    except Exception as e:
        logger.error(f"MA prediction failed: {str(e)}")
        results['Moving Average (Simple)'] = {
            'predictions': None,
            'error': str(e),
            'model': 'Moving Average (Simple)'
        }
    
    # Get Trend predictions
    try:
        trend_results = train_linear_trend_model(
            data, column=target_col,
            forecast_periods=forecast_periods
        )
        results['Linear Trend (Simple)'] = trend_results
    except Exception as e:
        logger.error(f"Trend prediction failed: {str(e)}")
        results['Linear Trend (Simple)'] = {
            'predictions': None,
            'error': str(e),
            'model': 'Linear Trend (Simple)'
        }
    
    return results

def get_available_prediction_methods():
    """Returns exactly matching model names for UI"""
    return [
        "Moving Average (Simple)",
        "Linear Trend (Simple)"
    ]

def get_prediction_method_descriptions():
    """Returns descriptions with exact name matching"""
    return {
        "Moving Average (Simple)": (
            "Projects prices based on the relationship between "
            "current price and its moving average"
        ),
        "Linear Trend (Simple)": (
            "Fits a linear trend to recent prices and "
            "projects it into the future"
        )
    }
