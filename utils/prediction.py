import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_features(data, target_col='Close', lag_days=5):
    """
    Create additional features for machine learning models.
    
    Args:
        data (pd.DataFrame): OHLCV data
        target_col (str): Column to predict
        lag_days (int): Number of lagged days to include as features
        
    Returns:
        pd.DataFrame: DataFrame with features
    """
    df = data.copy()
    
    # Technical indicators as features
    df['MA_5'] = df[target_col].rolling(window=5).mean()
    df['MA_20'] = df[target_col].rolling(window=20).mean()
    
    # Price momentum
    df['Return_1'] = df[target_col].pct_change(periods=1)
    
    # Add lagged variables
    for lag in range(1, lag_days + 1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def train_simple_ma_model(data, column='Close', forecast_periods=30, window=20):
    """
    Train and forecast using a simple moving average model.
    
    Args:
        data (pd.DataFrame): OHLCV data
        column (str): Column to predict
        forecast_periods (int): Number of periods to forecast
        window (int): Moving average window size
        
    Returns:
        dict: Dictionary containing predictions and model info
    """
    try:
        # Calculate the moving average
        last_value = data[column].iloc[-1]
        ma_value = data[column].rolling(window=window).mean().iloc[-1]
        
        # Simple projection - keep the same percentage difference from MA
        if ma_value != 0:
            ratio = last_value / ma_value
        else:
            ratio = 1.0
            
        # Create forecast series
        forecast_values = []
        for i in range(forecast_periods):
            forecast_values.append(last_value * ratio)
        
        # Create dates for the forecast period
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        # Return predictions and model info
        return {
            'predictions': pd.Series(forecast_values, index=forecast_dates),
            'model': 'Simple MA',
            'window': window
        }
    except Exception as e:
        logger.error(f"Error in Simple MA model: {str(e)}")
        return {
            'predictions': None,
            'model': 'Simple MA',
            'error': str(e)
        }

def train_linear_trend_model(data, column='Close', forecast_periods=30):
    """
    Train and forecast using a simple linear trend model.
    
    Args:
        data (pd.DataFrame): OHLCV data
        column (str): Column to predict
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        dict: Dictionary containing predictions and model info
    """
    try:
        # Use only the last 30 days for trend calculation
        last_n_days = min(30, len(data))
        values = data[column].values[-last_n_days:]
        x = np.arange(last_n_days)
        
        # Calculate a linear trend (y = mx + b)
        if len(values) > 1:
            m, b = np.polyfit(x, values, 1)
        else:
            m, b = 0, values[0]
        
        # Project the trend forward
        forecast_values = []
        for i in range(forecast_periods):
            next_x = last_n_days + i
            forecast_values.append(m * next_x + b)
        
        # Create dates for the forecast period
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        # Return predictions and model info
        return {
            'predictions': pd.Series(forecast_values, index=forecast_dates),
            'model': 'Linear Trend',
            'slope': m
        }
    except Exception as e:
        logger.error(f"Error in Linear Trend model: {str(e)}")
        return {
            'predictions': None,
            'model': 'Linear Trend',
            'error': str(e)
        }

def get_price_predictions(data, forecast_periods=30, target_col='Close'):
    """
    Generate price predictions using multiple simple models.
    
    Args:
        data (pd.DataFrame): OHLCV data
        forecast_periods (int): Number of periods to forecast
        target_col (str): Column to predict
        
    Returns:
        dict: Dictionary with predictions from various models
    """
    results = {}
    
    # Ensure we have enough data for forecasting
    if len(data) < 30:
        return {
            'error': 'Not enough data points for reliable forecasting. Please use a longer timeframe.'
        }
    
    # 1. Simple Moving Average Prediction
    ma_results = train_simple_ma_model(data, column=target_col, forecast_periods=forecast_periods)
    results['ma'] = ma_results
    
    # 2. Linear Trend Prediction
    trend_results = train_linear_trend_model(data, column=target_col, forecast_periods=forecast_periods)
    results['trend'] = trend_results
    
    return results

def get_available_prediction_methods():
    """
    Returns the list of available prediction methods.
    
    Returns:
        list: Names of prediction methods
    """
    return [
        "Moving Average (Simple)",
        "Linear Trend (Simple)"
    ]

def get_prediction_method_descriptions():
    """
    Returns descriptions of the prediction methods.
    
    Returns:
        dict: Descriptions of prediction methods
    """
    return {
        "Moving Average (Simple)": "A simple model that projects future prices based on the relationship between current price and its moving average.",
        "Linear Trend (Simple)": "A model that fits a linear trend to recent price data and projects that trend into the future."
    }
