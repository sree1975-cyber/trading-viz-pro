import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
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
    df['MA_50'] = df[target_col].rolling(window=50).mean()
    
    # Price momentum
    df['Return_1'] = df[target_col].pct_change(periods=1)
    df['Return_5'] = df[target_col].pct_change(periods=5)
    
    # Volatility
    df['Volatility_5'] = df[target_col].rolling(window=5).std()
    df['Volatility_20'] = df[target_col].rolling(window=20).std()
    
    # Price range indicators
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    
    # Add lagged variables
    for lag in range(1, lag_days + 1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def train_arima_model(data, column='Close', forecast_periods=30):
    """
    Train and forecast using an ARIMA model.
    
    Args:
        data (pd.DataFrame): OHLCV data
        column (str): Column to predict
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        dict: Dictionary containing predictions and model info
    """
    try:
        y = data[column].values
        
        logger.info("Finding optimal ARIMA parameters...")
        model = auto_arima(y, seasonal=False, suppress_warnings=True,
                           stepwise=True, max_p=5, max_q=5, max_d=2,
                           trace=False, error_action='ignore')
        
        order = model.order
        logger.info(f"Best ARIMA order: {order}")
        
        final_model = ARIMA(y, order=order)
        result = final_model.fit()
        
        forecast = result.forecast(steps=forecast_periods)
        
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecast, index=forecast_dates),
            'model': 'ARIMA',
            'order': order,
            'aic': result.aic,
        }
    except Exception as e:
        logger.error(f"Error in ARIMA model: {str(e)}")
        return {
            'predictions': None,
            'model': 'ARIMA',
            'error': str(e)
        }

def train_machine_learning_model(data, model_type='random_forest', forecast_periods=30, target_col='Close'):
    """
    Train and forecast using machine learning models.
    
    Args:
        data (pd.DataFrame): OHLCV data
        model_type (str): Type of ML model ('linear', 'ridge', 'lasso', 'random_forest', 'svr')
        forecast_periods (int): Number of periods to forecast
        target_col (str): Column to predict
        
    Returns:
        dict: Dictionary containing predictions and model info
    """
    try:
        df = create_features(data, target_col=target_col)
        
        y = df[target_col].values
        X = df.drop([target_col], axis=1)
        
        feature_names = X.columns
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.1)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        
        y_pred_actual = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        forecast_values = []
        
        last_known_values = X_scaled[-1:].copy()
        
        for _ in range(forecast_periods):
            next_pred = model.predict(last_known_values)
            forecast_values.append(next_pred[0])
            
            new_point = last_known_values[0].copy()
            
            lag_columns = [i for i, name in enumerate(feature_names) if target_col + '_Lag_' in name]
            lag_columns.sort(reverse=True)
            
            for i in range(len(lag_columns) - 1):
                new_point[lag_columns[i+1]] = new_point[lag_columns[i]]
            
            if lag_columns:
                new_point[lag_columns[0]] = next_pred[0]
                
            last_known_values = np.array([new_point])
        
        forecast_values = scaler_y.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()
        
        last_date = data.index[-1]
        
