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
    Create technical features for machine learning models.
    """
    df = data.copy()
    
    # Technical indicators
    df['MA_5'] = df[target_col].rolling(5).mean()
    df['MA_20'] = df[target_col].rolling(20).mean()
    df['MA_50'] = df[target_col].rolling(50).mean()
    df['Return_1'] = df[target_col].pct_change(1)
    df['Return_5'] = df[target_col].pct_change(5)
    df['Volatility_5'] = df[target_col].rolling(5).std()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Lag features
    for lag in range(1, lag_days+1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    
    df.dropna(inplace=True)
    return df

def train_simple_ma_model(data, column='Close', forecast_periods=30, window=20):
    """Simple Moving Average model"""
    try:
        last_price = data[column].iloc[-1]
        ma = data[column].rolling(window).mean().iloc[-1]
        ratio = last_price / ma if ma != 0 else 1.0
        
        forecast = [last_price * (ratio ** (i+1)) for i in range(forecast_periods)]
        dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecast, index=dates),
            'model': 'Simple MA',
            'window': window
        }
    except Exception as e:
        logger.error(f"MA Error: {str(e)}")
        return {'predictions': None, 'error': str(e)}

def train_linear_trend_model(data, column='Close', forecast_periods=30):
    """Linear Trend Projection model"""
    try:
        lookback = min(30, len(data))
        y = data[column].values[-lookback:]
        x = np.arange(lookback)
        
        if len(y) > 1:
            m, b = np.polyfit(x, y, 1)
        else:
            m, b = 0, y[0]
        
        forecast = [m*(lookback+i) + b for i in range(1, forecast_periods+1)]
        dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecast, index=dates),
            'model': 'Linear Trend',
            'slope': m
        }
    except Exception as e:
        logger.error(f"Trend Error: {str(e)}")
        return {'predictions': None, 'error': str(e)}

def train_arima_model(data, column='Close', forecast_periods=30):
    """ARIMA Time Series Model"""
    try:
        y = data[column].values
        model = auto_arima(y, seasonal=False, suppress_warnings=True,
                          stepwise=True, max_p=3, max_q=3, max_d=1)
        
        arima = ARIMA(y, order=model.order).fit()
        forecast = arima.forecast(steps=forecast_periods)
        dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecast, index=dates),
            'model': 'ARIMA',
            'order': model.order
        }
    except Exception as e:
        logger.error(f"ARIMA Error: {str(e)}")
        return {'predictions': None, 'error': str(e)}

def train_machine_learning_model(data, model_type='random_forest', 
                               forecast_periods=30, target_col='Close'):
    """Machine Learning Models"""
    try:
        df = create_features(data, target_col)
        y = df[target_col].values
        X = df.drop(target_col, axis=1)
        
        # Scale data
        scaler_X = MinMaxScaler().fit(X)
        scaler_y = MinMaxScaler().fit(y.reshape(-1,1))
        X_s = scaler_X.transform(X)
        y_s = scaler_y.transform(y.reshape(-1,1)).ravel()
        
        # Model selection
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100),
            'svr': SVR(kernel='rbf', C=100)
        }
        model = models[model_type]
        model.fit(X_s[:-forecast_periods], y_s[:-forecast_periods])
        
        # Forecast
        current_features = X_s[-1].copy()
        forecasts = []
        for _ in range(forecast_periods):
            pred = model.predict([current_features])[0]
            forecasts.append(pred)
            # Update lag features
            current_features = np.roll(current_features, 1)
            current_features[0] = pred
        
        # Inverse transform
        forecasts = scaler_y.inverse_transform(
            np.array(forecasts).reshape(-1,1)).ravel()
        dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), 
                            periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecasts, index=dates),
            'model': model_type,
            'error': None
        }
    except Exception as e:
        logger.error(f"ML Error: {str(e)}")
        return {'predictions': None, 'error': str(e)}

def train_lstm_model(data, forecast_periods=30, target_col='Close', window=60):
    """LSTM Neural Network Model"""
    try:
        # Data preparation
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data[[target_col]])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled)-window):
            X.append(scaled[i:i+window, 0])
            y.append(scaled[i+window, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window,1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Forecast
        forecast_seq = scaled[-window:]
        predictions = []
        for _ in range(forecast_periods):
            pred = model.predict(forecast_seq.reshape(1,window,1), verbose=0)[0,0]
            predictions.append(pred)
            forecast_seq = np.append(forecast_seq[1:], pred)
        
        # Inverse transform
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1,1)).ravel()
        dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), 
                            periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(predictions, index=dates),
            'model': 'LSTM',
            'error': None
        }
    except Exception as e:
        logger.error(f"LSTM Error: {str(e)}")
        return {'predictions': None, 'error': str(e)}

def get_price_predictions(data, forecast_periods=30, target_col='Close'):
    """Main prediction function combining all models"""
    results = {}
    
    # Simple models
    results['simple_ma'] = train_simple_ma_model(data, target_col, forecast_periods)
    results['linear_trend'] = train_linear_trend_model(data, target_col, forecast_periods)
    
    # Statistical models
    if len(data) >= 100:
        results['arima'] = train_arima_model(data, target_col, forecast_periods)
    
    # ML models
    ml_models = ['random_forest', 'linear', 'svr']
    for model in ml_models:
        results[model] = train_machine_learning_model(data, model, forecast_periods, target_col)
    
    # Deep learning
    if len(data) >= 200:
        results['lstm'] = train_lstm_model(data, forecast_periods, target_col)
    
    return results

def get_available_prediction_methods():
    return [
        "Simple MA (Technical)",
        "Linear Trend (Technical)",
        "ARIMA (Statistical)",
        "Random Forest (ML)", 
        "Linear Regression (ML)",
        "SVR (ML)",
        "LSTM (DL)"
    ]

def get_prediction_method_descriptions():
    return {
        "Simple MA (Technical)": "Projects prices based on moving average relationships",
        "Linear Trend (Technical)": "Extrapolates recent price trends using linear regression",
        "ARIMA (Statistical)": "Auto-regressive integrated moving average model for time series",
        "Random Forest (ML)": "Ensemble of decision trees capturing non-linear patterns",
        "Linear Regression (ML)": "Linear model for simple trend analysis",
        "SVR (ML)": "Support vector machine for complex regression patterns",
        "LSTM (DL)": "Long short-term memory neural network for sequential data"
    }
