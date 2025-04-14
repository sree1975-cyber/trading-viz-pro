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
    """
    df = data.copy()
    df['MA_5'] = df[target_col].rolling(window=5).mean()
    df['MA_20'] = df[target_col].rolling(window=20).mean()
    df['MA_50'] = df[target_col].rolling(window=50).mean()
    df['Return_1'] = df[target_col].pct_change(periods=1)
    df['Return_5'] = df[target_col].pct_change(periods=5)
    df['Volatility_5'] = df[target_col].rolling(window=5).std()
    df['Volatility_20'] = df[target_col].rolling(window=20).std()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    
    for lag in range(1, lag_days + 1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    
    df.dropna(inplace=True)
    return df

def train_arima_model(data, column='Close', forecast_periods=30):
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
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(forecast, index=forecast_dates),
            'model': 'ARIMA',
            'order': order,
            'aic': result.aic,
        }
    except Exception as e:
        logger.error(f"ARIMA Error: {str(e)}")
        return {'predictions': None, 'model': 'ARIMA', 'error': str(e)}

def train_machine_learning_model(data, model_type='random_forest', 
                               forecast_periods=30, target_col='Close'):
    try:
        df = create_features(data, target_col=target_col)
        y = df[target_col].values
        X = df.drop([target_col], axis=1)
        feature_names = X.columns
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, 
                                                          test_size=0.2, shuffle=False)
        
        model_map = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        }
        
        model = model_map.get(model_type)
        if not model:
            raise ValueError(f"Invalid model type: {model_type}")
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        forecast_values = []
        last_known = X_scaled[-1:].copy()
        
        for _ in range(forecast_periods):
            next_pred = model.predict(last_known)
            forecast_values.append(next_pred[0])
            new_point = last_known[0].copy()
            
            lag_cols = [i for i, name in enumerate(feature_names) 
                       if target_col + '_Lag_' in name]
            lag_cols.sort(reverse=True)
            
            for i in range(len(lag_cols) - 1):
                new_point[lag_cols[i+1]] = new_point[lag_cols[i]]
            
            if lag_cols:
                new_point[lag_cols[0]] = next_pred[0]
                
            last_known = np.array([new_point])
        
        forecast_values = scaler_y.inverse_transform(
            np.array(forecast_values).reshape(-1, 1)).flatten()
        
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecast_values, index=forecast_dates),
            'model': model_type,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    except Exception as e:
        logger.error(f"ML Error: {str(e)}")
        return {'predictions': None, 'model': model_type, 'error': str(e)}

def train_lstm_model(data, forecast_periods=30, target_col='Close', window_size=60):
    """
    LSTM Neural Network implementation (full corrected version)
    """
    try:
        values = data[target_col].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
        X, y = [], []
        for i in range(len(scaled) - window_size):
            X.append(scaled[i:i+window_size, 0])
            y.append(scaled[i+window_size, 0])
            
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, 
                validation_data=(X_test, y_test), verbose=0)
        
        forecast_seq = scaled[-window_size:].reshape(1, window_size, 1)
        forecast_values = []
        
        for _ in range(forecast_periods):
            pred = model.predict(forecast_seq, verbose=0)
            forecast_values.append(pred[0, 0])
            forecast_seq = np.append(forecast_seq[:, 1:, :], 
                                   pred.reshape(1, 1, 1), axis=1)
            
        forecast_values = scaler.inverse_transform(
            np.array(forecast_values).reshape(-1, 1)).flatten()
        
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=forecast_periods, freq='D')
        
        return {
            'predictions': pd.Series(forecast_values, index=forecast_dates),
            'model': 'LSTM',
            'mse': mean_squared_error(y_test, model.predict(X_test, verbose=0))
        }
    except Exception as e:
        logger.error(f"LSTM Error: {str(e)}")
        return {'predictions': None, 'model': 'LSTM', 'error': str(e)}

def get_price_predictions(data, forecast_periods=30, target_col='Close'):
    results = {}
    
    if len(data) < 100:
        return {'error': 'Minimum 100 data points required'}
    
    # ARIMA
    results['arima'] = train_arima_model(data, target_col, forecast_periods)
    
    # ML Models
    for model in ['random_forest', 'linear', 'svr']:
        results[model] = train_machine_learning_model(
            data, model, forecast_periods, target_col)
    
    # LSTM (if sufficient data)
    if len(data) >= 200:
        results['lstm'] = train_lstm_model(data, forecast_periods, target_col)
    
    return results

def get_available_prediction_methods():
    return [
        "ARIMA (Statistical)",
        "Random Forest (ML)",
        "Linear Regression (ML)",
        "Support Vector Regression (ML)",
        "LSTM Neural Network (DL)"
    ]

def get_prediction_method_descriptions():
    return {
        "ARIMA (Statistical)": "Auto-regressive integrated moving average model for time series",
        "Random Forest (ML)": "Ensemble of decision trees for non-linear patterns",
        "Linear Regression (ML)": "Linear model for simple trend analysis",
        "Support Vector Regression (ML)": "SVM-based regression for complex patterns",
        "LSTM Neural Network (DL)": "Recurrent neural network for sequential data"
    }

