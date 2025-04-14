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
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        y_pred_actual = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
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
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(forecast_values, index=forecast_dates),
            'model': model_type,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'actual_mse': mean_squared_error(y_test_actual, y_pred_actual)
        }
    except Exception as e:
        logger.error(f"Error in ML model: {str(e)}")
        return {
            'predictions': None,
            'model': model_type,
            'error': str(e)
        }

def train_lstm_model(data, forecast_periods=30, target_col='Close', window_size=60):
    """
    Train and forecast using LSTM neural network.
    """
    try:
        df = data.copy()
        values = df[target_col].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)
        X, y = [], []
        for i in range(len(scaled_values) - window_size):
            X.append(scaled_values[i:i+window_size, 0])
            y.append(scaled_values[i+window_size, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        y_pred_actual = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        forecast_values = []
        last_sequence = scaled_values[-window_size:].reshape(1, window_size, 1)
        for _ in range(forecast_periods):
            next_pred = model.predict(last_sequence)
            forecast_values.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
        forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1)).flatten()
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(forecast_values, index=forecast_dates),
            'model': 'LSTM',
            'mse': mse,
            'actual_mse': mean_squared_error(y_test_actual, y_pred_actual)
        }
    except Exception as e:
        logger.error(f"Error in LSTM model: {str(e)}")
        return {
            'predictions': None,
            'model': 'LSTM',
            'error': str(e)
        }

def get_price_predictions(data, forecast_periods=30, target_col='Close'):
    results = {}
    if len(data) < 100:
        return {
            'error': 'Not enough data points for reliable forecasting. Please use a longer timeframe.'
        }
    arima_results = train_arima_model(data, column=target_col, forecast_periods=forecast_periods)
    results['arima'] = arima_results
    ml_models = ['random_forest', 'linear', 'svr']
    for model_type in ml_models:
        ml_results = train_machine_learning_model(
            data, model_type=model_type, forecast_periods=forecast_periods, target_col=target_col
        )
        results[model_type] = ml_results
    if len(data) >= 200:
        lstm_results = train_lstm_model(data, forecast_periods=forecast_periods, target_col=target_col)
        results['lstm'] = lstm_results
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
        "ARIMA (Statistical)": "Auto-Regressive Integrated Moving Average model. Good for time series with trends but not good for highly volatile data.",
        "Random Forest (ML)": "Ensemble learning method using multiple decision trees. Good for capturing non-linear relationships in data.",
        "Linear Regression (ML)": "Simple linear model predicting a target variable based on features. Works well for linear trends.",
        "Support Vector Regression (ML)": "Support Vector Machine applied to regression problems. Works well with high-dimensional data.",
        "LSTM Neural Network (DL)": "Long Short-Term Memory network, a type of recurrent neural network capable of learning long-term dependencies."
    }
