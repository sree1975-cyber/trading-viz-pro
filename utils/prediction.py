import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_features(data, target_col='Close', lag_days=5):
    """Create additional features for machine learning models."""
    df = data.copy()
    df['MA_5'] = df[target_col].rolling(window=5).mean()
    df['MA_20'] = df[target_col].rolling(window=20).mean()
    df['Return_1'] = df[target_col].pct_change(periods=1)
    for lag in range(1, lag_days + 1):
        df[f'{target_col}_Lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)
    return df

def train_simple_ma_model(data, column='Close', forecast_periods=30, window=20):
    """Train and forecast using a simple moving average model."""
    try:
        last_value = data[column].iloc[-1]
        ma_value = data[column].rolling(window=window).mean().iloc[-1]
        ratio = last_value / ma_value if ma_value != 0 else 1.0
        forecast_values = [last_value * ratio for _ in range(forecast_periods)]
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
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
    """Train and forecast using a simple linear trend model."""
    try:
        last_n_days = min(30, len(data))
        values = data[column].values[-last_n_days:]
        x = np.arange(last_n_days)
        m, b = np.polyfit(x, values, 1) if len(values) > 1 else (0, values[0])
        forecast_values = [m * (last_n_days + i) + b for i in range(forecast_periods)]
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
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

def train_arima_model(data, column='Close', forecast_periods=30):
    """Train and forecast using an ARIMA model."""
    try:
        if len(data) < 50:
            return {'predictions': None, 'error': 'Need ≥50 data points', 'model': 'ARIMA'}
        y = data[column].values
        model = auto_arima(y, seasonal=False, suppress_warnings=True, stepwise=True, max_p=3, max_q=3, max_d=1)
        arima = ARIMA(y, order=model.order).fit()
        forecast = arima.forecast(steps=forecast_periods)
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(forecast, index=forecast_dates),
            'model': 'ARIMA',
            'order': model.order
        }
    except Exception as e:
        logger.error(f"Error in ARIMA model: {str(e)}")
        return {'predictions': None, 'error': str(e), 'model': 'ARIMA'}

def train_random_forest_model(data, forecast_periods=30, target_col='Close'):
    """Train and forecast using a Random Forest model."""
    try:
        df = create_features(data, target_col)
        y = df[target_col].values
        X = df.drop(target_col, axis=1)
        scaler_X = MinMaxScaler().fit(X)
        scaler_y = MinMaxScaler().fit(y.reshape(-1,1))
        X_s = scaler_X.transform(X)
        y_s = scaler_y.transform(y.reshape(-1,1)).ravel()
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_s, y_s)
        current_features = X_s[-1].copy()
        forecasts = []
        for _ in range(forecast_periods):
            pred = model.predict([current_features])[0]
            forecasts.append(pred)
            current_features = np.roll(current_features, 1)
            current_features[0] = pred
        forecasts = scaler_y.inverse_transform(np.array(forecasts).reshape(-1,1)).ravel()
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(forecasts, index=forecast_dates),
            'model': 'Random Forest',
            'error': None
        }
    except Exception as e:
        logger.error(f"Error in Random Forest model: {str(e)}")
        return {'predictions': None, 'error': str(e), 'model': 'Random Forest'}

def train_svr_model(data, forecast_periods=30, target_col='Close'):
    """Train and forecast using an SVR model."""
    try:
        df = create_features(data, target_col)
        y = df[target_col].values
        X = df.drop(target_col, axis=1)
        scaler_X = MinMaxScaler().fit(X)
        scaler_y = MinMaxScaler().fit(y.reshape(-1,1))
        X_s = scaler_X.transform(X)
        y_s = scaler_y.transform(y.reshape(-1,1)).ravel()
        model = SVR(kernel='rbf', C=100)
        model.fit(X_s, y_s)
        current_features = X_s[-1].copy()
        forecasts = []
        for _ in range(forecast_periods):
            pred = model.predict([current_features])[0]
            forecasts.append(pred)
            current_features = np.roll(current_features, 1)
            current_features[0] = pred
        forecasts = scaler_y.inverse_transform(np.array(forecasts).reshape(-1,1)).ravel()
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(forecasts, index=forecast_dates),
            'model': 'SVR',
            'error': None
        }
    except Exception as e:
        logger.error(f"Error in SVR model: {str(e)}")
        return {'predictions': None, 'error': str(e), 'model': 'SVR'}

def train_lstm_model(data, forecast_periods=30, target_col='Close', window=60):
    """Train and forecast using an LSTM model."""
    try:
        if len(data) < 100:
            return {'predictions': None, 'error': 'Need ≥100 data points', 'model': 'LSTM'}
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data[[target_col]])
        X, y = [], []
        for i in range(len(scaled)-window):
            X.append(scaled[i:i+window, 0])
            y.append(scaled[i+window, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window,1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        forecast_seq = scaled[-window:]
        predictions = []
        for _ in range(forecast_periods):
            pred = model.predict(forecast_seq.reshape(1,window,1), verbose=0)[0,0]
            predictions.append(pred)
            forecast_seq = np.append(forecast_seq[1:], pred)
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).ravel()
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        return {
            'predictions': pd.Series(predictions, index=forecast_dates),
            'model': 'LSTM',
            'error': None
        }
    except Exception as e:
        logger.error(f"Error in LSTM model: {str(e)}")
        return {'predictions': None, 'error': str(e), 'model': 'LSTM'}

def get_price_predictions(data, forecast_periods=30, target_col='Close'):
    """Generate price predictions using multiple models."""
    results = {}
    if len(data) < 30:
        return {'error': 'Not enough data points for reliable forecasting. Please use a longer timeframe.'}
    MODEL_FUNCTIONS = {
        'Moving Average (Simple)': lambda: train_simple_ma_model(data, target_col, forecast_periods),
        'Linear Trend (Simple)': lambda: train_linear_trend_model(data, target_col, forecast_periods),
        'ARIMA (Statistical)': lambda: train_arima_model(data, target_col, forecast_periods),
        'Random Forest (ML)': lambda: train_random_forest_model(data, forecast_periods, target_col),
        'SVR (ML)': lambda: train_svr_model(data, forecast_periods, target_col),
        'LSTM (DL)': lambda: train_lstm_model(data, forecast_periods, target_col)
    }
    for model_name, model_func in MODEL_FUNCTIONS.items():
        try:
            results[model_name] = model_func()
        except Exception as e:
            logger.error(f"Error in {model_name}: {str(e)}")
            results[model_name] = {'predictions': None, 'error': str(e), 'model': model_name}
    return results

def plot_predictions(data, predictions, target_col='Close'):
    """Plot the predictions from various models."""
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data[target_col], label='Actual Prices', color='blue')
    for model_name, result in predictions.items():
        if result['predictions'] is not None:
            plt.plot(result['predictions'].index, result['predictions'], label=f"{model_name} Predictions")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Predictions')
    plt.legend()
    plt.show()

def display_predictions_table(predictions):
    """Display the predictions in a table format."""
    table_data = []
    for model_name, result in predictions.items():
        if result['predictions'] is not None:
            table_data.append([model_name, result['predictions'].iloc[-1]])
        else:
            table_data.append([model_name, result['error']])
    table_df = pd.DataFrame(table_data, columns=['Model', 'Last Predicted Value'])
    print(table_df)

def get_available_prediction_methods():
    """Returns the list of available prediction methods."""
    return [
        "Moving Average (Simple)",
        "Linear Trend (Simple)",
        "ARIMA (Statistical)",
        "Random Forest (ML)",
        "SVR (ML)",
        "LSTM (DL)"
    ]

def get_prediction_method_descriptions():
    """Returns descriptions of the prediction methods."""
    return {
        "Moving Average (Simple)": "A simple model that projects future prices based on the relationship between current price and its moving average.",
        "Linear Trend (Simple)": "A model that fits a linear trend to recent price data and projects that trend into the future.",
        "ARIMA (Statistical)": "Auto-regressive integrated moving average model for time series forecasting.",
        "Random Forest (ML)": "Ensemble of decision trees capturing non-linear patterns.",
        "SVR (ML)": "Support vector machine for complex regression patterns.",
        "LSTM (DL)": "Long short-term memory neural network for sequential data."
    }

# Example usage
if __name__ == "__main__":
    # Generate example data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({'Close': np.random.rand(100)}, index=dates)

    # Get predictions
    predictions = get_price_predictions(data, forecast_periods=30, target_col='Close')

    # Plot predictions
    plot_predictions(data, predictions)

    # Display predictions table
    display_predictions_table(predictions)
