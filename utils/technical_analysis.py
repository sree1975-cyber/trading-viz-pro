import pandas as pd
import numpy as np

# Dictionary of available indicators with their parameters
available_indicators = {
    "SMA": {"periods": [20, 50, 200]},
    "EMA": {"periods": [9, 21, 55]},
    "RSI": {"period": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "Bollinger Bands": {"period": 20, "std_dev": 2},
    "Stochastic Oscillator": {"k_period": 14, "d_period": 3},
    "ATR": {"period": 14},
    "OBV": {},
    "Ichimoku Cloud": {"conversion_line": 9, "base_line": 26, "leading_span_b": 52, "displacement": 26},
    "Parabolic SAR": {"initial_af": 0.02, "max_af": 0.2},
    "ADX": {"period": 14},
    "CCI": {"period": 20},
}

def calculate_indicators(data, selected_indicators):
    """
    Calculate selected technical indicators for the given data.
    
    Args:
        data (pd.DataFrame): OHLCV data
        selected_indicators (list): List of indicators to calculate
        
    Returns:
        dict: Dictionary containing calculated indicators
    """
    result = {}
    
    for indicator in selected_indicators:
        try:
            if indicator == "SMA":
                # Create a nested dictionary for SMA data
                sma_dict = {}
                for period in available_indicators[indicator]["periods"]:
                    sma_dict[f"{period}"] = calculate_sma(data, period)
                result["SMA"] = sma_dict
            
            elif indicator == "EMA":
                # Create a nested dictionary for EMA data
                ema_dict = {}
                for period in available_indicators[indicator]["periods"]:
                    ema_dict[f"{period}"] = calculate_ema(data, period)
                result["EMA"] = ema_dict
            
            elif indicator == "RSI":
                period = available_indicators[indicator]["period"]
                result["RSI"] = calculate_rsi(data, period)
                
            elif indicator == "MACD":
                fast = available_indicators[indicator]["fast"]
                slow = available_indicators[indicator]["slow"]
                signal = available_indicators[indicator]["signal"]
                macd, signal_line, histogram = calculate_macd(data, fast, slow, signal)
                result["MACD"] = {"MACD": macd, "Signal": signal_line, "Histogram": histogram}
                
            elif indicator == "Bollinger Bands":
                period = available_indicators[indicator]["period"]
                std_dev = available_indicators[indicator]["std_dev"]
                upper, middle, lower = calculate_bollinger_bands(data, period, std_dev)
                result["Bollinger Bands"] = {"Upper": upper, "Middle": middle, "Lower": lower}
                
            elif indicator == "Stochastic Oscillator":
                k_period = available_indicators[indicator]["k_period"]
                d_period = available_indicators[indicator]["d_period"]
                k, d = calculate_stochastic_oscillator(data, k_period, d_period)
                result["Stochastic Oscillator"] = {"K": k, "D": d}
                
            elif indicator == "ATR":
                period = available_indicators[indicator]["period"]
                result["ATR"] = calculate_atr(data, period)
                
            elif indicator == "OBV":
                result["OBV"] = calculate_obv(data)
                
            elif indicator == "Ichimoku Cloud":
                conv_line = available_indicators[indicator]["conversion_line"]
                base_line = available_indicators[indicator]["base_line"]
                leading_span_b = available_indicators[indicator]["leading_span_b"]
                displacement = available_indicators[indicator]["displacement"]
                tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(data, conv_line, base_line, leading_span_b, displacement)
                result["Ichimoku Cloud"] = {
                    "Tenkan": tenkan,
                    "Kijun": kijun,
                    "Senkou A": senkou_a,
                    "Senkou B": senkou_b,
                    "Chikou": chikou
                }
                
            elif indicator == "Parabolic SAR":
                initial_af = available_indicators[indicator]["initial_af"]
                max_af = available_indicators[indicator]["max_af"]
                result["Parabolic SAR"] = calculate_parabolic_sar(data, initial_af, max_af)
                
            elif indicator == "ADX":
                period = available_indicators[indicator]["period"]
                adx, pdi, ndi = calculate_adx(data, period)
                result["ADX"] = {"ADX": adx, "+DI": pdi, "-DI": ndi}
                
            elif indicator == "CCI":
                period = available_indicators[indicator]["period"]
                result["CCI"] = calculate_cci(data, period)
        except Exception as e:
            # Create a placeholder to avoid KeyError for failed indicators
            print(f"Error calculating {indicator}: {str(e)}")
            if indicator in ["SMA", "EMA", "MACD", "Bollinger Bands", "Stochastic Oscillator", "Ichimoku Cloud", "ADX"]:
                result[indicator] = {"Error": f"Failed to calculate {indicator}"}
            else:
                result[indicator] = pd.Series(np.nan, index=data.index)
    
    return result

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    try:
        return data['Close'].rolling(window=period).mean()
    except Exception as e:
        # Fallback method
        sma = pd.Series(index=data.index)
        for i in range(len(data)):
            if i < period - 1:
                sma.iloc[i] = np.nan
            else:
                sma.iloc[i] = data['Close'].iloc[i-period+1:i+1].mean()
        return sma

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    try:
        return data['Close'].ewm(span=period, adjust=False).mean()
    except Exception as e:
        # Manual EMA calculation
        alpha = 2 / (period + 1)
        ema = pd.Series(index=data.index)
        ema.iloc[0] = data['Close'].iloc[0]
        for i in range(1, len(data)):
            ema.iloc[i] = data['Close'].iloc[i] * alpha + ema.iloc[i-1] * (1 - alpha)
        return ema

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD, Signal Line, and Histogram"""
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle_band = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_obv(data):
    """Calculate On-Balance Volume"""
    obv = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_ichimoku(data, conversion_line=9, base_line=26, leading_span_b=52, displacement=26):
    """Calculate Ichimoku Cloud"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
    high_9 = data['High'].rolling(window=conversion_line).max()
    low_9 = data['Low'].rolling(window=conversion_line).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
    high_26 = data['High'].rolling(window=base_line).max()
    low_26 = data['Low'].rolling(window=base_line).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, plotted 26 periods ahead
    high_52 = data['High'].rolling(window=leading_span_b).max()
    low_52 = data['Low'].rolling(window=leading_span_b).min()
    senkou_b = ((high_52 + low_52) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Close price plotted 26 periods behind
    chikou = data['Close'].shift(-displacement)
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calculate_parabolic_sar(data, initial_af=0.02, max_af=0.2):
    """Calculate Parabolic SAR"""
    # Initialize
    sar = pd.Series(index=data.index)
    trend = pd.Series(index=data.index)
    ep = pd.Series(index=data.index)
    af = pd.Series(index=data.index)
    
    # Set initial values
    trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
    sar.iloc[0] = data['Low'].iloc[0]
    ep.iloc[0] = data['High'].iloc[0]
    af.iloc[0] = initial_af
    
    # Calculate SAR for each period
    for i in range(1, len(data)):
        # Previous period's SAR
        prev_sar = sar.iloc[i-1]
        
        # Current SAR
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = prev_sar + af.iloc[i-1] * (ep.iloc[i-1] - prev_sar)
            # Make sure SAR is below the previous two lows
            sar.iloc[i] = min(sar.iloc[i], data['Low'].iloc[i-1], data['Low'].iloc[max(0, i-2)])
            
            # Update trend, EP, and AF
            if data['Low'].iloc[i] < sar.iloc[i]:
                trend.iloc[i] = -1  # Switch to downtrend
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = data['Low'].iloc[i]
                af.iloc[i] = initial_af
            else:
                trend.iloc[i] = 1
                if data['High'].iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = data['High'].iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + initial_af, max_af)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
        else:  # Downtrend
            sar.iloc[i] = prev_sar - af.iloc[i-1] * (prev_sar - ep.iloc[i-1])
            # Make sure SAR is above the previous two highs
            sar.iloc[i] = max(sar.iloc[i], data['High'].iloc[i-1], data['High'].iloc[max(0, i-2)])
            
            # Update trend, EP, and AF
            if data['High'].iloc[i] > sar.iloc[i]:
                trend.iloc[i] = 1  # Switch to uptrend
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = data['High'].iloc[i]
                af.iloc[i] = initial_af
            else:
                trend.iloc[i] = -1
                if data['Low'].iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = data['Low'].iloc[i]
                    af.iloc[i] = min(af.iloc[i-1] + initial_af, max_af)
                else:
                    ep.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = af.iloc[i-1]
    
    return sar

def calculate_adx(data, period=14):
    """Calculate Average Directional Index"""
    # True Range
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff().mul(-1)
    
    # Set values to 0 if not valid
    plus_dm[plus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm) | (minus_dm <= 0)] = 0
    
    minus_dm[minus_dm < 0] = 0
    minus_dm[(minus_dm < plus_dm) | (plus_dm <= 0)] = 0
    
    # Smooth +DM and -DM
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate Directional Index (DX)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
    
    # Calculate ADX as smoothed DX
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_cci(data, period=20):
    """Calculate Commodity Channel Index"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    mean_tp = typical_price.rolling(window=period).mean()
    
    # Manual calculation of mean absolute deviation since .mad() is deprecated
    def calculate_mad(x):
        return np.abs(x - x.mean()).mean()
    
    mean_deviation = typical_price.rolling(window=period).apply(calculate_mad, raw=True)
    
    # Avoid division by zero
    mean_deviation = mean_deviation.replace(0, np.nan)
    
    cci = (typical_price - mean_tp) / (0.015 * mean_deviation)
    
    return cci
