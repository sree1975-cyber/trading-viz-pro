import pandas as pd
import numpy as np

# Dictionary of pattern descriptions (keeping this the same for consistency)
pattern_descriptions = {
    "Double Top": "A bearish reversal pattern that forms after an extended uptrend. It consists of two consecutive peaks at roughly the same price level, indicating a potential trend reversal from bullish to bearish.",
    
    "Double Bottom": "A bullish reversal pattern that forms after an extended downtrend. It consists of two consecutive troughs at roughly the same price level, indicating a potential trend reversal from bearish to bullish.",
    
    "Head and Shoulders": "A bearish reversal pattern characterized by three peaks, with the middle peak (head) being higher than the two outer peaks (shoulders), which are at roughly the same level. It signals a trend reversal from bullish to bearish.",
    
    "Inverse Head and Shoulders": "A bullish reversal pattern characterized by three troughs, with the middle trough (head) being lower than the two outer troughs (shoulders), which are at roughly the same level. It signals a trend reversal from bearish to bullish.",
    
    "Triple Top": "A bearish reversal pattern consisting of three peaks at approximately the same price level. It indicates resistance and a potential trend reversal from bullish to bearish.",
    
    "Triple Bottom": "A bullish reversal pattern consisting of three troughs at approximately the same price level. It indicates support and a potential trend reversal from bearish to bullish.",
    
    "Ascending Triangle": "A bullish continuation pattern characterized by a flat upper resistance line and an upward-sloping lower support line. It suggests increased buying pressure and a potential breakout to the upside.",
    
    "Descending Triangle": "A bearish continuation pattern characterized by a flat lower support line and a downward-sloping upper resistance line. It suggests increased selling pressure and a potential breakout to the downside.",
    
    "Symmetrical Triangle": "A continuation pattern characterized by converging trend lines, where the upper line is descending and the lower line is ascending. It indicates a period of consolidation before a potential breakout in either direction.",
    
    "Rising Wedge": "A bearish reversal or continuation pattern formed by converging trend lines, both sloping upward with the upper line having a lesser slope. It signals a potential reversal from bullish to bearish.",
    
    "Falling Wedge": "A bullish reversal or continuation pattern formed by converging trend lines, both sloping downward with the lower line having a lesser slope. It signals a potential reversal from bearish to bullish.",
    
    "Cup and Handle": "A bullish continuation pattern resembling a cup with a handle. The cup forms as a rounded bottom, followed by a period of consolidation (the handle). It indicates a potential continuation of the uptrend.",
    
    "Bullish Flag": "A bullish continuation pattern consisting of a strong upward move (the flagpole) followed by a period of consolidation with parallel downward-sloping trend lines (the flag). It suggests a potential continuation of the uptrend.",
    
    "Bearish Flag": "A bearish continuation pattern consisting of a strong downward move (the flagpole) followed by a period of consolidation with parallel upward-sloping trend lines (the flag). It suggests a potential continuation of the downtrend.",
    
    "Bullish Pennant": "A bullish continuation pattern similar to a bullish flag, but with converging trend lines during the consolidation phase. It indicates a brief pause in the uptrend before a potential continuation.",
    
    "Bearish Pennant": "A bearish continuation pattern similar to a bearish flag, but with converging trend lines during the consolidation phase. It indicates a brief pause in the downtrend before a potential continuation.",
    
    "Bullish Engulfing": "A bullish reversal candlestick pattern consisting of a smaller bearish candle followed by a larger bullish candle that completely engulfs the previous candle. It signals a potential reversal from bearish to bullish.",
    
    "Bearish Engulfing": "A bearish reversal candlestick pattern consisting of a smaller bullish candle followed by a larger bearish candle that completely engulfs the previous candle. It signals a potential reversal from bullish to bearish.",
    
    "Morning Star": "A bullish reversal candlestick pattern consisting of three candles: a large bearish candle, a small-bodied candle, and a large bullish candle. It signals a potential reversal from bearish to bullish.",
    
    "Evening Star": "A bearish reversal candlestick pattern consisting of three candles: a large bullish candle, a small-bodied candle, and a large bearish candle. It signals a potential reversal from bullish to bearish."
}

def detect_patterns(data):
    """
    Detect common chart patterns in the price data.
    This is a simplified, more robust implementation that works across different environments.
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        list: List of detected patterns with details
    """
    try:
        patterns = []
        
        # Ensure we have enough data for pattern detection
        if len(data) < 20:
            return patterns
        
        # Detect candlestick patterns (they require fewer data points)
        patterns.extend(detect_candlestick_patterns(data))
        
        # For more complex patterns, we need more data
        if len(data) >= 40:
            # Detect a limited set of patterns for better performance
            patterns.extend(detect_double_patterns(data))
            
            # Only add these if we have substantial data
            if len(data) >= 60:
                patterns.extend(detect_head_and_shoulders(data))
        
        return patterns
    except Exception as e:
        # Return an empty list if any error occurs
        print(f"Error in pattern detection: {str(e)}")
        return []

def detect_candlestick_patterns(data):
    """Detect common candlestick patterns - simplified robust version"""
    patterns = []
    
    try:
        # Loop through the data to find patterns (skip the first 3 and last 1)
        for i in range(3, len(data) - 1):
            # Get string representation of date that's compatible with all pandas versions
            try:
                current_date = data.index[i].strftime('%Y-%m-%d')
            except:
                # Fallback for other datetime formats
                current_date = str(data.index[i]).split(' ')[0]
            
            # Get the current and previous candles
            current = data.iloc[i]
            prev1 = data.iloc[i-1]
            prev2 = data.iloc[i-2]
            
            # Bullish Engulfing
            if (prev1['Close'] < prev1['Open'] and  # Previous candle is bearish
                current['Open'] < prev1['Close'] and  # Current opens below previous close
                current['Close'] > prev1['Open'] and  # Current closes above previous open
                current['Close'] > current['Open']):  # Current candle is bullish
                
                # Calculate pattern strength based on size of the engulfing candle
                strength = (current['Close'] - current['Open']) / (prev1['Open'] - prev1['Close'])
                strength = min(strength * 100, 100)  # Cap at 100%
                
                patterns.append({
                    'name': 'Bullish Engulfing',
                    'type': 'Reversal',
                    'timestamp': current_date,
                    'strength': f"{strength:.1f}%"
                })
            
            # Bearish Engulfing
            if (prev1['Close'] > prev1['Open'] and  # Previous candle is bullish
                current['Open'] > prev1['Close'] and  # Current opens above previous close
                current['Close'] < prev1['Open'] and  # Current closes below previous open
                current['Close'] < current['Open']):  # Current candle is bearish
                
                # Calculate pattern strength based on size of the engulfing candle
                strength = (current['Open'] - current['Close']) / (prev1['Close'] - prev1['Open'])
                strength = min(strength * 100, 100)  # Cap at 100%
                
                patterns.append({
                    'name': 'Bearish Engulfing',
                    'type': 'Reversal',
                    'timestamp': current_date,
                    'strength': f"{strength:.1f}%"
                })
            
            # Morning Star (need 3 candles)
            if (prev2['Close'] < prev2['Open'] and  # First candle is bearish
                abs(prev1['Close'] - prev1['Open']) < abs(prev2['Close'] - prev2['Open']) * 0.3 and  # Second candle is small
                current['Close'] > current['Open'] and  # Third candle is bullish
                current['Close'] > (prev2['Open'] + prev2['Close']) / 2):  # Third candle closes above midpoint of first
                
                # Calculate pattern strength
                strength = (current['Close'] - current['Open']) / ((prev2['Open'] - prev2['Close']) * 0.5)
                strength = min(strength * 100, 100)  # Cap at 100%
                
                patterns.append({
                    'name': 'Morning Star',
                    'type': 'Reversal',
                    'timestamp': current_date,
                    'strength': f"{strength:.1f}%"
                })
            
            # Evening Star (need 3 candles)
            if (prev2['Close'] > prev2['Open'] and  # First candle is bullish
                abs(prev1['Close'] - prev1['Open']) < abs(prev2['Close'] - prev2['Open']) * 0.3 and  # Second candle is small
                current['Close'] < current['Open'] and  # Third candle is bearish
                current['Close'] < (prev2['Open'] + prev2['Close']) / 2):  # Third candle closes below midpoint of first
                
                # Calculate pattern strength
                strength = (current['Open'] - current['Close']) / ((prev2['Close'] - prev2['Open']) * 0.5)
                strength = min(strength * 100, 100)  # Cap at 100%
                
                patterns.append({
                    'name': 'Evening Star',
                    'type': 'Reversal',
                    'timestamp': current_date,
                    'strength': f"{strength:.1f}%"
                })
        
        return patterns
    except Exception as e:
        print(f"Error in candlestick detection: {str(e)}")
        return []

def detect_double_patterns(data):
    """Detect double top/bottom patterns - simplified robust version"""
    patterns = []
    
    try:
        # Need at least 20 data points for this pattern
        if len(data) < 20:
            return patterns
        
        # Convert to numpy arrays for faster processing
        highs = data['High'].values
        lows = data['Low'].values
        
        # Parameters for pattern detection
        window = min(10, len(data) // 4)  # Window to look for the second peak/trough
        threshold = 0.03  # Threshold for similarity in price levels (3%)
        
        # Loop through the data safely to find patterns
        for i in range(window, len(data) - window - 1):
            # Get string representation of date that's compatible with all pandas versions
            try:
                current_date = data.index[i].strftime('%Y-%m-%d')
            except:
                # Fallback for other datetime formats
                current_date = str(data.index[i]).split(' ')[0]
            
            # Check for Double Top
            if i >= 5 and i + 6 < len(data):
                # Get local ranges safely
                pre_range = range(max(0, i-5), i)
                post_range = range(i+1, min(i+6, len(data)))
                
                # Check if current point is a local maximum (potential peak)
                if all(highs[i] > highs[j] for j in pre_range) and all(highs[i] > highs[j] for j in post_range):
                    # Look for another similar peak
                    for j in range(i + 5, min(i + window + 1, len(data) - 6)):
                        if j + 6 >= len(data):
                            continue
                        
                        j_pre_range = range(max(0, j-5), j)
                        j_post_range = range(j+1, min(j+6, len(data)))
                        
                        # Check if we found another peak
                        if all(highs[j] > highs[k] for k in j_pre_range) and all(highs[j] > highs[k] for k in j_post_range):
                            # Check if the peaks are at similar levels
                            if abs(highs[i] - highs[j]) / highs[i] < threshold:
                                # Get min between safely
                                between_range = range(i, min(j+1, len(data)))
                                min_between = min(lows[k] for k in between_range)
                                
                                if (highs[i] - min_between) / highs[i] > 0.03:  # At least 3% drop between peaks
                                    # Calculate pattern strength
                                    strength = ((highs[i] - min_between) / highs[i]) * 100
                                    strength = min(strength * 3, 100)  # Scale and cap
                                    
                                    try:
                                        date_str = data.index[j].strftime('%Y-%m-%d')
                                    except:
                                        date_str = str(data.index[j]).split(' ')[0]
                                    
                                    patterns.append({
                                        'name': 'Double Top',
                                        'type': 'Reversal',
                                        'timestamp': date_str,
                                        'strength': f"{strength:.1f}%"
                                    })
            
            # Check for Double Bottom (similar logic to Double Top but for lows)
            if i >= 5 and i + 6 < len(data):
                # Get local ranges safely
                pre_range = range(max(0, i-5), i)
                post_range = range(i+1, min(i+6, len(data)))
                
                # Check if current point is a local minimum (potential trough)
                if all(lows[i] < lows[j] for j in pre_range) and all(lows[i] < lows[j] for j in post_range):
                    # Look for another similar trough
                    for j in range(i + 5, min(i + window + 1, len(data) - 6)):
                        if j + 6 >= len(data):
                            continue
                        
                        j_pre_range = range(max(0, j-5), j)
                        j_post_range = range(j+1, min(j+6, len(data)))
                        
                        # Check if we found another trough
                        if all(lows[j] < lows[k] for k in j_pre_range) and all(lows[j] < lows[k] for k in j_post_range):
                            # Check if the troughs are at similar levels
                            if abs(lows[i] - lows[j]) / lows[i] < threshold:
                                # Get max between safely
                                between_range = range(i, min(j+1, len(data)))
                                max_between = max(highs[k] for k in between_range)
                                
                                if (max_between - lows[i]) / lows[i] > 0.03:  # At least 3% rise between troughs
                                    # Calculate pattern strength
                                    strength = ((max_between - lows[i]) / lows[i]) * 100
                                    strength = min(strength * 3, 100)  # Scale and cap
                                    
                                    try:
                                        date_str = data.index[j].strftime('%Y-%m-%d')
                                    except:
                                        date_str = str(data.index[j]).split(' ')[0]
                                    
                                    patterns.append({
                                        'name': 'Double Bottom',
                                        'type': 'Reversal',
                                        'timestamp': date_str,
                                        'strength': f"{strength:.1f}%"
                                    })
        
        return patterns
    except Exception as e:
        print(f"Error in double pattern detection: {str(e)}")
        return []

def detect_head_and_shoulders(data):
    """Detect head and shoulders (and inverse) patterns - simplified robust version"""
    patterns = []
    
    try:
        # Need at least 30 data points for this pattern
        if len(data) < 30:
            return patterns
        
        # Extract the highs and lows safely using numpy
        highs = data['High'].values
        lows = data['Low'].values
        
        # Parameters for pattern detection
        window = min(8, len(data) // 5)  # Smaller window for better performance
        threshold = 0.03  # Threshold for similarity in shoulder levels (3%)
        
        # Loop through the data to find patterns (with safety checks)
        for i in range(window, len(data) - 2*window - 1):
            if i + 5 >= len(data) or i < 5:
                continue
            
            # Check for a potential left shoulder (local maximum) safely
            i_pre_range = range(max(0, i-5), i)
            i_post_range = range(i+1, min(i+6, len(data)))
            
            if all(highs[i] > highs[j] for j in i_pre_range) and all(highs[i] > highs[j] for j in i_post_range):
                # Look for a head (higher peak)
                for j in range(i + 5, min(i + window + 1, len(data) - 6)):
                    if j + 5 >= len(data):
                        continue
                    
                    j_pre_range = range(max(0, j-5), j)
                    j_post_range = range(j+1, min(j+6, len(data)))
                    
                    if (highs[j] > highs[i] and  # Head is higher than left shoulder
                        all(highs[j] > highs[k] for k in j_pre_range) and 
                        all(highs[j] > highs[k] for k in j_post_range)):
                        
                        # Look for a right shoulder (similar height to left shoulder)
                        for k in range(j + 5, min(j + window + 1, len(data) - 6)):
                            if k + 5 >= len(data):
                                continue
                                
                            k_pre_range = range(max(0, k-5), k)
                            k_post_range = range(k+1, min(k+6, len(data)))
                            
                            if (all(highs[k] > highs[m] for m in k_pre_range) and 
                                all(highs[k] > highs[m] for m in k_post_range)):
                                
                                # Check if shoulders are at similar levels
                                if abs(highs[i] - highs[k]) / highs[i] < threshold:
                                    # Check if there's a neckline (support level)
                                    between1 = range(i, min(j+1, len(data)))
                                    between2 = range(j, min(k+1, len(data)))
                                    
                                    trough1 = min(lows[t] for t in between1)
                                    trough2 = min(lows[t] for t in between2)
                                    
                                    # Ensure the troughs are at similar levels
                                    if abs(trough1 - trough2) / trough1 < threshold:
                                        # Calculate pattern strength
                                        strength = ((highs[j] - min(trough1, trough2)) / highs[j]) * 100
                                        strength = min(strength * 2, 100)  # Scale and cap
                                        
                                        try:
                                            date_str = data.index[k].strftime('%Y-%m-%d')
                                        except:
                                            date_str = str(data.index[k]).split(' ')[0]
                                        
                                        patterns.append({
                                            'name': 'Head and Shoulders',
                                            'type': 'Reversal',
                                            'timestamp': date_str,
                                            'strength': f"{strength:.1f}%"
                                        })
        
        return patterns
    except Exception as e:
        print(f"Error in head and shoulders detection: {str(e)}")
        return []
