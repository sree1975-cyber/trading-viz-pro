import pandas as pd
import numpy as np
from datetime import datetime

# Dictionary of pattern descriptions
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
    
    Args:
        data (pd.DataFrame): OHLCV data
        
    Returns:
        list: List of detected patterns with details
    """
    patterns = []
    
    # Ensure we have enough data for pattern detection
    if len(data) < 30:
        return patterns
    
    # Detect candlestick patterns (they require fewer data points)
    patterns.extend(detect_candlestick_patterns(data))
    
    # For more complex patterns, we need more data
    if len(data) >= 50:
        patterns.extend(detect_double_patterns(data))
        patterns.extend(detect_head_and_shoulders(data))
        patterns.extend(detect_triangle_patterns(data))
        patterns.extend(detect_wedge_patterns(data))
        patterns.extend(detect_flag_patterns(data))
    
    return patterns

def detect_candlestick_patterns(data):
    """Detect common candlestick patterns"""
    patterns = []
    
    # Loop through the data to find patterns
    for i in range(2, len(data)):
        # Current date for the pattern
        current_date = data.index[i].strftime('%Y-%m-%d')
        
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
        if (i >= 2 and
            prev2['Close'] < prev2['Open'] and  # First candle is bearish
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
        if (i >= 2 and
            prev2['Close'] > prev2['Open'] and  # First candle is bullish
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

def detect_double_patterns(data):
    """Detect double top/bottom patterns"""
    patterns = []
    
    # Need at least 20 data points for this pattern
    if len(data) < 20:
        return patterns
    
    # Extract the highs and lows
    highs = data['High'].values
    lows = data['Low'].values
    
    # Parameters for pattern detection
    window = 15  # Window to look for the second peak/trough
    threshold = 0.02  # Threshold for similarity in price levels (2%)
    
    # Loop through the data to find patterns
    for i in range(window, len(data) - window):
        current_date = data.index[i].strftime('%Y-%m-%d')
        
        # Check if current point is a local maximum (potential peak)
        if (highs[i] > max(highs[i-5:i]) and highs[i] > max(highs[i+1:i+6])):
            # Look for another similar peak before and after
            for j in range(i + 5, min(i + window + 1, len(data) - 5)):
                if (highs[j] > max(highs[j-5:j]) and highs[j] > max(highs[j+1:min(j+6, len(data)-1)])):
                    # Check if the peaks are at similar levels
                    if abs(highs[i] - highs[j]) / highs[i] < threshold:
                        # Check if there's a significant trough between the peaks
                        min_between = min(lows[i:j+1])
                        if (highs[i] - min_between) / highs[i] > 0.03:  # At least 3% drop between peaks
                            # Calculate pattern strength
                            strength = ((highs[i] - min_between) / highs[i]) * 100
                            strength = min(strength * 3, 100)  # Scale and cap
                            
                            patterns.append({
                                'name': 'Double Top',
                                'type': 'Reversal',
                                'timestamp': data.index[j].strftime('%Y-%m-%d'),
                                'strength': f"{strength:.1f}%"
                            })
        
        # Check if current point is a local minimum (potential trough)
        if (lows[i] < min(lows[i-5:i]) and lows[i] < min(lows[i+1:i+6])):
            # Look for another similar trough before and after
            for j in range(i + 5, min(i + window + 1, len(data) - 5)):
                if (lows[j] < min(lows[j-5:j]) and lows[j] < min(lows[j+1:min(j+6, len(data)-1)])):
                    # Check if the troughs are at similar levels
                    if abs(lows[i] - lows[j]) / lows[i] < threshold:
                        # Check if there's a significant peak between the troughs
                        max_between = max(highs[i:j+1])
                        if (max_between - lows[i]) / lows[i] > 0.03:  # At least 3% rise between troughs
                            # Calculate pattern strength
                            strength = ((max_between - lows[i]) / lows[i]) * 100
                            strength = min(strength * 3, 100)  # Scale and cap
                            
                            patterns.append({
                                'name': 'Double Bottom',
                                'type': 'Reversal',
                                'timestamp': data.index[j].strftime('%Y-%m-%d'),
                                'strength': f"{strength:.1f}%"
                            })
    
    return patterns

def detect_head_and_shoulders(data):
    """Detect head and shoulders (and inverse) patterns"""
    patterns = []
    
    # Need at least 30 data points for this pattern
    if len(data) < 30:
        return patterns
    
    # Extract the highs and lows
    highs = data['High'].values
    lows = data['Low'].values
    
    # Parameters for pattern detection
    window = 10  # Window to look for shoulders
    threshold = 0.03  # Threshold for similarity in shoulder levels (3%)
    
    # Loop through the data to find patterns
    for i in range(window, len(data) - 2*window):
        # Check for a potential left shoulder (local maximum)
        if (highs[i] > max(highs[i-5:i]) and highs[i] > max(highs[i+1:i+6])):
            # Look for a head (higher peak)
            for j in range(i + 5, i + window + 1):
                if (highs[j] > highs[i] and  # Head is higher than left shoulder
                    highs[j] > max(highs[j-5:j]) and highs[j] > max(highs[j+1:j+6])):
                    # Look for a right shoulder (similar height to left shoulder)
                    for k in range(j + 5, min(j + window + 1, len(data) - 5)):
                        if (highs[k] > max(highs[k-5:k]) and highs[k] > max(highs[k+1:min(k+6, len(data)-1)])):
                            # Check if shoulders are at similar levels
                            if abs(highs[i] - highs[k]) / highs[i] < threshold:
                                # Check if there's a neckline (support level connecting the troughs)
                                trough1 = min(lows[i:j+1])
                                trough2 = min(lows[j:k+1])
                                
                                # Ensure the troughs are at similar levels
                                if abs(trough1 - trough2) / trough1 < threshold:
                                    # Calculate pattern strength
                                    strength = ((highs[j] - min(trough1, trough2)) / highs[j]) * 100
                                    strength = min(strength * 2, 100)  # Scale and cap
                                    
                                    patterns.append({
                                        'name': 'Head and Shoulders',
                                        'type': 'Reversal',
                                        'timestamp': data.index[k].strftime('%Y-%m-%d'),
                                        'strength': f"{strength:.1f}%"
                                    })
        
        # Check for a potential left shoulder (local minimum) - Inverse H&S
        if (lows[i] < min(lows[i-5:i]) and lows[i] < min(lows[i+1:i+6])):
            # Look for a head (lower trough)
            for j in range(i + 5, i + window + 1):
                if (lows[j] < lows[i] and  # Head is lower than left shoulder
                    lows[j] < min(lows[j-5:j]) and lows[j] < min(lows[j+1:j+6])):
                    # Look for a right shoulder (similar height to left shoulder)
                    for k in range(j + 5, min(j + window + 1, len(data) - 5)):
                        if (lows[k] < min(lows[k-5:k]) and lows[k] < min(lows[k+1:min(k+6, len(data)-1)])):
                            # Check if shoulders are at similar levels
                            if abs(lows[i] - lows[k]) / lows[i] < threshold:
                                # Check if there's a neckline (resistance level connecting the peaks)
                                peak1 = max(highs[i:j+1])
                                peak2 = max(highs[j:k+1])
                                
                                # Ensure the peaks are at similar levels
                                if abs(peak1 - peak2) / peak1 < threshold:
                                    # Calculate pattern strength
                                    strength = ((max(peak1, peak2) - lows[j]) / lows[j]) * 100
                                    strength = min(strength * 2, 100)  # Scale and cap
                                    
                                    patterns.append({
                                        'name': 'Inverse Head and Shoulders',
                                        'type': 'Reversal',
                                        'timestamp': data.index[k].strftime('%Y-%m-%d'),
                                        'strength': f"{strength:.1f}%"
                                    })
    
    return patterns

def detect_triangle_patterns(data):
    """Detect triangle patterns (ascending, descending, symmetrical)"""
    patterns = []
    
    # Need at least 20 data points for this pattern
    if len(data) < 20:
        return patterns
    
    # Extract close prices
    closes = data['Close'].values
    highs = data['High'].values
    lows = data['Low'].values
    
    # Parameters
    min_points = 5  # Minimum number of points to form a triangle
    min_duration = 10  # Minimum number of days for a valid triangle
    
    # Loop through potential triangles
    for start in range(len(data) - min_duration):
        end = min(start + 30, len(data) - 1)  # Look at most 30 bars ahead
        
        if end - start < min_duration:
            continue
        
        # Find the high points and low points
        high_points = []
        low_points = []
        
        for i in range(start + 1, end - 1):
            # Local highs
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                high_points.append((i, highs[i]))
            
            # Local lows
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                low_points.append((i, lows[i]))
        
        # Need at least 3 points for each
        if len(high_points) < 3 or len(low_points) < 3:
            continue
        
        # Check for ascending triangle (flat top, rising bottom)
        if max(point[1] for point in high_points) - min(point[1] for point in high_points) < 0.02 * high_points[0][1]:  # Flat top
            # Check if lows are ascending
            low_values = [point[1] for point in low_points]
            ascending = True
            for i in range(1, len(low_values)):
                if low_values[i] < low_values[i-1]:
                    ascending = False
                    break
            
            if ascending and (low_values[-1] - low_values[0]) / low_values[0] > 0.01:  # At least 1% rise
                # Calculate pattern strength
                height = high_points[0][1] - low_points[0][1]
                width = data.index[end] - data.index[start]
                strength = min(100, (height / low_points[0][1]) * 100)
                
                patterns.append({
                    'name': 'Ascending Triangle',
                    'type': 'Continuation',
                    'timestamp': data.index[end].strftime('%Y-%m-%d'),
                    'strength': f"{strength:.1f}%"
                })
        
        # Check for descending triangle (flat bottom, falling top)
        if max(point[1] for point in low_points) - min(point[1] for point in low_points) < 0.02 * low_points[0][1]:  # Flat bottom
            # Check if highs are descending
            high_values = [point[1] for point in high_points]
            descending = True
            for i in range(1, len(high_values)):
                if high_values[i] > high_values[i-1]:
                    descending = False
                    break
            
            if descending and (high_values[0] - high_values[-1]) / high_values[0] > 0.01:  # At least 1% fall
                # Calculate pattern strength
                height = high_points[0][1] - low_points[0][1]
                width = data.index[end] - data.index[start]
                strength = min(100, (height / high_points[0][1]) * 100)
                
                patterns.append({
                    'name': 'Descending Triangle',
                    'type': 'Continuation',
                    'timestamp': data.index[end].strftime('%Y-%m-%d'),
                    'strength': f"{strength:.1f}%"
                })
        
        # Check for symmetrical triangle (both converging)
        high_values = [point[1] for point in high_points]
        low_values = [point[1] for point in low_points]
        
        high_descending = True
        for i in range(1, len(high_values)):
            if high_values[i] > high_values[i-1]:
                high_descending = False
                break
        
        low_ascending = True
        for i in range(1, len(low_values)):
            if low_values[i] < low_values[i-1]:
                low_ascending = False
                break
        
        if high_descending and low_ascending:
            # Calculate pattern strength
            height_start = high_points[0][1] - low_points[0][1]
            height_end = high_points[-1][1] - low_points[-1][1]
            convergence = (height_start - height_end) / height_start
            strength = min(100, convergence * 200)  # Scale convergence
            
            patterns.append({
                'name': 'Symmetrical Triangle',
                'type': 'Continuation',
                'timestamp': data.index[end].strftime('%Y-%m-%d'),
                'strength': f"{strength:.1f}%"
            })
    
    return patterns

def detect_wedge_patterns(data):
    """Detect wedge patterns (rising, falling)"""
    patterns = []
    
    # Need at least 20 data points for this pattern
    if len(data) < 20:
        return patterns
    
    # Extract prices
    highs = data['High'].values
    lows = data['Low'].values
    
    # Parameters
    min_duration = 15  # Minimum number of days for a valid wedge
    
    # Loop through potential wedges
    for start in range(len(data) - min_duration):
        end = min(start + 40, len(data) - 1)  # Look at most 40 bars ahead
        
        if end - start < min_duration:
            continue
        
        # Find the high points and low points
        high_points = []
        low_points = []
        
        for i in range(start + 1, end - 1):
            # Local highs
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                high_points.append((i, highs[i]))
            
            # Local lows
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                low_points.append((i, lows[i]))
        
        # Need at least 3 points for each
        if len(high_points) < 3 or len(low_points) < 3:
            continue
        
        # Rising Wedge (both lines rising, but converging)
        high_indices = [point[0] for point in high_points]
        high_values = [point[1] for point in high_points]
        low_indices = [point[0] for point in low_points]
        low_values = [point[1] for point in low_points]
        
        # Check if both lines are rising
        highs_rising = high_values[-1] > high_values[0]
        lows_rising = low_values[-1] > low_values[0]
        
        if highs_rising and lows_rising:
            # Calculate slopes
            high_slope = (high_values[-1] - high_values[0]) / (high_indices[-1] - high_indices[0])
            low_slope = (low_values[-1] - low_values[0]) / (low_indices[-1] - low_indices[0])
            
            # In a rising wedge, lower line rises faster than upper line
            if low_slope > high_slope:
                # Calculate pattern strength
                convergence = (high_slope - low_slope) / high_slope
                strength = min(100, abs(convergence) * 200)
                
                patterns.append({
                    'name': 'Rising Wedge',
                    'type': 'Reversal',
                    'timestamp': data.index[end].strftime('%Y-%m-%d'),
                    'strength': f"{strength:.1f}%"
                })
        
        # Falling Wedge (both lines falling, but converging)
        highs_falling = high_values[-1] < high_values[0]
        lows_falling = low_values[-1] < low_values[0]
        
        if highs_falling and lows_falling:
            # Calculate slopes (will be negative for falling lines)
            high_slope = (high_values[-1] - high_values[0]) / (high_indices[-1] - high_indices[0])
            low_slope = (low_values[-1] - low_values[0]) / (low_indices[-1] - low_indices[0])
            
            # In a falling wedge, upper line falls faster than lower line
            if high_slope < low_slope:
                # Calculate pattern strength
                convergence = (low_slope - high_slope) / low_slope
                strength = min(100, abs(convergence) * 200)
                
                patterns.append({
                    'name': 'Falling Wedge',
                    'type': 'Reversal',
                    'timestamp': data.index[end].strftime('%Y-%m-%d'),
                    'strength': f"{strength:.1f}%"
                })
    
    return patterns

def detect_flag_patterns(data):
    """Detect flag and pennant patterns"""
    patterns = []
    
    # Need at least 15 data points for this pattern
    if len(data) < 15:
        return patterns
    
    # Extract prices
    closes = data['Close'].values
    
    # Parameters
    min_pole = 5  # Minimum number of days for the flagpole
    min_flag = 5  # Minimum number of days for the flag
    max_flag = 15  # Maximum number of days for the flag
    
    # Loop through potential patterns
    for i in range(min_pole, len(data) - min_flag):
        # Check for a strong move up (potential bullish flagpole)
        if (closes[i] - closes[i-min_pole]) / closes[i-min_pole] > 0.05:  # At least 5% move
            # Look for a consolidation period (flag)
            flag_start = i
            
            # Find where the consolidation might end
            for j in range(flag_start + min_flag, min(flag_start + max_flag, len(data) - 1)):
                # Calculate the range of prices during the flag
                flag_range = max(closes[flag_start:j+1]) - min(closes[flag_start:j+1])
                flag_avg = sum(closes[flag_start:j+1]) / (j - flag_start + 1)
                
                # Flag should be relatively small compared to the pole
                if flag_range / (closes[i] - closes[i-min_pole]) < 0.5:
                    # Check if the flag prices are trending down slightly
                    flag_slope = (closes[j] - closes[flag_start]) / (j - flag_start)
                    
                    if flag_slope < 0:  # Downward sloping flag
                        # Calculate pattern strength
                        pole_size = (closes[i] - closes[i-min_pole]) / closes[i-min_pole]
                        flag_tightness = 1 - (flag_range / (closes[i] - closes[i-min_pole]))
                        strength = min(100, (pole_size * 10 + flag_tightness) * 50)
                        
                        patterns.append({
                            'name': 'Bullish Flag',
                            'type': 'Continuation',
                            'timestamp': data.index[j].strftime('%Y-%m-%d'),
                            'strength': f"{strength:.1f}%"
                        })
                        break
        
        # Check for a strong move down (potential bearish flagpole)
        if (closes[i-min_pole] - closes[i]) / closes[i-min_pole] > 0.05:  # At least 5% move
            # Look for a consolidation period (flag)
            flag_start = i
            
            # Find where the consolidation might end
            for j in range(flag_start + min_flag, min(flag_start + max_flag, len(data) - 1)):
                # Calculate the range of prices during the flag
                flag_range = max(closes[flag_start:j+1]) - min(closes[flag_start:j+1])
                flag_avg = sum(closes[flag_start:j+1]) / (j - flag_start + 1)
                
                # Flag should be relatively small compared to the pole
                if flag_range / (closes[i-min_pole] - closes[i]) < 0.5:
                    # Check if the flag prices are trending up slightly
                    flag_slope = (closes[j] - closes[flag_start]) / (j - flag_start)
                    
                    if flag_slope > 0:  # Upward sloping flag
                        # Calculate pattern strength
                        pole_size = (closes[i-min_pole] - closes[i]) / closes[i-min_pole]
                        flag_tightness = 1 - (flag_range / (closes[i-min_pole] - closes[i]))
                        strength = min(100, (pole_size * 10 + flag_tightness) * 50)
                        
                        patterns.append({
                            'name': 'Bearish Flag',
                            'type': 'Continuation',
                            'timestamp': data.index[j].strftime('%Y-%m-%d'),
                            'strength': f"{strength:.1f}%"
                        })
                        break
    
    return patterns
