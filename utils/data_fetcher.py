import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_stock_data(symbol, period="1mo", interval="1d"):
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        pd.DataFrame: OHLCV data for the specified stock
    """
    try:
        # Validate input parameters
        if symbol is None or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Stock symbol cannot be empty or None")
            
        # Use strip to remove any whitespace and upper to standardize
        symbol = symbol.strip().upper()
        
        # Check if valid interval for the period
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'] and period in ['1y', '2y', '5y', '10y', 'max']:
            # For intraday data, can't fetch too far back, adjust period
            period = '7d'
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol} with period {period} and interval {interval}")
        
        return data
    except Exception as e:
        raise Exception(f"Error fetching stock data: {str(e)}")

def fetch_crypto_data(symbol, period="1mo", interval="1d"):
    """
    Fetch cryptocurrency data from Yahoo Finance.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC-USD')
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        pd.DataFrame: OHLCV data for the specified cryptocurrency
    """
    try:
        # Validate input parameters
        if symbol is None or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("Cryptocurrency symbol cannot be empty or None")
            
        # Use strip to remove any whitespace
        symbol = symbol.strip()
            
        # Ensure symbol has -USD suffix
        if not (symbol.upper().endswith('-USD') or symbol.upper().endswith('-USDT') or symbol.upper().endswith('-USDC')):
            symbol = f"{symbol}-USD"
        
        # Check if valid interval for the period
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'] and period in ['1y', '2y', '5y', '10y', 'max']:
            # For intraday data, can't fetch too far back, adjust period
            period = '7d'
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol} with period {period} and interval {interval}")
        
        return data
    except Exception as e:
        raise Exception(f"Error fetching cryptocurrency data: {str(e)}")

def get_available_stocks():
    """
    Returns a list of popular stock symbols and names.
    
    Returns:
        list: List of stock symbols and names
    """
    # List of popular stocks by market cap (major indices + popular tech, finance, etc)
    stocks = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'XOM', 'CSCO', 'DIS', 'NFLX',
        'ADBE', 'CMCSA', 'PFE', 'VZ', 'KO', 'PEP', 'T', 'MRK', 'INTC', 'CVX',
        'ABT', 'CRM', 'PYPL', 'NKE', 'TMO', 'MCD', 'ABBV', 'COST', 'ACN', 'AVGO',
        'DHR', 'LLY', 'TXN', 'NEE', 'MDT', 'UNP', 'BMY', 'PM', 'QCOM', 'HON',
        'UPS', 'IBM', 'SBUX', 'LIN', 'AMT', 'ORCL', 'BA', 'MMM', 'RTX', 'GE',
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VGT', 'XLK', 'XLF', 'XLE'
    ]
    
    return stocks

def get_available_cryptos():
    """
    Returns a list of popular cryptocurrency symbols.
    
    Returns:
        list: List of cryptocurrency symbols
    """
    # List of popular cryptocurrencies
    cryptos = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'DOT-USD',
        'MATIC-USD', 'LINK-USD', 'LTC-USD', 'BCH-USD', 'XLM-USD', 'UNI-USD', 'ATOM-USD',
        'AVAX-USD', 'FTM-USD', 'AAVE-USD', 'ALGO-USD', 'ETC-USD', 'MANA-USD',
        'SHIB-USD', 'NEAR-USD', 'ICP-USD', 'FIL-USD', 'HBAR-USD', 'VET-USD',
        'SAND-USD', 'AXS-USD', 'THETA-USD', 'EOS-USD', 'EGLD-USD', 'NEO-USD',
        'KSM-USD', 'RUNE-USD', 'ONE-USD', 'GALA-USD', 'ENJ-USD', 'XTZ-USD',
        'FLOW-USD', 'CHZ-USD', 'BAT-USD', 'ZEC-USD', 'DASH-USD', 'CAKE-USD'
    ]
    
    return cryptos
