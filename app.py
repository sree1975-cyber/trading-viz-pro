import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import io

from utils.data_fetcher import fetch_stock_data, fetch_crypto_data, get_available_stocks, get_available_cryptos
from utils.technical_analysis import calculate_indicators, available_indicators
from utils.chart_utils import create_candlestick_chart, add_indicator_to_chart, add_volume_to_chart
from utils.patterns import detect_patterns, pattern_descriptions
from utils.prediction import (
    get_price_predictions,
    get_available_prediction_methods,
    get_prediction_method_descriptions
)

# Page configuration
st.set_page_config(
    page_title="TradingViz Pro",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []
if 'asset_data' not in st.session_state:
    st.session_state['asset_data'] = None
if 'selected_asset' not in st.session_state:
    st.session_state['selected_asset'] = None

# Title and description
st.title("TradingViz Pro")
st.markdown("An advanced trading tool with technical analysis, interactive charts, and market insights")

# Sidebar for asset selection and timeframe
with st.sidebar:
    st.header("Settings")
    
    # Asset Type Selection
    asset_type = st.selectbox("Select Asset Type", ["Stocks", "Cryptocurrencies"])
    
    # Asset Selection
    if asset_type == "Stocks":
        available_assets = get_available_stocks()
        
        # Option tabs for symbol selection
        symbol_option = st.radio("Select symbol option:", ["Choose from popular stocks", "Enter custom symbol"])
        
        if symbol_option == "Enter custom symbol":
            custom_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", 
                                         st.session_state.get('custom_symbol', ""))
            if custom_symbol:
                # Store directly entered symbol in session state for later use
                selected_asset = custom_symbol.upper().strip()
                st.session_state['custom_symbol'] = selected_asset
            elif 'custom_symbol' in st.session_state:
                selected_asset = st.session_state['custom_symbol']
            else:
                selected_asset = "AAPL"  # Default to Apple if nothing entered
        else:
            # Show popular stocks in categorized selectbox
            popular_categories = {
                "Tech Giants": ["AAPL - Apple Inc.", "MSFT - Microsoft Corp.", "GOOGL - Alphabet Inc.", "AMZN - Amazon.com Inc.", "META - Meta Platforms Inc."],
                "Financial": ["JPM - JPMorgan Chase & Co.", "BAC - Bank of America Corp.", "V - Visa Inc.", "MA - Mastercard Inc."],
                "Healthcare": ["JNJ - Johnson & Johnson", "PFE - Pfizer Inc.", "UNH - UnitedHealth Group Inc."],
                "Consumer": ["KO - Coca-Cola Co.", "PEP - PepsiCo Inc.", "MCD - McDonald's Corp.", "SBUX - Starbucks Corp."],
                "Industrial": ["BA - Boeing Co.", "CAT - Caterpillar Inc.", "GE - General Electric Co."],
                "Energy": ["XOM - Exxon Mobil Corp.", "CVX - Chevron Corp."]
            }
            
            category = st.selectbox("Select category", list(popular_categories.keys()))
            selected_asset = st.selectbox("Select Stock", popular_categories[category], index=0)
        
        # Always display the final symbol in a visible way
        symbol_display = selected_asset.split(" - ")[0] if " - " in selected_asset else selected_asset
        st.sidebar.markdown(f"**Selected Symbol:** {symbol_display}")
        # Store the clean symbol for later use
        st.session_state['clean_symbol'] = symbol_display
    else:  # Cryptocurrencies
        available_assets = get_available_cryptos()
        
        # Option tabs for symbol selection
        symbol_option = st.radio("Select symbol option:", ["Choose from popular cryptocurrencies", "Enter custom symbol"])
        
        if symbol_option == "Enter custom symbol":
            custom_symbol = st.text_input("Enter Crypto Symbol (e.g., BTC-USD)", 
                                          st.session_state.get('custom_crypto', ""))
            if custom_symbol:
                # Store directly entered symbol in session state for later use
                selected_asset = custom_symbol.upper().strip()
                st.session_state['custom_crypto'] = selected_asset
            elif 'custom_crypto' in st.session_state:
                selected_asset = st.session_state['custom_crypto']
            else:
                selected_asset = "BTC-USD"  # Default to Bitcoin if nothing entered
        else:
            # Show popular cryptocurrencies in categorized selectbox
            popular_cryptos = {
                "Major Coins": ["BTC-USD - Bitcoin", "ETH-USD - Ethereum", "BNB-USD - Binance Coin", "XRP-USD - Ripple"],
                "Altcoins": ["SOL-USD - Solana", "ADA-USD - Cardano", "DOGE-USD - Dogecoin", "DOT-USD - Polkadot"],
                "Stablecoins": ["USDT-USD - Tether", "USDC-USD - USD Coin", "DAI-USD - Dai"],
                "DeFi": ["UNI-USD - Uniswap", "AAVE-USD - Aave", "CAKE-USD - PancakeSwap"],
                "Metaverse": ["MANA-USD - Decentraland", "SAND-USD - The Sandbox", "AXS-USD - Axie Infinity"]
            }
            
            category = st.selectbox("Select category", list(popular_cryptos.keys()))
            selected_asset = st.selectbox("Select Cryptocurrency", popular_cryptos[category], index=0)
        
        # Always display the final symbol in a visible way
        symbol_display = selected_asset.split(" - ")[0] if " - " in selected_asset else selected_asset
        st.sidebar.markdown(f"**Selected Symbol:** {symbol_display}")
        # Store the clean symbol for later use
        st.session_state['clean_symbol'] = symbol_display
    
    # Timeframe Selection
    timeframe_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
    timeframe = st.selectbox(
        "Select Timeframe",
        timeframe_options
    )
    
    # Interval Selection - dynamically filtered based on timeframe
    if timeframe in ["1d", "5d"]:
        # Short timeframes - allow minute intervals
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"]
        interval_default = "15m" if timeframe == "1d" else "30m"
        interval_help = "For 1-5 day timeframes, minute-level data is available"
    elif timeframe in ["1mo"]:
        # Medium timeframes - hourly or daily
        interval_options = ["1h", "1d", "5d", "1wk"]
        interval_default = "1h"
        interval_help = "For 1 month timeframe, hourly or daily intervals work best"
    elif timeframe in ["3mo"]:
        # Medium timeframes - daily 
        interval_options = ["1d", "5d", "1wk"]
        interval_default = "1d"
        interval_help = "For 3 month timeframe, daily intervals work best"
    else:
        # Long timeframes - only daily or longer intervals
        interval_options = ["1d", "5d", "1wk", "1mo"]
        interval_default = "1d"
        interval_help = "For longer timeframes (6mo+), minute/hour data isn't available"
    
    # Find the default index
    try:
        default_idx = interval_options.index(interval_default)
    except ValueError:
        default_idx = 0
        
    interval = st.selectbox(
        "Select Interval", 
        interval_options, 
        index=default_idx,
        help=interval_help
    )
    
    # Info message about timeframe/interval relationship
    timeframe_pairs = {
        "1d": "Use 1m-90m intervals for intraday data",
        "5d": "Use 5m-1h intervals for detailed analysis",
        "1mo": "Use 1h or 1d intervals for best results",
        "3mo": "Use 1d intervals (hourly data may be incomplete)",
        "6mo": "Use 1d or 5d intervals (minute/hour data unavailable)",
        "1y+": "Use 1d, 5d, 1wk intervals (minute/hour data unavailable)"
    }
    
    # Show appropriate help message based on timeframe 
    if timeframe in ["1d", "5d", "1mo", "3mo", "6mo"]:
        help_msg = timeframe_pairs[timeframe]
    else:
        help_msg = timeframe_pairs["1y+"]
        
    st.info(help_msg)
    
    # Use the clean symbol stored in session state
    symbol = st.session_state.get('clean_symbol', selected_asset)
    if not symbol:
        symbol = selected_asset.split(" - ")[0] if selected_asset and " - " in selected_asset else selected_asset
    
    # Store the currently selected display name (for UI)
    display_name = selected_asset if " - " in selected_asset else symbol
    
    # Refresh button
    if st.button("Fetch Data"):
        # Validate that we have a valid symbol
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            st.sidebar.error("Please select a valid asset before fetching data")
        else:
            # Store both the display name and the actual symbol
            st.session_state['selected_asset_display'] = display_name
            st.session_state['selected_asset'] = symbol
            
            with st.spinner(f"Fetching data for {symbol}..."):
                try:
                    if asset_type == "Stocks":
                        data = fetch_stock_data(symbol, timeframe, interval)
                    else:
                        data = fetch_crypto_data(symbol, timeframe, interval)
                    
                    st.session_state['asset_data'] = data
                    st.session_state['asset_type'] = asset_type
                    st.success(f"Data for {symbol} fetched successfully!")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    # Watchlist Management
    st.header("Watchlist")
    
    # Add to watchlist
    if st.button("Add to Watchlist"):
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            st.sidebar.error("Please select a valid asset to add to watchlist")
        elif selected_asset not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(selected_asset)
            st.sidebar.success(f"Added {selected_asset} to watchlist")
        else:
            st.sidebar.warning(f"{selected_asset} is already in your watchlist")
    
    # Show watchlist
    if st.session_state['watchlist']:
        for idx, asset in enumerate(st.session_state['watchlist']):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{idx+1}. {asset}")
            with col2:
                if st.button("×", key=f"remove_{idx}"):
                    st.session_state['watchlist'].remove(asset)
                    st.rerun()
    else:
        st.info("Your watchlist is empty")

# Main content area
if st.session_state['asset_data'] is not None and st.session_state['selected_asset'] is not None:
    data = st.session_state['asset_data']
    asset = st.session_state['selected_asset']
    asset_type = st.session_state['asset_type']
    
    # Display current price and change
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - previous_price
        price_change_percent = (price_change / previous_price) * 100
        
        # Get the latest date
        latest_date = data.index[-1]
        date_str = latest_date.strftime("%Y-%m-%d")
        # If we have time data (for intraday charts)
        if hasattr(latest_date, 'hour') and (latest_date.hour != 0 or latest_date.minute != 0):
            date_str = latest_date.strftime("%Y-%m-%d %H:%M")
        
        # Price metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"Current Price ({asset})",
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_percent:.2f}%)"
            )
        with col2:
            st.metric(
                label="Daily High",
                value=f"${data['High'].iloc[-1]:.2f}"
            )
        with col3:
            st.metric(
                label="Daily Low",
                value=f"${data['Low'].iloc[-1]:.2f}"
            )
            
        # Display latest date below the metrics
        st.caption(f"**Last updated:** {date_str}")
        
        # Technical Analysis Indicators Selection
        st.header("Technical Analysis")
        indicators_tab, patterns_tab, performance_tab, predictions_tab = st.tabs(["Indicators", "Patterns", "Performance", "Price Predictions"])
        
        with indicators_tab:
            selected_indicators = st.multiselect(
                "Select Technical Indicators",
                list(available_indicators.keys())
            )
            
            # Calculate indicators
            if selected_indicators:
                indicator_data = calculate_indicators(data, selected_indicators)
                
                # Create candlestick chart with selected indicators
                fig = create_candlestick_chart(data, asset)
                
                # Add volume
                fig = add_volume_to_chart(fig, data)
                
                # Add indicators to chart
                for indicator in selected_indicators:
                    fig = add_indicator_to_chart(fig, indicator_data, indicator, available_indicators[indicator])
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Store indicator data in session state to prevent errors when switching tabs
                st.session_state['indicator_data'] = indicator_data
                
                # Display indicator data
                for indicator in selected_indicators:
                    with st.expander(f"{indicator} Data"):
                        try:
                            # Special handling for indicators with multiple outputs and correct dictionary structure
                            if indicator == "SMA" and indicator in indicator_data:
                                for period, series in indicator_data[indicator].items():
                                    st.write(f"SMA {period}:")
                                    st.write(series.dropna().tail())
                            elif indicator == "EMA" and indicator in indicator_data:
                                for period, series in indicator_data[indicator].items():
                                    st.write(f"EMA {period}:")
                                    st.write(series.dropna().tail())
                            elif indicator in ["MACD", "Bollinger Bands", "Stochastic Oscillator", "ADX", "Ichimoku Cloud"] and indicator in indicator_data:
                                if "Error" in indicator_data[indicator]:
                                    st.error(indicator_data[indicator]["Error"])
                                else:
                                    for component, series in indicator_data[indicator].items():
                                        st.write(f"{component}:")
                                        st.write(series.dropna().tail())
                            elif indicator in indicator_data:
                                # Standard single-output indicators
                                st.write(indicator_data[indicator].dropna().tail())
                            else:
                                st.error(f"Data for {indicator} not available")
                        except Exception as e:
                            st.error(f"Error displaying {indicator} data: {str(e)}")
            else:
                # Just display price chart if no indicators are selected
                fig = create_candlestick_chart(data, asset)
                fig = add_volume_to_chart(fig, data)
                st.plotly_chart(fig, use_container_width=True)
        
        with patterns_tab:
            # Pattern Recognition
            st.subheader("Pattern Recognition")
            patterns_found = detect_patterns(data)
            
            if patterns_found:
                st.success(f"Found {len(patterns_found)} patterns")
                for pattern in patterns_found:
                    with st.expander(f"{pattern['name']} ({pattern['timestamp']})"):
                        st.write(f"**Type:** {pattern['type']}")
                        st.write(f"**Strength:** {pattern['strength']}")
                        st.write(f"**Description:** {pattern_descriptions.get(pattern['name'], 'No description available')}")
            else:
                st.info("No significant patterns detected in the current timeframe")
        
        with performance_tab:
            # Performance Metrics
            st.subheader("Performance Metrics")
            
            # Calculate performance metrics
            returns = data['Close'].pct_change().dropna()
            daily_returns = returns.mean() * 100
            volatility = returns.std() * 100
            sharpe_ratio = (daily_returns / volatility) * np.sqrt(252) if volatility != 0 else 0
            max_drawdown = (data['Close'] / data['Close'].cummax() - 1.0).min() * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Daily Return", f"{daily_returns:.2f}%")
                st.metric("Volatility", f"{volatility:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
            
            # Returns over time
            st.subheader("Cumulative Returns")
            cumulative_returns = (1 + returns).cumprod() - 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values * 100,
                mode='lines',
                line=dict(color='green' if cumulative_returns.iloc[-1] >= 0 else 'red'),
                name='Cumulative Returns (%)'
            ))
            fig.update_layout(
                title="Cumulative Returns Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Returns (%)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with predictions_tab:
            st.subheader("Price Predictions")
            
            # Description of price prediction
            st.markdown("""
            This section uses various prediction models to forecast future price movements. These predictions
            are based on historical data patterns and should be used for educational purposes only.
            """)
            
            st.warning("Price predictions are estimates only and not financial advice. Past performance does not guarantee future results.")
            
            # Prediction settings
            forecast_periods = st.slider("Forecast Periods (Days)", min_value=7, max_value=60, value=30)
            
            prediction_methods = get_available_prediction_methods()
            method_descriptions = get_prediction_method_descriptions()
            
            selected_methods = st.multiselect(
                "Select Prediction Methods",
                prediction_methods,
                default=[prediction_methods[0]]  # Default to first method (ARIMA)
            )
            
            if selected_methods:
                # Show descriptions of selected methods
                for method in selected_methods:
                    with st.expander(f"About {method}"):
                        st.write(method_descriptions[method])
                
                # Generate predictions button
                if st.button("Generate Predictions"):
                    with st.spinner("Generating price predictions... This may take a few minutes"):
                        # Map streamlit-friendly method names to the internal method names
                        method_mapping = {
                            "Moving Average (Simple)": "ma",
                            "Linear Trend (Simple)": "trend"
                        }
                        
                        # Get predictions
                        try:
                            predictions = get_price_predictions(data, forecast_periods=forecast_periods)
                            
                            if 'error' in predictions:
                                st.error(predictions['error'])
                            else:
                                # Create forecast visualization
                                fig = go.Figure()
                                
                                # Add historical data
                                fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data['Close'].values,
                                    mode='lines',
                                    name='Historical Data',
                                    line=dict(color='blue')
                                ))
                                
                                # Add predictions for selected methods
                                colors = ['red', 'green', 'purple', 'orange', 'brown']
                                used_colors = []
                                
                                for i, method_name in enumerate(selected_methods):
                                    internal_name = method_mapping[method_name]
                                    
                                    if internal_name in predictions and predictions[internal_name]['predictions'] is not None:
                                        color = colors[i % len(colors)]
                                        used_colors.append(color)
                                        
                                        pred_series = predictions[internal_name]['predictions']
                                        
                                        fig.add_trace(go.Scatter(
                                            x=pred_series.index,
                                            y=pred_series.values,
                                            mode='lines',
                                            name=f"{method_name} Forecast",
                                            line=dict(color=color, dash='dash')
                                        ))
                                
                                # Layout
                                fig.update_layout(
                                    title=f"{forecast_periods}-Day Price Forecast for {asset}",
                                    xaxis_title="Date",
                                    yaxis_title="Price ($)",
                                    hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display prediction metrics
                                st.subheader("Model Performance Metrics")
                                
                                metrics_data = []
                                for method_name in selected_methods:
                                    internal_name = method_mapping[method_name]
                                    
                                    if internal_name in predictions and predictions[internal_name]['predictions'] is not None:
                                        pred_info = predictions[internal_name]
                                        
                                        # Get relevant metrics
                                        metrics = {}
                                        metrics['Model'] = method_name
                                        metrics['Last Known Price'] = f"${data['Close'].iloc[-1]:.2f}"
                                        metrics['Forecast (End of Period)'] = f"${pred_info['predictions'].iloc[-1]:.2f}"
                                        
                                        change = ((pred_info['predictions'].iloc[-1] / data['Close'].iloc[-1]) - 1) * 100
                                        metrics['Predicted Change'] = f"{change:.2f}%"
                                        
                                        # Add any model-specific metrics
                                        if 'mse' in pred_info:
                                            metrics['MSE'] = f"{pred_info['mse']:.4f}"
                                        if 'r2' in pred_info:
                                            metrics['R²'] = f"{pred_info['r2']:.4f}"
                                        if 'aic' in pred_info:
                                            metrics['AIC'] = f"{pred_info['aic']:.2f}"
                                        
                                        metrics_data.append(metrics)
                                
                                if metrics_data:
                                    # Convert to DataFrame for nice display
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.table(metrics_df)
                                    
                                    st.info("""
                                    **Model information:**
                                    - **Moving Average**: Uses historical relationship between price and its moving average.
                                    - **Linear Trend**: Fits a straight line to recent price movements to project future direction.
                                    """)
                                
                                # Disclaimer
                                st.caption("""
                                DISCLAIMER: These predictions are based on historical data and mathematical models.
                                Financial markets are influenced by many factors not captured in these models. 
                                Do not make investment decisions based solely on these forecasts.
                                """)
                        except Exception as e:
                            st.error(f"Error generating predictions: {str(e)}")
                            st.info("Try using a longer timeframe to get more training data, or select different prediction methods.")
                        
            else:
                st.info("Please select at least one prediction method to generate forecasts.")
        
        # Data Export Section
        st.header("Export Data")
        export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button("Export Data"):
            with st.spinner("Preparing export..."):
                if export_format == "CSV":
                    buffer = io.StringIO()
                    # Create a copy to avoid modifying the original data
                    export_data = data.copy()
                    # Convert timezone-aware datetimes to timezone-naive for consistency
                    if isinstance(export_data.index, pd.DatetimeIndex) and export_data.index.tz is not None:
                        export_data.index = export_data.index.tz_localize(None)
                    export_data.to_csv(buffer, index=True)
                    buffer.seek(0)
                    st.download_button(
                        label="Download CSV",
                        data=buffer.getvalue(),
                        file_name=f"{asset}_data.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    # Convert timezone-aware datetimes to timezone-naive
                    export_data = data.copy()
                    # Check if index is DatetimeIndex and has timezone info
                    if isinstance(export_data.index, pd.DatetimeIndex) and export_data.index.tz is not None:
                        export_data.index = export_data.index.tz_localize(None)
                    export_data.to_excel(buffer, index=True)
                    buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name=f"{asset}_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:  # JSON
                    buffer = io.StringIO()
                    # Create a copy to avoid modifying the original data
                    export_data = data.copy()
                    # Convert timezone-aware datetimes to timezone-naive for consistency
                    if isinstance(export_data.index, pd.DatetimeIndex) and export_data.index.tz is not None:
                        export_data.index = export_data.index.tz_localize(None)
                    export_data.to_json(buffer, orient="index", date_format="iso")
                    buffer.seek(0)
                    st.download_button(
                        label="Download JSON",
                        data=buffer.getvalue(),
                        file_name=f"{asset}_data.json",
                        mime="application/json"
                    )
else:
    # Initial state or no data
    st.info("👈 Select an asset and fetch data to begin your analysis")
    
    # Show demo charts
    st.header("Example Charts")
    
    # Create a sample dataset for demonstration
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.normal(100, 5, len(dates)),
        'High': np.random.normal(105, 5, len(dates)),
        'Low': np.random.normal(95, 5, len(dates)),
        'Close': np.random.normal(100, 5, len(dates)),
        'Volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    
    # Ensure High is the highest, Low is the lowest
    for i in range(len(sample_data)):
        values = [sample_data.iloc[i]['Open'], sample_data.iloc[i]['Close']]
        sample_data.iloc[i, sample_data.columns.get_loc('High')] = max(values) + abs(np.random.normal(3, 1))
        sample_data.iloc[i, sample_data.columns.get_loc('Low')] = min(values) - abs(np.random.normal(3, 1))
    
    fig = create_candlestick_chart(sample_data, "Example")
    fig = add_volume_to_chart(fig, sample_data)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("This is a sample chart for demonstration. Select an asset from the sidebar to view real data.")

# Footer
st.markdown("---")

# Application footer
st.markdown("TradingViz Pro - Advanced Technical Analysis Tool")
