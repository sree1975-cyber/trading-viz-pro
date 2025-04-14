import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_candlestick_chart(data, asset_name):
    """
    Create a basic candlestick chart.
    
    Args:
        data (pd.DataFrame): OHLCV data
        asset_name (str): Name of the asset to display in the title
        
    Returns:
        plotly.graph_objects.Figure: Candlestick chart
    """
    # Create figure with secondary y-axis for indicators
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{asset_name} Price", "Volume"),
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=asset_name,
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{asset_name} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def add_volume_to_chart(fig, data):
    """
    Add volume bars to the chart.
    
    Args:
        fig (plotly.graph_objects.Figure): Existing chart
        data (pd.DataFrame): OHLCV data
        
    Returns:
        plotly.graph_objects.Figure: Chart with volume added
    """
    # Colors for volume bars based on price movement
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker=dict(color=colors),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    return fig

def add_indicator_to_chart(fig, indicator_data, indicator_name, indicator_params):
    """
    Add technical indicator to the chart.
    
    Args:
        fig (plotly.graph_objects.Figure): Existing chart
        indicator_data (dict): Dictionary containing indicator data
        indicator_name (str): Name of the indicator
        indicator_params (dict): Parameters for the indicator
        
    Returns:
        plotly.graph_objects.Figure: Chart with indicator added
    """
    try:
        if indicator_name == "SMA":
            # Check if we have SMA data in the new nested dict format
            if "SMA" in indicator_data and isinstance(indicator_data["SMA"], dict):
                # New format (nested dictionary)
                for period, series in indicator_data["SMA"].items():
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series,
                            mode="lines",
                            name=f"SMA {period}",
                            line=dict(width=1, dash="dot")
                        ),
                        row=1, col=1
                    )
            # Fallback to old format if needed
            else:
                for period in indicator_params["periods"]:
                    if f"SMA_{period}" in indicator_data:
                        fig.add_trace(
                            go.Scatter(
                                x=indicator_data[f"SMA_{period}"].index,
                                y=indicator_data[f"SMA_{period}"],
                                mode="lines",
                                name=f"SMA {period}",
                                line=dict(width=1, dash="dot")
                            ),
                            row=1, col=1
                        )
                
        elif indicator_name == "EMA":
            # Check if we have EMA data in the new nested dict format
            if "EMA" in indicator_data and isinstance(indicator_data["EMA"], dict):
                # New format (nested dictionary)
                for period, series in indicator_data["EMA"].items():
                    fig.add_trace(
                        go.Scatter(
                            x=series.index,
                            y=series,
                            mode="lines",
                            name=f"EMA {period}",
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
            # Fallback to old format if needed
            else:
                for period in indicator_params["periods"]:
                    if f"EMA_{period}" in indicator_data:
                        fig.add_trace(
                            go.Scatter(
                                x=indicator_data[f"EMA_{period}"].index,
                                y=indicator_data[f"EMA_{period}"],
                                mode="lines",
                                name=f"EMA {period}",
                                line=dict(width=1)
                            ),
                            row=1, col=1
                        )
                
        elif indicator_name == "RSI":
            # Create new row for RSI
            try:
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["RSI"].index,
                        y=indicator_data["RSI"],
                        mode="lines",
                        name="RSI",
                        line=dict(color="purple")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                # Add overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", secondary_y=True, row=1, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", secondary_y=True, row=1, col=1)
                
                # Update y-axis
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1, secondary_y=True)
            except Exception as e:
                # Alternative approach without secondary_y
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["RSI"].index,
                        y=indicator_data["RSI"],
                        mode="lines",
                        name="RSI",
                        line=dict(color="purple", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
        elif indicator_name == "MACD":
            try:
                # Add MACD, Signal, and Histogram
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["MACD"]["MACD"].index,
                        y=indicator_data["MACD"]["MACD"],
                        mode="lines",
                        name="MACD",
                        line=dict(color="blue")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["MACD"]["Signal"].index,
                        y=indicator_data["MACD"]["Signal"],
                        mode="lines",
                        name="Signal",
                        line=dict(color="red")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                # Update y-axis
                fig.update_yaxes(title_text="MACD", row=1, col=1, secondary_y=True)
            except Exception as e:
                # Alternative approach without secondary_y
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["MACD"]["MACD"].index,
                        y=indicator_data["MACD"]["MACD"],
                        mode="lines",
                        name="MACD",
                        line=dict(color="blue", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["MACD"]["Signal"].index,
                        y=indicator_data["MACD"]["Signal"],
                        mode="lines",
                        name="Signal",
                        line=dict(color="red", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
        elif indicator_name == "Bollinger Bands":
            # Add Upper, Middle, and Lower Bands
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Bollinger Bands"]["Upper"].index,
                    y=indicator_data["Bollinger Bands"]["Upper"],
                    mode="lines",
                    name="Upper Band",
                    line=dict(color="rgba(250, 0, 0, 0.5)", width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Bollinger Bands"]["Middle"].index,
                    y=indicator_data["Bollinger Bands"]["Middle"],
                    mode="lines",
                    name="Middle Band",
                    line=dict(color="rgba(0, 0, 250, 0.5)", width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Bollinger Bands"]["Lower"].index,
                    y=indicator_data["Bollinger Bands"]["Lower"],
                    mode="lines",
                    name="Lower Band",
                    line=dict(color="rgba(0, 250, 0, 0.5)", width=1),
                    fill='tonexty', 
                    fillcolor='rgba(200, 200, 200, 0.2)'
                ),
                row=1, col=1
            )
                
        elif indicator_name == "Stochastic Oscillator":
            try:
                # Add Stochastic K and D lines
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["Stochastic Oscillator"]["K"].index,
                        y=indicator_data["Stochastic Oscillator"]["K"],
                        mode="lines",
                        name="%K",
                        line=dict(color="blue")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["Stochastic Oscillator"]["D"].index,
                        y=indicator_data["Stochastic Oscillator"]["D"],
                        mode="lines",
                        name="%D",
                        line=dict(color="red")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                # Add overbought/oversold lines
                fig.add_hline(y=80, line_dash="dash", line_color="red", secondary_y=True, row=1, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", secondary_y=True, row=1, col=1)
                
                # Update y-axis
                fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=1, col=1, secondary_y=True)
            except Exception as e:
                # Alternative approach without secondary_y
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["Stochastic Oscillator"]["K"].index,
                        y=indicator_data["Stochastic Oscillator"]["K"],
                        mode="lines",
                        name="%K",
                        line=dict(color="blue", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["Stochastic Oscillator"]["D"].index,
                        y=indicator_data["Stochastic Oscillator"]["D"],
                        mode="lines",
                        name="%D",
                        line=dict(color="red", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
        elif indicator_name == "ATR":
            try:
                # Add ATR line
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ATR"].index,
                        y=indicator_data["ATR"],
                        mode="lines",
                        name="ATR",
                        line=dict(color="purple")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                # Update y-axis
                fig.update_yaxes(title_text="ATR", row=1, col=1, secondary_y=True)
            except Exception as e:
                # Alternative approach without secondary_y
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ATR"].index,
                        y=indicator_data["ATR"],
                        mode="lines",
                        name="ATR",
                        line=dict(color="purple", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
        elif indicator_name == "OBV":
            # Check if OBV data exists and is valid
            if indicator_data.get("OBV") is not None and not indicator_data["OBV"].empty:
                try:
                    # Make sure we have a secondary y-axis for the volume pane
                    # Add OBV line to volume pane
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_data["OBV"].index,
                            y=indicator_data["OBV"],
                            mode="lines",
                            name="OBV",
                            line=dict(color="black")
                        ),
                        row=2, col=1
                    )
                    
                    # Update y-axis - use secondary_y=False since we're on the main axis in row 2
                    fig.update_yaxes(title_text="Volume/OBV", row=2, col=1)
                except Exception as e:
                    # If any error, add simple trace without secondary_y
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_data["OBV"].index,
                            y=indicator_data["OBV"],
                            mode="lines",
                            name="OBV",
                            line=dict(color="black", width=1, dash="dot")
                        ),
                        row=1, col=1
                    )
            else:
                # If OBV data is missing, just add a simple annotation
                fig.add_annotation(
                    x=0.5,  # Middle of the chart (0-1 scale)
                    y=0.5,  # Middle of the chart vertically (0-1 scale)
                    text="OBV data not available",
                    showarrow=False,
                    font=dict(color="red", size=12),
                    xref="paper",
                    yref="paper"
                )
                
        elif indicator_name == "Ichimoku Cloud":
            # Add Tenkan-sen (Conversion Line)
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Ichimoku Cloud"]["Tenkan"].index,
                    y=indicator_data["Ichimoku Cloud"]["Tenkan"],
                    mode="lines",
                    name="Conversion Line",
                    line=dict(color="blue", width=1)
                ),
                row=1, col=1
            )
            
            # Add Kijun-sen (Base Line)
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Ichimoku Cloud"]["Kijun"].index,
                    y=indicator_data["Ichimoku Cloud"]["Kijun"],
                    mode="lines",
                    name="Base Line",
                    line=dict(color="red", width=1)
                ),
                row=1, col=1
            )
            
            # Add Senkou Span A (Leading Span A)
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Ichimoku Cloud"]["Senkou A"].index,
                    y=indicator_data["Ichimoku Cloud"]["Senkou A"],
                    mode="lines",
                    name="Leading Span A",
                    line=dict(color="green", width=1)
                ),
                row=1, col=1
            )
            
            # Add Senkou Span B (Leading Span B) with fill for cloud
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Ichimoku Cloud"]["Senkou B"].index,
                    y=indicator_data["Ichimoku Cloud"]["Senkou B"],
                    mode="lines",
                    name="Leading Span B",
                    line=dict(color="red", width=1),
                    fill='tonexty',
                    fillcolor='rgba(0, 250, 0, 0.1)'
                ),
                row=1, col=1
            )
                
        elif indicator_name == "Parabolic SAR":
            # Add Parabolic SAR points
            fig.add_trace(
                go.Scatter(
                    x=indicator_data["Parabolic SAR"].index,
                    y=indicator_data["Parabolic SAR"],
                    mode="markers",
                    name="Parabolic SAR",
                    marker=dict(color="black", size=4, symbol="diamond")
                ),
                row=1, col=1
            )
                
        elif indicator_name == "ADX":
            try:
                # Add ADX, +DI, and -DI lines
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ADX"]["ADX"].index,
                        y=indicator_data["ADX"]["ADX"],
                        mode="lines",
                        name="ADX",
                        line=dict(color="black")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ADX"]["+DI"].index,
                        y=indicator_data["ADX"]["+DI"],
                        mode="lines",
                        name="+DI",
                        line=dict(color="green")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ADX"]["-DI"].index,
                        y=indicator_data["ADX"]["-DI"],
                        mode="lines",
                        name="-DI",
                        line=dict(color="red")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                # Add strong trend line
                fig.add_hline(y=25, line_dash="dash", line_color="gray", secondary_y=True, row=1, col=1)
                
                # Update y-axis
                fig.update_yaxes(title_text="ADX", range=[0, 100], row=1, col=1, secondary_y=True)
            except Exception as e:
                # Alternative approach without secondary_y
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ADX"]["ADX"].index,
                        y=indicator_data["ADX"]["ADX"],
                        mode="lines",
                        name="ADX",
                        line=dict(color="black", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ADX"]["+DI"].index,
                        y=indicator_data["ADX"]["+DI"],
                        mode="lines",
                        name="+DI",
                        line=dict(color="green", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["ADX"]["-DI"].index,
                        y=indicator_data["ADX"]["-DI"],
                        mode="lines",
                        name="-DI",
                        line=dict(color="red", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
                
        elif indicator_name == "CCI":
            try:
                # Add CCI line
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["CCI"].index,
                        y=indicator_data["CCI"],
                        mode="lines",
                        name="CCI",
                        line=dict(color="purple")
                    ),
                    row=1, col=1,
                    secondary_y=True
                )
                
                # Add overbought/oversold lines
                fig.add_hline(y=100, line_dash="dash", line_color="red", secondary_y=True, row=1, col=1)
                fig.add_hline(y=-100, line_dash="dash", line_color="green", secondary_y=True, row=1, col=1)
                
                # Update y-axis
                fig.update_yaxes(title_text="CCI", row=1, col=1, secondary_y=True)
            except Exception as e:
                # Alternative approach without secondary_y
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["CCI"].index,
                        y=indicator_data["CCI"],
                        mode="lines",
                        name="CCI",
                        line=dict(color="purple", width=1, dash="dot")
                    ),
                    row=1, col=1
                )
    except Exception as e:
        # Add error annotation to chart
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"Error adding {indicator_name}: {str(e)}",
            showarrow=False,
            font=dict(color="red", size=12),
            xref="paper",
            yref="paper"
        )
        
    return fig
