import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Fetch market data
def get_data(symbol, start_date, end_date):
    """
    Fetch data from yfinance and prepare it for backtrader.
    """
    data_df = yf.download(symbol, start=start_date, end=end_date)
  
    data_df = data_df[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
    data_df.columns = ['close', 'high', 'low', 'open', 'volume']  # Rename columns to lowercase
    data_df.dropna(inplace=True)  # Drop any rows with NaN values
    data_df.index = pd.to_datetime(data_df.index)  # Ensure index is datetime

    return data_df

# Plot equity curve
def plot_equity_curve(strat, data_df):
    """
    Plot the equity curve of the portfolio value over time.
    """
    portfolio_values = strat.analyzers.portfolio_values.values
    dates = data_df.index[-len(portfolio_values):]

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86C1', width=2)
        )
    )

    fig_equity.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white'
    )

    return fig_equity