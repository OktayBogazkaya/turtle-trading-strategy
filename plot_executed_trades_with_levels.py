import plotly.graph_objects as go
import pandas as pd

def plot_executed_trades_with_levels(data_df, trade_tracker, strategy, symbol):
    """
    Create an interactive candlestick chart with:
    - Executed trades (arrows and text)
    - N-day high/low lines
    - Stop loss levels during trades
    """
    # Ensure data_df index is tz-aware UTC
    if data_df.index.tz is None:
        data_df.index = data_df.index.tz_localize('UTC')

    # Calculate 20-day high/low
    high_20d = pd.Series(strategy.high_price.array, index=data_df.index)
    low_20d = pd.Series(strategy.low_price.array, index=data_df.index)

    # Create figure
    fig = go.Figure()

    # Add SMA trend line
    sma_trend = pd.Series(strategy.sma_trend.array, index=data_df.index)
    fig.add_trace(
        go.Scatter(
            x=data_df.index,
            y=sma_trend,
            mode='lines',
            line=dict(color='white', width=1),
            name=f'SMA {strategy.params.trend_period}'
        )
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data_df.index,
            open=data_df['open'],
            high=data_df['high'],
            low=data_df['low'],
            close=data_df['close'],
            name='Price'
        )
    )

    # Add N-day high line
    fig.add_trace(
        go.Scatter(
            x=data_df.index,
            y=high_20d,
            mode='lines',
            line=dict(color='yellow', width=1, dash='dash'),
            name='High'
        )
    )

    # Add N-day low line
    fig.add_trace(
        go.Scatter(
            x=data_df.index,
            y=low_20d,
            mode='lines',
            line=dict(color='yellow', width=1, dash='dash'),
            name='Low'
        )
    )

    # Process trades and create stop loss lines
    for i in range(len(trade_tracker.trades)):
        trade = trade_tracker.trades[i]
        entry_date = pd.to_datetime(trade['entry_date']).tz_localize('UTC')

        sell_log = [log for log in strategy.log_entries
                   if "SELL CREATE" in log and "Stop:" in log
                   and entry_date <= pd.to_datetime(log.split()[0]).tz_localize('UTC')][0]

        stop_date = pd.to_datetime(sell_log.split()[0]).tz_localize('UTC')
        stop_price = float(sell_log.split("Stop:")[1].split()[0])

        fig.add_trace(
            go.Scatter(
                x=[stop_date],
                y=[stop_price],
                mode='markers',
                marker=dict(symbol='cross', color='orange', size=8),
                name=f'Stop Loss (Trade {trade["trade_id"]})'
            )
        )

    # Handle buy points - both closed and open trades
    buy_info = []

    # Add closed trades buy info
    for trade in trade_tracker.trades:
        buy_info.append({
            'date': pd.to_datetime(trade['entry_date']).tz_localize('UTC'),
            'price': trade['entry_price'],
            'size': trade['size'],
            'trade_id': trade['trade_id']
        })

    # Add open trades buy info
    for trade_id, trade in trade_tracker.open_trades.items():
        buy_info.append({
            'date': pd.to_datetime(trade['entry_date']).tz_localize('UTC'),
            'price': trade['entry_price'],
            'size': trade['size'],
            'trade_id': trade_id
        })

    # Sort by trade ID
    buy_info = sorted(buy_info, key=lambda x: x['trade_id'])

    buy_dates = [trade['date'] for trade in buy_info]
    buy_prices = [trade['price'] for trade in buy_info]
    buy_sizes = [trade['size'] for trade in buy_info]
    buy_ids = [trade['trade_id'] for trade in buy_info]

    buy_texts = [
        f"Buy #{trade_id}<br>" +
        f"Date: {date.strftime('%Y-%m-%d')}<br>" +
        f"Price: ${price:.2f}<br>" +
        f"Size: {size} shares"
        for date, price, size, trade_id in zip(buy_dates, buy_prices, buy_sizes, buy_ids)
    ]

    # Add buy arrows and text
    for i in range(len(buy_dates)):
        # Add arrow
        fig.add_annotation(
            x=buy_dates[i],
            y=buy_prices[i],
            text="↑",
            font=dict(size=20, color='cyan'),
            showarrow=False,
            yshift=-20
        )

        # Add "BUY #N" text
        fig.add_annotation(
            x=buy_dates[i],
            y=buy_prices[i],
            text=f"BUY #{buy_ids[i]}",
            font=dict(size=12, color='cyan'),
            showarrow=False,
            yshift=-40
        )

    # Add marker points for buy hover information
    fig.add_trace(
        go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            marker=dict(symbol='circle', size=1, color='cyan', opacity=0),
            name='Buy Execute',
            text=buy_texts,
            hovertemplate="%{text}<extra></extra>"
        )
    )

    # Handle sell points (closed trades only)
    sell_dates = [pd.to_datetime(trade['exit_date']).tz_localize('UTC') for trade in trade_tracker.trades]
    sell_prices = [trade['exit_price'] for trade in trade_tracker.trades]
    sell_pnls = [trade['pnl_pct'] for trade in trade_tracker.trades]
    sell_ids = [trade['trade_id'] for trade in trade_tracker.trades]

    sell_texts = [
        f"Sell #{trade_id}<br>" +
        f"Date: {date.strftime('%Y-%m-%d')}<br>" +
        f"Price: ${price:.2f}<br>" +
        f"P&L: {pnl:.1f}%"
        for date, price, pnl, trade_id in zip(sell_dates, sell_prices, sell_pnls, sell_ids)
    ]

    # Add sell arrows and text
    for i in range(len(sell_dates)):
        # Add arrow
        fig.add_annotation(
            x=sell_dates[i],
            y=sell_prices[i],
            text="↓",
            font=dict(size=20, color='magenta'),
            showarrow=False,
            yshift=20
        )

        # Add "SELL #N" text
        fig.add_annotation(
            x=sell_dates[i],
            y=sell_prices[i],
            text=f"SELL #{sell_ids[i]}",
            font=dict(size=12, color='magenta'),
            showarrow=False,
            yshift=40
        )

    # Add marker points for sell hover information
    fig.add_trace(
        go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            marker=dict(symbol='circle', size=1, color='magenta', opacity=0),
            name='Sell Execute',
            text=sell_texts,
            hovertemplate="%{text}<extra></extra>"
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Price Chart with Signals and Levels',
            font=dict(color='white')
        ),
        yaxis_title=dict(
            text='Price',
            font=dict(color='white')
        ),
        xaxis_title=dict(
            text='Date',
            font=dict(color='white')
        ),
        template='plotly_dark',
        hovermode='closest',
        showlegend=True,
        height=800,
        plot_bgcolor='rgb(20,20,20)',
        paper_bgcolor='rgb(20,20,20)',
        legend=dict(font=dict(color='white'))
    )

    return fig