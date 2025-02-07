import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from datetime import datetime, timedelta
import itertools
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import streamlit as st

from plot_executed_trades_with_levels import plot_executed_trades_with_levels

class PortfolioValue(bt.Analyzer):
    """
    Analyzer to track portfolio value over time.
    """
    def __init__(self):
        self.values = []

    def next(self):
        self.values.append(self.strategy.broker.getvalue())

class TradeTracker:
    """
    Track open and closed trades, and calculate best/worst trades.
    """
    def __init__(self):
        self.trades = []
        self.open_trades = {}
        self.trade_counter = 1

    def add_open_trade(self, trade_info):
        """
        Add a new open trade to the tracker.
        """
        trade_id = self.trade_counter
        self.open_trades[trade_id] = {
            'trade_id': trade_id,
            'entry_date': trade_info['entry_date'],
            'entry_price': trade_info['entry_price'],
            'size': trade_info['size']
        }
        self.trade_counter += 1
        return trade_id

    def add_closed_trade(self, trade_id, trade_info):
        """
        Add a closed trade to the tracker and remove it from open trades.
        """
        trade_info['trade_id'] = trade_id
        self.trades.append(trade_info)
        if trade_id in self.open_trades:
            del self.open_trades[trade_id]

    def get_best_worst_trades(self):
        """
        Return the best and worst trades based on net PnL.
        """
        if not self.trades:
            return None, None
        sorted_trades = sorted(self.trades, key=lambda x: x['net_pnl'], reverse=True)
        return sorted_trades[0], sorted_trades[-1]

class TurtleStrategy(bt.Strategy):
    """
    Implementation of the Turtle Trading Strategy.

    The strategy uses breakout levels, ATR for position sizing,
    and trending indicators for trade decisions.
    """
    params = (
        ('period_high', 20),     # Breakout period high
        ('period_low', 10),      # Breakout period low
        ('atr_period', 14),      # ATR period
        ('risk_pct', 0.02),      # risk per trade
        ('atr_multiplier', 2.5), # N-day ATR multiplier for stops
        ('adx_period', 14),      # ADX period
        ('adx_threshold', 0),    # ADX threshold for trend strength
        ('trend_period', 50),    # Moving average trend filter
    )

    def __init__(self):
        """
        Initialize indicators and variables for the strategy.
        """
        self.highest_price = None
        self.order = None
        self.stop_price = None
        self.buy_price = None
        self.buy_comm = None
        self.entry_size = None
        self.entry_date = None
        self.current_trade_id = None
        self.trade_tracker = TradeTracker()
        self.log_entries = []

        # Indicators
        self.high_price = bt.indicators.Highest(self.data.high, period=self.params.period_high)
        self.low_price = bt.indicators.Lowest(self.data.low, period=self.params.period_low)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period, plotname='ATR')
        self.adx = bt.indicators.DirectionalMovement(self.data, period=self.params.adx_period, plotname='ADX')
        self.sma_trend = bt.indicators.SMA(self.data.close, period=self.params.trend_period)

    def log(self, txt, dt=None):
        """
        Log messages with a timestamp.
        """
        dt = dt or self.datas[0].datetime.date(0)
        log_entry = f'{dt.isoformat()} {txt}'
        self.log_entries.append(log_entry)
        print(log_entry)

    def calculate_stop_loss(self):
        """
        Calculate and update the stop-loss price.
        """
        new_stop = self.highest_price - (self.atr[0] * self.params.atr_multiplier)
        self.stop_price = max(new_stop, self.stop_price) if self.stop_price else new_stop

    def _handle_buy_order(self, order):
        """
        Handle completed buy order execution and logging.
        """
        # Store trade entry details
        self.buy_price = order.executed.price
        self.buy_comm = order.executed.comm
        self.entry_size = order.executed.size
        self.entry_date = self.data.datetime.date(0)

        # Prepare and track trade info
        trade_info = {
            'entry_date': self.entry_date,
            'entry_price': self.buy_price,
            'size': self.entry_size
        }
        self.current_trade_id = self.trade_tracker.add_open_trade(trade_info)

        # Log trade entry
        self.log(f'BUY #{self.current_trade_id} EXECUTED | Price: {order.executed.price:.2f} | '
                f'Size: {order.executed.size:.0f} | Cost: {order.executed.value:.2f} | '
                f'Comm: {order.executed.comm:.2f}')
        self.highest_price = order.executed.price

    def _handle_sell_order(self, order):
        """
        Handle completed sell order execution, calculations, and logging.
        """
        # Calculate PnL metrics
        gross_pnl = (order.executed.price - self.buy_price) * self.entry_size
        net_pnl = gross_pnl - self.buy_comm - order.executed.comm
        pnl_pct = (net_pnl / (self.buy_price * self.entry_size)) * 100

        # Calculate dates
        exit_date = self.data.datetime.date(0)
        holding_days = (exit_date - self.entry_date).days

        # Prepare trade info for tracking
        trade_info = {
            'entry_date': self.entry_date,
            'exit_date': exit_date,
            'entry_price': self.buy_price,
            'exit_price': order.executed.price,
            'size': self.entry_size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'holding_days': holding_days,
            'total_commission': self.buy_comm + order.executed.comm
        }
        self.trade_tracker.add_closed_trade(self.current_trade_id, trade_info)

        # Log trade exit
        self.log(f'SELL #{self.current_trade_id} EXECUTED | Price: {order.executed.price:.2f} | '
                f'Size: {order.executed.size:.0f} | Cost: {order.executed.value:.2f} '
                f'Comm: {order.executed.comm:.2f} |\n\nTrade P/L: {net_pnl:.2f} ({pnl_pct:.2f}%) | '
                f'Entry: {self.buy_price:.2f} | Exit: {order.executed.price:.2f} | '
                f'Size: {order.executed.size:.0f} | Total Commission: '
                f'{(self.buy_comm + order.executed.comm):.2f} |\n\nHolding Period: {holding_days} days')

        # Reset trade variables
        self._reset_trade_variables()

    def _reset_trade_variables(self):
        """
        Reset all trade-related variables after trade completion.
        """
        self.highest_price = None
        self.stop_price = None
        self.buy_price = None
        self.buy_comm = None
        self.entry_size = None
        self.entry_date = None
        self.current_trade_id = None

    def notify_order(self, order):
        """
        Handle order notifications from the broker.

        Args:
            order: The order notification from the broker
        """
        # Skip if order is pending
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Handle completed orders
        if order.status in [order.Completed]:
            if order.isbuy():
                self._handle_buy_order(order)
            else:
                self._handle_sell_order(order)
        # Handle failed orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def calculate_position_size(self, price):
        """
        Calculate the position size based on ATR and risk percentage.
        """
        current_atr = self.atr[0]
        dollar_volatility = current_atr * self.params.atr_multiplier
        if dollar_volatility == 0:
            return 0
        account_value = self.broker.getvalue()
        dollar_risk = account_value * self.params.risk_pct
        position_size = int(dollar_risk / dollar_volatility)
        return position_size

    def next(self):
        """
        Define the strategy logic for each step in the backtest.
        """
        if self.order:
            return

        if not self.position:
            # Check for buy conditions
            if (self.data.close[0] > self.high_price[-1] and
                self.data.close[0] > self.sma_trend[0] and
                self.adx.adx[0] > self.params.adx_threshold):

                shares = self.calculate_position_size(self.data.close[0])

                if shares > 0:
                    risk_amount = shares * self.atr[0] * self.params.atr_multiplier
                    account_value = self.broker.getvalue()
                    risk_pct = (risk_amount / account_value) * 100

                    self.log(f'BUY CREATE | Price: {self.data.close[0]:.2f} | '
                            f'Shares: {shares} | Risk Amount: {risk_amount:.2f} |'
                            f'({risk_pct:.2f}%)')
                    self.order = self.buy(size=shares)

        else:
            # Update highest price and calculate stop loss
            if self.highest_price is None:
                self.highest_price = self.position.price
            else:
                self.highest_price = max(self.highest_price, self.data.close[0])

            self.calculate_stop_loss()

            # Calculate unrealized PnL
            unrealized_pnl = (self.data.close[0] - self.buy_price) * self.entry_size - self.buy_comm
            unrealized_pnl_pct = (unrealized_pnl / (self.buy_price * self.entry_size)) * 100

            # Check for sell conditions
            if (self.data.close[0] < self.stop_price or
                self.data.close[0] < self.low_price[-1] or
                self.data.close[0] < self.sma_trend[0]* (1 - 0.005)):

                exit_reason = ("Trailing Stop" if self.data.close[0] < self.stop_price else
                             "Low Breakout" if self.data.close[0] < self.low_price[-1] else
                             "Trend Break")

                self.log(f'SELL CREATE ({exit_reason}) | Price: {self.data.close[0]:.2f} | '
                        f'Stop: {self.stop_price:.2f} | '
                        f'Current P/L: {unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)')

                self.order = self.sell(size=self.position.size)

    def print_best_worst_trades(self):
        """
        Print best and worst trades analysis.
        """
        best_trade, worst_trade = self.trade_tracker.get_best_worst_trades()

        print("\n=== BEST AND WORST TRADES ANALYSIS ===")
        print("\nBEST TRADE:")
        print(f"Entry Date: {best_trade['entry_date']}")
        print(f"Exit Date: {best_trade['exit_date']}")
        print(f"Holding Period: {best_trade['holding_days']} days")
        print(f"Entry Price: ${best_trade['entry_price']:.2f}")
        print(f"Exit Price: ${best_trade['exit_price']:.2f}")
        print(f"Position Size: {best_trade['size']} shares")
        print(f"Net P/L: ${best_trade['net_pnl']:.2f} ({best_trade['pnl_pct']:.2f}%)")
        print(f"Total Commission: ${best_trade['total_commission']:.2f}")

        print("\nWORST TRADE:")
        print(f"Entry Date: {worst_trade['entry_date']}")
        print(f"Exit Date: {worst_trade['exit_date']}")
        print(f"Holding Period: {worst_trade['holding_days']} days")
        print(f"Entry Price: ${worst_trade['entry_price']:.2f}")
        print(f"Exit Price: ${worst_trade['exit_price']:.2f}")
        print(f"Position Size: {worst_trade['size']} shares")
        print(f"Net P/L: ${worst_trade['net_pnl']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
        print(f"Total Commission: ${worst_trade['total_commission']:.2f}")

def analyze_trades(strat):
    """
    Analyze trade results and calculate metrics.
    """
    global trade_analyzer, drawdown_analyzer, sharpe_ratio_analyzer
    global closed_trades, open_trades, win_rate, avg_win_loss_ratio

    # Retrieve the trade analysis from the results
    trade_analyzer = strat.analyzers.trades.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    sharpe_ratio_analyzer = strat.analyzers.sharpe.get_analysis()

    annual_returns = strat.analyzers.annualreturns.get_analysis()
    annual_returns_df = pd.DataFrame(list(annual_returns.items()), columns=["Year", "Total Return (%)"])
    annual_returns_df["Total Return (%)"] = annual_returns_df["Total Return (%)"] * 100

    portfolio_values = results[0].analyzers.portfolio_values.values

    # Extract relevant data
    closed_trades = trade_analyzer['total']['closed']
    open_trades = trade_analyzer['total']['open']
    gross_pnl = trade_analyzer['pnl']['gross']['total']
    net_pnl = trade_analyzer['pnl']['net']['total']
    sharpe_ratio = sharpe_ratio_analyzer['sharperatio']

    max_drawdown = drawdown_analyzer['max']['drawdown']
    max_drawdown_len = drawdown_analyzer['max']['len']

    # Win rate
    win_rate = trade_analyzer['won']['total'] / closed_trades if closed_trades > 0 else 0

    # Expectancy: (avg win * win rate) + (avg loss * loss rate)
    avg_win = trade_analyzer['won']['pnl']['average']
    avg_loss = trade_analyzer['lost']['pnl']['average']
    expectancy = (avg_win * (trade_analyzer['won']['total'] / closed_trades)) + \
                (avg_loss * (trade_analyzer['lost']['total'] / closed_trades))

    # Average Win/Loss ratio
    avg_win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan

    # Long and Short trades count
    longs = trade_analyzer['long']['total']
    shorts = trade_analyzer['short']['total']

    # Print Trade Analysis
    print("\n=== TRADE ANALYSIS ===")
    print(f"\nOverall Performance:")
    print("Starting Portfolio Value: %.2f" % cerebro.broker.startingcash)
    print("Ending Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print('Annualized Return: %.2f%%' % (strat.analyzers.returns.get_analysis()['rnorm100']))
    print(f"Annual Returns: {annual_returns_df}")
    print('Sharpe Ratio:', sharpe_ratio)
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Max Drawdown Length: {max_drawdown_len if max_drawdown_len is not None else 'N/A'}")

    print(f"\nTrade Statistics:")
    print(f"Total Closed Trades: {closed_trades}")
    print(f"Open Trades: {open_trades}")

    print(f"Expectancy: {expectancy:.2f}")

    print(f"\nProfit Metrics:")
    print(f"Avg Win: {avg_win:.2f}")
    print(f"Avg Loss: {avg_loss:.2f}")
    print(f"Avg Win / Avg Loss Ratio: {avg_win_loss_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Long Trades: {longs}")
    print(f"Short Trades: {shorts}")

def get_data(symbol, start_date, end_date):
    """
    Fetch data from yfinance and prepare it for backtrader.
    """
    data_df = yf.download(symbol, start=start_date, end=end_date)
  
    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].copy()
    data_df.columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']  # Rename columns to lowercase
    data_df.dropna(inplace=True)  # Drop any rows with NaN values
    data_df.index = pd.to_datetime(data_df.index)  # Ensure index is datetime

    return data_df

# Select ticker and date range
st.sidebar.title('Basic Settings')

symbol = st.sidebar.text_input('Enter Stock Symbol', value='NVDA')
initial_cash = st.sidebar.number_input('Initial Cash', value=100000.0, step=1000.0)
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-12-31'))

# Add About section to sidebar
st.sidebar.markdown("""
---
### About Turtle Trading
The Turtle Trading strategy was famously created in 1983 when Richard Dennis made a bet that he could teach anyone to trade successfully. 
The strategy uses trend-following rules, position sizing based on volatility, and systematic risk management to capture large market moves. 
For more, please visit [Macro-Ops](https://macro-ops.com/richard-dennis-turtle-trading-strategy-explained/).
""")

# Add footer to sidebar
st.sidebar.markdown("""
---
Made with ❤️ by [RhinoInsight](https://twitter.com/rhinoinsight)
""")

# Main page title
st.title('Turtle Trading Strategy Backtester')

# Strategy parameters in main page
st.header('Strategy Parameters')

col1, col2, col3 = st.columns(3)

with col1:
    period_high = st.number_input('Breakout Period High', value=20, help='Number of days for breakout high calculation')
    atr_period = st.number_input('ATR Period', value=14, help='Average True Range calculation period')
    adx_period = st.number_input('ADX Period', value=14, help='Average Directional Index calculation period')

with col2:
    period_low = st.number_input('Breakout Period Low', value=10, help='Number of days for breakout low calculation')
    risk_pct = st.number_input('Risk Percentage', value=0.02, format="%.3f", help='Risk per trade as decimal (0.02 = 2%)')
    adx_threshold = st.number_input('ADX Threshold', value=0, help='Minimum ADX value for trade entry')

with col3:
    trend_period = st.number_input('Trend Period', value=50, help='Moving average period for trend filter')
    atr_multiplier = st.number_input('ATR Multiplier', value=2.5, help='Multiplier for ATR-based stops')

# Convert dates to string format for yfinance
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

# Display results in Streamlit
if st.button('Run Backtest'):
    with st.spinner('Running backtest...'):
        data_df = get_data(symbol, start_date, end_date)
        
        # Initialize backtest
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=data_df)
        cerebro.adddata(data)
        
        # Pass the UI parameters to the strategy
        cerebro.addstrategy(TurtleStrategy,
            period_high=period_high,
            period_low=period_low,
            atr_period=atr_period,
            risk_pct=risk_pct,
            atr_multiplier=atr_multiplier,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            trend_period=trend_period
        )
        
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(PortfolioValue, _name="portfolio_values")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # Run the backtest
        results = cerebro.run()
        strat = results[0]

        analyze_trades(strat)

        # Display results in Streamlit
        st.header('Backtest Results')

        tab1, tab2, tab3, tab4 = st.tabs(['Performance Metrics', 'Best/Worst Trades', 'Equity Curve', 'Trade Chart'])
        
        with tab1:
            trade_analyzer = strat.analyzers.trades.get_analysis()
            drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
            sharpe_ratio_analyzer = strat.analyzers.sharpe.get_analysis()
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Starting Cash', f'${initial_cash:,.2f}')
                st.metric('Sharpe Ratio', f"{sharpe_ratio_analyzer['sharperatio']:.2f}")
                st.metric('Closed Trades', f"{closed_trades}")
            with col2:
                st.metric('Final Portfolio Value', f'${cerebro.broker.getvalue():,.2f}')
                st.metric('Win Rate', f"{win_rate:.2%}")
                st.metric('Open Trades', f"{open_trades}")
            with col3:
                st.metric('Total Return', f"{((cerebro.broker.getvalue() - initial_cash) / initial_cash * 100):.2f}%")
                st.metric('AvgWin/Loss Ratio', f"{avg_win_loss_ratio:.2f}")
                st.metric('Max Drawdown', f"{drawdown_analyzer['max']['drawdown']:.2f}%")
        
        with tab2:
            # Display best/worst trades
            st.subheader('Best and Worst Trades')
            best_trade, worst_trade = strat.trade_tracker.get_best_worst_trades()
            
            if best_trade and worst_trade:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Best Trade")
                    st.write(f"Entry Date: {best_trade['entry_date']}")
                    st.write(f"Exit Date: {best_trade['exit_date']}")
                    st.write(f"Holding Period: {best_trade['holding_days']} days")
                    st.write(f"Net P/L: ${best_trade['net_pnl']:.2f}")
                    st.write(f"Return: {best_trade['pnl_pct']:.2f}%")
                
                with col2:
                    st.write("Worst Trade")
                    st.write(f"Entry Date: {worst_trade['entry_date']}")
                    st.write(f"Exit Date: {worst_trade['exit_date']}")
                    st.write(f"Holding Period: {worst_trade['holding_days']} days")
                    st.write(f"Net P/L: ${worst_trade['net_pnl']:.2f}")
                    st.write(f"Return: {worst_trade['pnl_pct']:.2f}%")
        with tab3:
            st.subheader('Equity Curve')
            # Display equity curve
            portfolio_values = results[0].analyzers.portfolio_values.values
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
            
            st.plotly_chart(fig_equity, use_container_width=True)

        with tab4:
            st.subheader('Trade Chart')
            fig = plot_executed_trades_with_levels(data_df, strat.trade_tracker, strat, symbol)
            st.plotly_chart(fig, use_container_width=True)
        
        # Add collapsible section to display trade logs
        with st.expander("View Trade Logs"):
            for log_entry in strat.log_entries:
                st.text(log_entry)
                if "Trade P/L:" in log_entry:
                    st.text("-" * 80) 
