# Turtle Trading Strategy

This project implements the Turtle Trading strategy using Backtrader and Streamlit. The strategy is a trend-following system that uses breakout levels and the Average True Range (ATR) for position sizing and trade decisions.

## Features

- **Backtesting**: Simulate the Turtle Trading Strategy over different asset classes
- **Streamlit Interface**: A user-friendly web interface to input parameters and view results.
- **Trade Analysis**: Analyze trades to determine best and worst trades, calculate performance metrics, and visualize results.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/turtle-trading-strategy.git
   cd turtle-trading-strategy
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Navigate through the app**:
   - Choose stock or crypto ticker, set basic settings and trading parameters and run the backtest 

## Strategy Parameters

- **Breakout Period High**: Number of days for breakout high calculation
- **Breakout Period Low**: Number of days for breakout low calculation
- **ATR Period**: Average True Range calculation period
- **ATR Multiplier**: Multiplier for ATR-based stops
- **Risk Percentage**: Risk per trade as a decimal (e.g., 0.02 for 2%)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact me on [LinkedIn](https://www.linkedin.com/in/oktay-bogazkaya/).
