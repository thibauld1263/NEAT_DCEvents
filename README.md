# NEAT_DCEvents
Machine Learning trading system using NEAT (NeuroEvolution of Augmenting Topologies) and Directional Change events

## What It Does

Trains a neural network to trade by:
1. Converting tick data into Directional Change (DC) events
2. Calculating technical features (RSI, volatility, momentum, etc.)
3. Evolving a neural network using NEAT to make trading decisions
4. Optimizing for Sharpe Ratio (risk-adjusted returns)

## Features

- **Event-based trading**: Uses Directional Change instead of time bars (inspired by the Alpha Engine fromRichard B. Olsen and James Glattfelder: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2951348)
- **Neuroevolution**: NEAT evolves network topology and weights
- **Sharpe optimization**: Maximizes risk-adjusted returns
- **Live trading**: Socket-based integration with CTrader 
- **Backtesting**: Full walk-forward validation with train/test splits

## Configuration

Edit `Final Project.py` to adjust:

- `MARKET_DATA`: Path to your CSV tick data
- `DC_THRESHOLD`: Directional Change threshold (default: 5 pips)
- `GENERATIONS`: Number of NEAT generations (default: 50)
- `TRAIN_SPLIT`: Train/test split ratio (default: 0.20)
- `OPEN_THRESHOLD`: Signal strength to open trades (default: 0.5)
- `CLOSE_THRESHOLD`: Signal strength to close trades (default: 0.6)

## Input Features (9)

1. Event type (UP/DOWN)
2. Volatility
3. RSI
4. Market speed (price velocity)
5. Momentum
6. Price distance from EMA
7. Trend bias
8. In position (0/1)
9. Unrealized P&L

## Output Signals (3)

1. Buy signal
2. Sell signal
3. Close signal

## Files

- `Final Project.py`: Main entry point
- `shared_features.py`: Feature calculation (shared between backtest and live)
- `dc_processor.py`: Converts ticks to DC events
- `scalping_backtester.py`: Backtesting engine
- `config-scalping.txt`: NEAT configuration

## Risk Management

- Adaptive stop-loss based on volatility
- Spread filter (blocks trades when spread > 2 pips)
- Warmup period (waits 20 events before trading)
- Position sizing via `LOT_SIZE` parameter

## Results

After training, you get:
- `winner.pkl`: Trained model
- `results.png`: Equity curves and P&L distribution
- `backtest_trades.csv`: All trades with timestamps and P&L


## Examples of trained models:

<img width="1400" height="1000" alt="1" src="https://github.com/user-attachments/assets/2cae78b1-bd35-4cc9-ba3e-c5e2e7d32756" />
<img width="2100" height="1500" alt="2" src="https://github.com/user-attachments/assets/a06b9f43-bcb5-4916-a762-ea2b3f57616b" />
<img width="2100" height="1500" alt="8" src="https://github.com/user-attachments/assets/86db5c77-f7cb-4473-8e57-0968750c3db2" />
<img width="2100" height="1500" alt="9" src="https://github.com/user-attachments/assets/06621fd0-0f07-472a-8ad4-a44d4f84289b" />



## License

MIT

## Disclaimer

**This is for educational purposes only. Trading forex carries significant risk. Use at your own risk.**
