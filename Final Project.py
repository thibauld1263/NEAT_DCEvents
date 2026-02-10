"""
NEAT Trading 
Modes: train, backtest, live
"""

import neat
import pickle
import os
import sys
import socket
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import time
from dc_processor import DCProcessor
from scalping_backtester import Backtester
from shared_features import FeatureCalculator, normalize_features, WARMUP_EVENTS

# ============ CONFIG ============

MARKET_DATA = r"C:\Users\thiba\Desktop\MARKET DATA\market_data_AUDCAD.csv"
NEAT_CONFIG = r"C:\Users\thiba\Desktop\FINAL NEAT PROJECT\V3\config-scalping.txt"
MODEL_PATH = r"C:\Users\thiba\Desktop\winner.pkl"

GENERATIONS = 50
TRAIN_SPLIT = 0.20  

INITIAL_BALANCE = 10000.0
PIP_VALUE = 10.0
PIP_SIZE = 0.0001
DC_THRESHOLD = 5 * PIP_SIZE

# Trading Parameters
VOL_SCALE_FACTOR = 1e4
PIPS_NORM = 10.0
OPEN_THRESHOLD = 0.5   
CLOSE_THRESHOLD = 0.6  
MIN_SL_PIPS = 25.0
VOL_SL_MULT = 2.5

# Live
HOST = "127.0.0.1"
PORT = 9001
LOT_SIZE = 0.05

# ============ GLOBALS ============

BACKTESTER = None
BACKTESTER_OOS = None

# ============ DATA ============

def load_events() -> pd.DataFrame:
    if not os.path.exists(MARKET_DATA):
        print(f"ERROR: {MARKET_DATA} not found")
        sys.exit(1)
    
    processor = DCProcessor(threshold=DC_THRESHOLD, pip_size=PIP_SIZE)
    events = processor.process_ticks(MARKET_DATA)
    print(f"Loaded {len(events)} DC events")
    return events

# ============ TRAINING ============

def calculate_sharpe(equity_curve):
    """
    Calculate Sharpe Ratio (annualized).
    Sharpe = (mean return / std return) * sqrt(252)
    
    Sharpe > 2.0 = Excellent
    Sharpe > 1.0 = Good
    Sharpe < 0.5 = Poor
    
    Automatically penalizes:
    - Excessive trades (increases volatility)
    - Drawdowns (negative returns)
    - Erratic performance
    """
    if len(equity_curve) < 3:
        return -10.0
    
    # Calculate returns
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    if len(returns) == 0:
        return -10.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        # Perfect equity (no variance) - check if profitable
        if mean_return > 0:
            return 10.0  # Perfect upward
        else:
            return -10.0  # Perfect downward
    
    # Sharpe ratio (annualized assuming 252 trading days)
    sharpe = (mean_return / std_return) * np.sqrt(252)
    
    return sharpe

def eval_genomes(genomes, config):
    """
    Sharpe Ratio fitness:
    - Sharpe measures return/risk ratio
    - Automatically penalizes excessive trades and drawdowns
    - Filters: minimum trades, max OOS loss
    """
    global BACKTESTER, BACKTESTER_OOS
    
    for genome_id, genome in genomes:
        try:
            # Evaluate IS (train)
            is_pnl = BACKTESTER.evaluate(genome, config)
            is_stats = BACKTESTER.get_stats()
            
            # Evaluate OOS (test)
            oos_pnl = BACKTESTER_OOS.evaluate(genome, config)
            oos_stats = BACKTESTER_OOS.get_stats()
            
            # === STRICT FILTERS ===
            # Minimum trades required
            if is_stats['trades'] < 100:
                genome.fitness = -10000.0
                continue
            if oos_stats['trades'] < 50:
                genome.fitness = -10000.0
                continue
            
            # Max acceptable OOS loss
            if oos_pnl < -500:
                genome.fitness = -10000.0
                continue
            
            # === SHARPE CALCULATION ===
            equity_is = BACKTESTER.get_equity_curve()
            equity_oos = BACKTESTER_OOS.get_equity_curve()
            
            sharpe_is = calculate_sharpe(equity_is)
            sharpe_oos = calculate_sharpe(equity_oos)
            
            # === FITNESS ===
            # Weighted average: IS counts more (more data)
            genome.fitness = (sharpe_is * 0.7) + (sharpe_oos * 0.3)
            
        except Exception as e:
            print(f"Error genome {genome_id}: {e}")
            genome.fitness = -10000.0

def train():
    global BACKTESTER, BACKTESTER_OOS
    
    events = load_events()
    
    # Split IS/OOS
    split = int(len(events) * TRAIN_SPLIT)
    train_events = events.iloc[:split].reset_index(drop=True)
    test_events = events.iloc[split:].reset_index(drop=True)
    
    print(f"Train (IS): {len(train_events)} events ({TRAIN_SPLIT*100:.0f}%)")
    print(f"Test (OOS): {len(test_events)} events ({(1-TRAIN_SPLIT)*100:.0f}%)")
    
    BACKTESTER = Backtester(
        events=train_events,
        initial_balance=INITIAL_BALANCE,
        pip_value=PIP_VALUE,
        pip_size=PIP_SIZE,
        vol_scale_factor=VOL_SCALE_FACTOR,
        pips_norm=PIPS_NORM,
        open_threshold=OPEN_THRESHOLD,
        close_threshold=CLOSE_THRESHOLD,
        min_sl_pips=MIN_SL_PIPS,
        vol_sl_mult=VOL_SL_MULT
    )
    
    BACKTESTER_OOS = Backtester(
        events=test_events,
        initial_balance=INITIAL_BALANCE,
        pip_value=PIP_VALUE,
        pip_size=PIP_SIZE,
        vol_scale_factor=VOL_SCALE_FACTOR,
        pips_norm=PIPS_NORM,
        open_threshold=OPEN_THRESHOLD,
        close_threshold=CLOSE_THRESHOLD,
        min_sl_pips=MIN_SL_PIPS,
        vol_sl_mult=VOL_SL_MULT
    )
    
    # NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG
    )
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    
    winner = p.run(eval_genomes, GENERATIONS)
    
    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(winner, f)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Winner saved: {MODEL_PATH}")
    print(f"Combined Fitness: {winner.fitness:.2f}")
    
    # Final IS
    print("\n=== FINAL IS (TRAIN) ===")
    BACKTESTER.evaluate(winner, config)
    BACKTESTER.print_report()
    
    # Final OOS
    print("\n=== FINAL OOS (TEST) ===")
    BACKTESTER_OOS.evaluate(winner, config)
    BACKTESTER_OOS.print_report()
    
    # Plots
    plot_results(BACKTESTER, BACKTESTER_OOS, winner, config)
    
    # Export trades
    export_trades(BACKTESTER, BACKTESTER_OOS)

# ============ BACKTEST ============

def backtest(debug: bool = False):
    """Run backtest with optional debug output"""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found")
        return
    
    if debug:
        print("\n" + "="*60)
        print("DEBUG MODE ENABLED - Showing all decisions")
        print("="*60 + "\n")
    
    events = load_events()
    
    # Split IS/OOS
    split = int(len(events) * TRAIN_SPLIT)
    train_events = events.iloc[:split].reset_index(drop=True)
    test_events = events.iloc[split:].reset_index(drop=True)
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG
    )
    
    with open(MODEL_PATH, 'rb') as f:
        winner = pickle.load(f)
    
    # Train
    print("=== TRAIN (IS) ===")
    train_bt = Backtester(
        train_events,
        INITIAL_BALANCE,
        PIP_VALUE,
        PIP_SIZE,
        vol_scale_factor=VOL_SCALE_FACTOR,
        pips_norm=PIPS_NORM,
        open_threshold=OPEN_THRESHOLD,
        close_threshold=CLOSE_THRESHOLD,
        min_sl_pips=MIN_SL_PIPS,
        vol_sl_mult=VOL_SL_MULT,
        debug=debug  # Pass debug flag
    )
    train_bt.evaluate(winner, config)
    train_bt.print_report()
    
    # Test
    print("\n=== TEST (OOS) ===")
    test_bt = Backtester(
        test_events,
        INITIAL_BALANCE,
        PIP_VALUE,
        PIP_SIZE,
        vol_scale_factor=VOL_SCALE_FACTOR,
        pips_norm=PIPS_NORM,
        open_threshold=OPEN_THRESHOLD,
        close_threshold=CLOSE_THRESHOLD,
        min_sl_pips=MIN_SL_PIPS,
        vol_sl_mult=VOL_SL_MULT,
        debug=debug  # Pass debug flag
    )
    test_bt.evaluate(winner, config)
    test_bt.print_report()
    
    # Plots
    plot_results(train_bt, test_bt, winner, config)
    
    # Export trades
    export_trades(train_bt, test_bt)

# ============ EXPORT TRADES ============

def export_trades(train_bt: Backtester, test_bt: Backtester):
    """Export trade results to CSV"""
    try:
        # Train trades
        train_data = []
        for t in train_bt.trades:
            entry_time = train_bt.events.iloc[t.entry_idx]['Timestamp']
            exit_time = train_bt.events.iloc[t.exit_idx]['Timestamp']
            train_data.append({
                'Set': 'TRAIN',
                'Side': t.side,
                'Entry_Time': entry_time,
                'Entry_Price': t.entry_price,
                'Exit_Time': exit_time,
                'Exit_Price': t.exit_price,
                'PnL_Pips': t.pnl_pips,
                'PnL_USD': t.pnl_usd
            })
        
        # Test trades
        test_data = []
        for t in test_bt.trades:
            entry_time = test_bt.events.iloc[t.entry_idx]['Timestamp']
            exit_time = test_bt.events.iloc[t.exit_idx]['Timestamp']
            test_data.append({
                'Set': 'TEST',
                'Side': t.side,
                'Entry_Time': entry_time,
                'Entry_Price': t.entry_price,
                'Exit_Time': exit_time,
                'Exit_Price': t.exit_price,
                'PnL_Pips': t.pnl_pips,
                'PnL_USD': t.pnl_usd
            })
        
        # Combine and export
        all_trades = train_data + test_data
        df = pd.DataFrame(all_trades)
        
        filename = 'backtest_trades.csv'
        df.to_csv(filename, index=False)
        print(f"\nTrades exported: {filename}")
        print(f"Total trades: {len(all_trades)} (Train: {len(train_data)}, Test: {len(test_data)})")
        
    except Exception as e:
        print(f"Error exporting trades: {e}")

# ============ PLOTS ============

def plot_results(train_bt: Backtester, test_bt: Backtester, genome, config):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    # Equity curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Train Equity
    equity_train = [INITIAL_BALANCE]
    for t in train_bt.trades:
        equity_train.append(equity_train[-1] + t.pnl_usd)
    
    axes[0, 0].plot(equity_train, 'b-', linewidth=1)
    axes[0, 0].axhline(y=INITIAL_BALANCE, color='gray', linestyle='--')
    axes[0, 0].set_title(f'TRAIN Equity (PnL: ${train_bt.get_stats()["pnl"]:.0f})')
    axes[0, 0].set_xlabel('Trade #')
    axes[0, 0].set_ylabel('Equity ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. OOS Equity
    equity_oos = [INITIAL_BALANCE]
    for t in test_bt.trades:
        equity_oos.append(equity_oos[-1] + t.pnl_usd)
    
    axes[0, 1].plot(equity_oos, 'r-', linewidth=1)
    axes[0, 1].axhline(y=INITIAL_BALANCE, color='gray', linestyle='--')
    axes[0, 1].set_title(f'OOS Equity (PnL: ${test_bt.get_stats()["pnl"]:.0f})')
    axes[0, 1].set_xlabel('Trade #')
    axes[0, 1].set_ylabel('Equity ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Train Pips Distribution
    pips_train = [t.pnl_pips for t in train_bt.trades]
    axes[1, 0].hist(pips_train, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_title(f'TRAIN Pips (Avg: {np.mean(pips_train):.2f})')
    axes[1, 0].set_xlabel('Pips')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. OOS Pips Distribution
    pips_oos = [t.pnl_pips for t in test_bt.trades]
    axes[1, 1].hist(pips_oos, bins=20, color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--')
    axes[1, 1].set_title(f'OOS Pips (Avg: {np.mean(pips_oos):.2f})')
    axes[1, 1].set_xlabel('Pips')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print(f"\nPlot saved: results.png")
    plt.show()

# ============ LIVE ============

class DCStream:
    """
    Live DC Stream using SHARED FeatureCalculator.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.mode = 0
        self.extreme = None
        self.last_event_time = None
        
        # Use shared feature calculator - SAME CODE as backtest
        self.feature_calc = FeatureCalculator()
    
    def on_price(self, price: float):
        """Process a price tick, return features dict if DC triggered"""
        current_time = time.time()
        
        if self.extreme is None:
            self.extreme = price
            self.last_event_time = current_time
            return None
        
        event_type = None
        
        # DC detection logic (same as backtest dc_processor.py)
        if self.mode == 0:
            if price > self.extreme * (1 + self.threshold):
                self.mode = 1
                self.extreme = price
                event_type = "DC_UP"
            elif price < self.extreme * (1 - self.threshold):
                self.mode = -1
                self.extreme = price
                event_type = "DC_DOWN"
        elif self.mode == 1:
            if price > self.extreme:
                self.extreme = price
            elif price < self.extreme * (1 - self.threshold):
                self.mode = -1
                self.extreme = price
                event_type = "DC_DOWN"
        else:  # mode == -1
            if price < self.extreme:
                self.extreme = price
            elif price > self.extreme * (1 + self.threshold):
                self.mode = 1
                self.extreme = price
                event_type = "DC_UP"
        
        if event_type:
            # Calculate time gap
            time_gap = current_time - self.last_event_time if self.last_event_time else 1.0
            self.last_event_time = current_time
            
            # Use SHARED feature calculator - EXACT SAME as backtest
            dc_event = self.feature_calc.calculate(price, event_type, time_gap)
            
            return dc_event.to_dict()
        
        return None

def live():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found")
        return

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG
    )
    
    with open(MODEL_PATH, 'rb') as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    print(f"Connecting to {HOST}:{PORT}...")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    conn, addr = server.accept()
    print(f"Connected: {addr}")

    dc = DCStream(DC_THRESHOLD)
    position = None  # {'side': 'BUY'/'SELL', 'entry': float}
    buffer = b""

    while True:
        data = conn.recv(4096)
        if not data:
            print("Connection closed")
            break
        
        buffer += data
        
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            
            try:
                payload = json.loads(line.decode())
                symbol = payload.get("symbol", "UNKNOWN")
                bid = float(payload["bid"])
                ask = float(payload["ask"])
                price = (bid + ask) / 2
                
                # Calculate current spread
                spread_pips = (ask - bid) / PIP_SIZE
                
                event = dc.on_price(price)
                if not event:
                    continue
                
                # ============ DEBUG: Event count & warmup ============
                event_count = dc.feature_calc.event_count
                warmup_done = event_count >= WARMUP_EVENTS
                
                # Calculate unrealized pips for position
                in_position = position is not None
                unrealized_pips = 0.0
                
                if position:
                    if position['side'] == 'BUY':
                        unrealized_pips = (price - position['entry']) / PIP_SIZE
                    else:
                        unrealized_pips = (position['entry'] - price) / PIP_SIZE

                # Use SHARED normalization function - EXACT SAME as backtest
                features = normalize_features(
                    event_type=event['type'],
                    volatility=event['volatility'],
                    rsi=event['rsi'],
                    market_speed=event['market_speed'],
                    momentum=event['momentum'],
                    price_distance=event['price_distance'],
                    trend_bias=event['trend_bias'],
                    in_position=in_position,
                    unrealized_pips=unrealized_pips,
                    vol_scale_factor=VOL_SCALE_FACTOR,
                    pips_norm=PIPS_NORM
                )
                
                output = net.activate(features)
                
                buy_signal = output[0]
                sell_signal = output[1]
                close_signal = output[2]
                signal_strength = buy_signal - sell_signal
                
                # ============ DEBUG OUTPUT ============
                print(f"\n{'='*60}")
                print(f"{symbol} EVENT #{event_count} | {event['type']} @ {price:.5f} | Warmup: {'DONE' if warmup_done else f'{event_count}/{WARMUP_EVENTS}'}")
                print(f"{'='*60}")
                
                # Raw features
                print(f"RAW FEATURES:")
                print(f"  volatility:     {event['volatility']:.8f}")
                print(f"  rsi:            {event['rsi']:.2f}")
                print(f"  market_speed:   {event['market_speed']:.4f}")
                print(f"  momentum:       {event['momentum']:.4f}")
                print(f"  price_distance: {event['price_distance']:.4f}")
                print(f"  trend_bias:     {event['trend_bias']:.4f}")
                
                # Normalized features (what NN sees)
                print(f"NORMALIZED (NN INPUT):")
                print(f"  [0] event_type:    {features[0]:+.2f}")
                print(f"  [1] volatility:    {features[1]:+.4f}")
                print(f"  [2] rsi:           {features[2]:+.4f}")
                print(f"  [3] market_speed:  {features[3]:+.4f}")
                print(f"  [4] momentum:      {features[4]:+.4f}")
                print(f"  [5] price_dist:    {features[5]:+.4f}")
                print(f"  [6] trend_bias:    {features[6]:+.4f}")
                print(f"  [7] in_position:   {features[7]:.0f}")
                print(f"  [8] unreal_pips:   {features[8]:+.4f}")
                
                # NN output
                print(f"NN OUTPUT:")
                print(f"  buy={buy_signal:+.4f} | sell={sell_signal:+.4f} | close={close_signal:+.4f}")
                print(f"  signal_strength: {signal_strength:+.4f} (threshold: {OPEN_THRESHOLD})")
                
                # Skip trading during warmup
                if not warmup_done:
                    print(f">>> SKIPPING: Warmup not complete ({event_count}/{WARMUP_EVENTS})")
                    continue

                # SAFETY EXIT (adaptive SL)
                volatility = event['volatility']
                if position:
                    adaptive_sl = max(
                        MIN_SL_PIPS,
                        (volatility * price / PIP_SIZE) * VOL_SL_MULT
                    )
                    if unrealized_pips <= -adaptive_sl:
                        conn.sendall(b"CLOSE\n")
                        print(f">>> CLOSE (SL hit: {unrealized_pips:.1f} pips <= -{adaptive_sl:.1f})")
                        position = None
                        continue

                # CLOSE (model decision)
                if position and close_signal > CLOSE_THRESHOLD:
                    # Only close if spread is reasonable
                    if spread_pips > 2.0:
                        print(f">>> CLOSE BLOCKED: Wide spread ({spread_pips:.1f} pips)")
                        continue
                    conn.sendall(b"CLOSE\n")
                    print(f">>> CLOSE {position['side']} (close_signal={close_signal:.2f} > {CLOSE_THRESHOLD})")
                    position = None
                    continue

                # OPEN (only if spread is reasonable)
                # Rollover filter removed - spread filter is sufficient
                if not position and spread_pips <= 2.0:
                    if signal_strength > OPEN_THRESHOLD:
                        cmd = f"BUY {LOT_SIZE:.2f} 0 0\n"
                        conn.sendall(cmd.encode())
                        position = {'side': 'BUY', 'entry': price}
                        print(f">>> BUY @ {price:.5f} (signal={signal_strength:.2f} > {OPEN_THRESHOLD})")
                    elif signal_strength < -OPEN_THRESHOLD:
                        cmd = f"SELL {LOT_SIZE:.2f} 0 0\n"
                        conn.sendall(cmd.encode())
                        position = {'side': 'SELL', 'entry': price}
                        print(f">>> SELL @ {price:.5f} (signal={signal_strength:.2f} < -{OPEN_THRESHOLD})")
                    else:
                        print(f">>> NO TRADE: signal ({signal_strength:.2f}) within threshold")
                elif spread_pips > 2.0 and not position:
                    print(f">>> BLOCKED: Wide spread ({spread_pips:.1f} pips)")

            except Exception as e:
                import traceback
                print(f"Error: {e}")
                traceback.print_exc()

# ============ MAIN ============

if __name__ == '__main__':
    # Si argument passé en ligne de commande
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'train':
            train()
        elif mode == 'backtest':
            # Check for debug flag
            debug_mode = len(sys.argv) > 2 and sys.argv[2].lower() == 'debug'
            backtest(debug=debug_mode)
        elif mode == 'live':
            live()
        else:
            print("Usage: python 'Final Project.py' [train|backtest|backtest debug|live]")
    else:
        # Menu interactif
        print("\n" + "="*50)
        print("NEAT TRADING")
        print("="*50)
        print("\n1. Train")
        print("2. Backtest")
        print("3. Backtest (DEBUG)")
        print("4. Live")
        print("5. Exit")
        
        choice = input("\nChoix (1-5): ").strip()
        
        if choice == '1':
            train()
        elif choice == '2':
            backtest()
        elif choice == '3':
            backtest(debug=True)
        elif choice == '4':
            live()
        else:
            print("Exit.")
