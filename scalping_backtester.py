"""
NEAT Backtester
"""

import neat
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from shared_features import normalize_features, WARMUP_EVENTS

# Configs are injected from Final Project.py

@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    side: str
    pnl_pips: float
    pnl_usd: float

class Backtester:
    def __init__(
        self,
        events: pd.DataFrame,
        initial_balance: float = 10000.0,
        pip_value: float = 10.0,
        pip_size: float = 0.0001,
        vol_scale_factor: float = 1e4,
        pips_norm: float = 10.0,
        open_threshold: float = 0.2,
        close_threshold: float = 0.4,
        min_sl_pips: float = 15.0,
        vol_sl_mult: float = 2.5,
        debug: bool = False  # Enable verbose debug output
    ):
        self.events = events
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.pip_size = pip_size
        self.vol_scale_factor = vol_scale_factor
        self.pips_norm = pips_norm
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.min_sl_pips = min_sl_pips
        self.vol_sl_mult = vol_sl_mult
        self.debug = debug
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = None  # None ou {'side': 'BUY'/'SELL', 'entry_price': float, 'entry_idx': int}
        self.trades: List[Trade] = []

    def evaluate(self, genome, config) -> float:
        """
        Evaluate NEAT genome.
        
        Inputs (9) - compact + normalized:
            0: event_type (1=UP, -1=DOWN)
            1: volatility_norm (tanh)
            2: rsi_norm ([-1, 1])
            3: market_speed (tanh [0, 1])
            4: momentum ([-1, 1])
            5: price_distance ([-1, 1])
            6: trend_bias ([-1, 1])
            7: in_position (0/1)
            8: unrealized_pips_norm (tanh)
        
        Outputs (3):
            0: buy_signal
            1: sell_signal
            2: close_signal
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.reset()

        for idx in range(len(self.events)):
            event = self.events.iloc[idx]
            price = event['Price']
            event_type = 1.0 if event.get('Type') == 'DC_UP' else -1.0
            volatility = event.get('Volatility', 0.0)
            rsi = event.get('RSI', 50.0)
            market_speed = event.get('Market_Speed', 0.0)
            momentum = event.get('Momentum', 0.0)  
            price_distance = event.get('Price_Distance', 0.0)
            trend_bias = event.get('Trend_Bias', 0.0)
            
            # WARMUP SKIP: Match live trading behavior
            # FeatureCalculator starts counting at 1, so idx+1 is event_count
            if (idx + 1) < WARMUP_EVENTS:
                if self.debug:
                    print(f"Skipping event {idx+1}/{WARMUP_EVENTS} (Warmup)")
                continue

            # Spread check
            spread_pips = event.get('Spread_Pips', 0.0)
            
            # Rollover hour filter (21h-01h UTC / 9 PM - 1 AM) - BACKTEST ONLY - DEFAULT FROM CTRADER
            # This crosses midnight, so we need OR logic: hour >= 21 OR hour < 1
            event_hour = event.get('Hour', 12)  # Default to 12 if missing
            is_rollover_hour = (event_hour >= 21 or event_hour < 1)

            # Features
            in_position = 1.0 if self.position else 0.0
            position_side = 0.0
            unrealized_pips = 0.0
            
            if self.position:
                position_side = 1.0 if self.position['side'] == 'BUY' else -1.0
                if self.position['side'] == 'BUY':
                    unrealized_pips = (price - self.position['entry_price']) / self.pip_size
                else:
                    unrealized_pips = (self.position['entry_price'] - price) / self.pip_size

            # Use SHARED normalization 
            features = normalize_features(
                event_type=event.get('Type', 'DC_DOWN'),
                volatility=volatility,
                rsi=rsi,
                market_speed=market_speed,
                momentum=momentum,
                price_distance=price_distance,
                trend_bias=trend_bias,
                in_position=self.position is not None,
                unrealized_pips=unrealized_pips,
                vol_scale_factor=self.vol_scale_factor,
                pips_norm=self.pips_norm
            )
            output = net.activate(features)

            buy_signal = output[0]
            sell_signal = output[1]
            close_signal = output[2]

            signal_strength = buy_signal - sell_signal
            
            # ============ DEBUG OUTPUT ============
            if self.debug:
                event_type_str = event.get('Type', 'DC_DOWN')
                print(f"\n{'='*60}")
                print(f"EVENT #{idx} | {event_type_str} @ {price:.5f}")
                print(f"{'='*60}")
                
                # Raw features
                print(f"RAW FEATURES:")
                print(f"  volatility:     {volatility:.8f}")
                print(f"  rsi:            {rsi:.2f}")
                print(f"  market_speed:   {market_speed:.4f}")
                print(f"  momentum:       {momentum:.4f}")
                print(f"  price_distance: {price_distance:.4f}")
                print(f"  trend_bias:     {trend_bias:.4f}")
                print(f"  spread_pips:    {spread_pips:.2f}")
                print(f"  hour:           {event_hour} (rollover: {is_rollover_hour})")
                
                # Normalized features
                print(f"NORMALIZED (NN INPUT):")
                for i, f in enumerate(features):
                    print(f"  [{i}]: {f:+.4f}")
                
                # NN output
                print(f"NN OUTPUT:")
                print(f"  buy={buy_signal:+.4f} | sell={sell_signal:+.4f} | close={close_signal:+.4f}")
                print(f"  signal_strength: {signal_strength:+.4f} (threshold: {self.open_threshold})")

            # 1. SAFETY EXIT (adaptive SL)
            if self.position:
                adaptive_sl = max(
                    self.min_sl_pips,
                    (volatility * price / self.pip_size) * self.vol_sl_mult
                )
                if unrealized_pips <= -adaptive_sl:
                    if self.debug:
                        print(f">>> CLOSE (SL hit: {unrealized_pips:.1f} <= -{adaptive_sl:.1f})")
                    self._close_position(idx, price, reason="SL")
                    continue

            # 2. CLOSE (model decision)
            if self.position and close_signal > self.close_threshold:
                # Only close if spread is reasonable
                if spread_pips > 2.0:
                    if self.debug:
                        print(f">>> CLOSE BLOCKED: Wide spread ({spread_pips:.1f} pips)")
                    continue  # Skip close during wide spread
                if self.debug:
                    print(f">>> CLOSE (signal: {close_signal:.2f} > {self.close_threshold})")
                self._close_position(idx, price, reason="SIGNAL")
                continue

            # 3. OPEN (only if spread is reasonable AND not rollover hour)
            if not self.position and spread_pips <= 2.0 and not is_rollover_hour:
                if signal_strength > self.open_threshold:
                    if self.debug:
                        print(f">>> BUY (signal: {signal_strength:.2f} > {self.open_threshold})")
                    self._open_position(idx, price, 'BUY')
                elif signal_strength < -self.open_threshold:
                    if self.debug:
                        print(f">>> SELL (signal: {signal_strength:.2f} < -{self.open_threshold})")
                    self._open_position(idx, price, 'SELL')
                elif self.debug:
                    print(f">>> NO TRADE: signal ({signal_strength:.2f}) within threshold")
            elif self.debug and not self.position:
                if is_rollover_hour:
                    print(f">>> BLOCKED: Rollover hour ({event_hour})")
                elif spread_pips > 2.0:
                    print(f">>> BLOCKED: Wide spread ({spread_pips:.1f} pips)")

        # Fermer position ouverte à la fin
        if self.position:
            final_price = self.events.iloc[-1]['Price']
            self._close_position(len(self.events) - 1, final_price, reason="END")

        # Fitness = PnL total (pénalité si trop peu de trades)
        if len(self.trades) < 5:
            return -10000.0
        
        return self.balance - self.initial_balance

    def _open_position(self, idx: int, price: float, side: str):
        self.position = {
            'side': side,
            'entry_price': price,
            'entry_idx': idx
        }

    def _close_position(self, idx: int, price: float, reason: str = ""):
        if not self.position:
            return
        
        entry_price = self.position['entry_price']
        side = self.position['side']
        
        if side == 'BUY':
            pnl_pips = (price - entry_price) / self.pip_size
        else:
            pnl_pips = (entry_price - price) / self.pip_size
        
        pnl_usd = pnl_pips * self.pip_value
        self.balance += pnl_usd

        self.trades.append(Trade(
            entry_idx=self.position['entry_idx'],
            exit_idx=idx,
            entry_price=entry_price,
            exit_price=price,
            side=side,
            pnl_pips=pnl_pips,
            pnl_usd=pnl_usd
        ))
        
        self.position = None

    def get_stats(self) -> dict:
        if not self.trades:
            return {
                'trades': 0,
                'pnl': 0.0,
                'win_rate': 0.0,
                'avg_pips': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        wins = [t for t in self.trades if t.pnl_pips > 0]
        max_dd, max_dd_pct = self._compute_drawdown()
        return {
            'trades': len(self.trades),
            'pnl': self.balance - self.initial_balance,
            'win_rate': len(wins) / len(self.trades) * 100,
            'avg_pips': np.mean([t.pnl_pips for t in self.trades]),
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct
        }

    def print_report(self):
        stats = self.get_stats()
        print(f"Trades: {stats['trades']}")
        print(f"PnL: ${stats['pnl']:.2f}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Avg Pips: {stats['avg_pips']:.2f}")
        print(f"Max Drawdown: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']*100:.1f}%)")

    def _compute_drawdown(self) -> tuple:
        equity = self.get_equity_curve()

        peak = equity[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        for value in equity:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd / peak if peak > 0 else 0.0

        return max_dd, max_dd_pct

    def get_equity_curve(self) -> List[float]:
        equity = [self.initial_balance]
        for t in self.trades:
            equity.append(equity[-1] + t.pnl_usd)
        return equity
