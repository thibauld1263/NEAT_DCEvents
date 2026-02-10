"""
SHARED FEATURE CALCULATOR
=========================
ONE module for ALL feature calculations.
Used by: dc_processor.py (backtest) AND Final Project.py (live)

This eliminates ANY possibility of mismatch between backtest and live.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict

# ============ CONSTANTS ============
RSI_PERIOD = 14
EMA_SLOW_SPAN = 26
MOMENTUM_WINDOW = 10
TREND_WINDOW = 10
SPEED_WINDOW = 20  
VOL_WINDOW = 20
WARMUP_EVENTS = 20   # Minimum events before live trading 

@dataclass
class DCEvent:
    """A single Directional Change event with all features"""
    event_type: str  # "DC_UP" or "DC_DOWN"
    price: float
    volatility: float
    rsi: float
    market_speed: float  
    momentum: float  
    price_distance: float
    trend_bias: float  
    
    def to_dict(self) -> Dict:
        return {
            'type': self.event_type,
            'price': self.price,
            'volatility': self.volatility,
            'rsi': self.rsi,
            'market_speed': self.market_speed,
            'momentum': self.momentum,  
            'price_distance': self.price_distance,
            'trend_bias': self.trend_bias
        }

class FeatureCalculator:
    """
    Unified feature calculator for DC events.
    
    EXACTLY THE SAME LOGIC for backtest and live.
    """
    
    def __init__(self):
        # Price history for rolling calculations
        self.prices = deque(maxlen=200)  # Extra buffer for safety
        self.time_gaps = deque(maxlen=200)
        
        # EMA (exponential moving average)
        self.ema_slow = None
        
        # Market Speed (EMA of absolute velocity)
        self.speed_ema = None
        
        # Trend Bias (EMA of direction: +1/-1)
        self.trend_ema = 0.0
        self.last_event_type = None
        
        # Event counter
        self.event_count = 0
        
        # Momentum tracking
        self.momentum_history = deque(maxlen=MOMENTUM_WINDOW)
    
    def calculate(self, price: float, event_type: str, time_gap: float) -> DCEvent:
        """
        Calculate ALL features for a DC event.
        
        Args:
            price: Current price at DC event
            event_type: "DC_UP" or "DC_DOWN"
            time_gap: Seconds since last DC event
        
        Returns:
            DCEvent with all features calculated
        """
        self.event_count += 1
        self.prices.append(price)
        self.time_gaps.append(time_gap)
        
        # Calculate each feature
        volatility = self._calc_volatility()
        ema_slow = self._calc_ema_slow(price)
        rsi = self._calc_rsi()
        momentum = self._calc_momentum(time_gap)
        trend_bias = self._calc_trend_bias(event_type)
        market_speed = self._calc_market_speed(time_gap)
        price_distance = self._calc_price_distance(price, ema_slow, volatility)
        
        return DCEvent(
            event_type=event_type,
            price=price,
            volatility=volatility,
            rsi=rsi,
            market_speed=market_speed,
            momentum=momentum,
            price_distance=price_distance,
            trend_bias=trend_bias
        )
    
    def _calc_volatility(self) -> float:
        """
        Volatility = std of percent returns over last 20 events
        Uses ddof=1 (sample std) to match pandas
        """
        if len(self.prices) < 2:
            return 0.0
        
        # Need at least 2 prices to calculate 1 return
        prices_arr = np.array(self.prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]  # Percent returns
        
        if len(returns) < 2:
            return 0.0
        
        # Use last VOL_WINDOW returns (or all if less available)
        window = min(VOL_WINDOW, len(returns))
        vol = np.std(returns[-window:], ddof=1)  # Sample std to match pandas
        
        return float(vol)
    
    def _calc_ema_slow(self, price: float) -> float:
        """
        EMA_Slow = Exponential Moving Average with span=26
        Formula: alpha * price + (1-alpha) * previous_ema
        where alpha = 2 / (span + 1)
        """
        alpha = 2.0 / (EMA_SLOW_SPAN + 1.0)
        
        if self.ema_slow is None:
            self.ema_slow = price
        else:
            self.ema_slow = alpha * price + (1.0 - alpha) * self.ema_slow
        
        return self.ema_slow
    
    def _calc_rsi(self) -> float:
        """
        RSI = 100 - (100 / (1 + RS))
        where RS = avg_gain / avg_loss over last 14 periods
        """
        if len(self.prices) <= RSI_PERIOD:
            return 50.0  # Default during warmup
        
        # Get last RSI_PERIOD+1 prices to calculate RSI_PERIOD deltas
        prices_arr = np.array(list(self.prices)[-(RSI_PERIOD + 1):])
        deltas = np.diff(prices_arr)
        
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain > 0:
            rsi = 100.0  # All gains, no losses
        else:
            rsi = 50.0  # No movement
        
        return float(rsi)
    
    def _calc_momentum(self, time_gap: float) -> float:
        """
        Momentum = Average price velocity over last N events
        
        Velocity = (price_change / time_gap) for each event
        Positive = price moving up fast
        Negative = price moving down fast
        
        Normalized using tanh to [-1, 1]
        """
        if len(self.prices) < 2:
            return 0.0
        
        # Calculate current velocity (price change per second)
        price_change = self.prices[-1] - self.prices[-2]
        velocity = price_change / (time_gap + 1e-9)
        
        # Store velocity
        self.momentum_history.append(velocity)
        
        if len(self.momentum_history) < 2:
            return 0.0
        
        # Average velocity over window
        avg_velocity = float(np.mean(self.momentum_history))
        
        # Normalize: typical velocity is ~0.00001 per second for forex
        # Scale factor to get meaningful values
        normalized = np.tanh(avg_velocity * 100000)  # Scale and squash to [-1, 1]
        
        return float(normalized)
    
    def _calc_trend_bias(self, event_type: str) -> float:
        """
        Trend Bias = EMA of Direction (+1 or -1)
        
        Provides a smooth, noise-filtered indication of the current trend.
        Range: [-1, 1]
           +1 = Strong Uptrend
            0 = Neutral / Chop
           -1 = Strong Downtrend
        """
        # Direction value
        direction = 1.0 if event_type == "DC_UP" else -1.0
        
        # EMA calculation
        # Alpha 0.1 corresponds to ~19 event center of mass
        # It's responsive but not jumpy
        alpha = 0.1
        
        self.trend_ema = alpha * direction + (1.0 - alpha) * self.trend_ema
        
        return float(self.trend_ema)
    
    def _calc_market_speed(self, time_gap: float) -> float:
        """
        Market Speed = EMA of Absolute Velocity (Price Change / Time)
        
        Replaces bulky "Regime" calculation.
        Starts working immediately (no long history needed).
        Normalized via tanh to [0, 1].
        """
        if len(self.prices) < 2:
            return 0.0
            
        # Current velocity (abs)
        price_change = abs(self.prices[-1] - self.prices[-2])
        velocity = price_change / (time_gap + 1e-9)
        
        # EMA smoothing
        alpha = 2.0 / (SPEED_WINDOW + 1.0)
        
        if self.speed_ema is None:
            self.speed_ema = velocity
        else:
            self.speed_ema = alpha * velocity + (1.0 - alpha) * self.speed_ema
            
        # Normalize: 
        # Typical forex speed ~0.00001 (1 pip/sec is huge)
        # Scale by 100,000 matches momentum scale
        # Use tanh to squash to [0, 1] range (since abs velocity is positive)
        
        normalized = np.tanh(self.speed_ema * 100000)
        
        return float(normalized)
    
    def _calc_price_distance(self, price: float, ema_slow: float, volatility: float) -> float:
        """
        Price Distance = (Price - EMA_Slow) / Volatility
        Normalized to [-1, 1] by clipping to [-3, 3] then dividing by 3
        """
        if volatility <= 0:
            return 0.0
        
        distance = (price - ema_slow) / (volatility + 1e-9)
        distance = np.clip(distance, -3.0, 3.0) / 3.0
        
        return float(distance)
    
    def reset(self):
        """Reset all state (for running multiple backtests)"""
        self.prices.clear()
        self.time_gaps.clear()
        # self.regime_values.clear()  # Removed
        self.momentum_history.clear()
        self.ema_slow = None
        self.speed_ema = None
        self.trend_ema = 0.0
        self.last_event_type = None
        self.event_count = 0

# ============ NORMALIZATION FUNCTIONS ============
# These convert raw features to neural network inputs

def normalize_features(
    event_type: str,
    volatility: float,
    rsi: float,
    market_speed: float, # REPLACED regime
    momentum: float,  
    price_distance: float,
    trend_bias: float,
    in_position: bool,
    unrealized_pips: float,
    vol_scale_factor: float = 1e4,
    pips_norm: float = 10.0
) -> list:
    """
    Normalize features for neural network input.
    
    Returns list of 9 normalized features in order:
        0: event_type (1 for UP, -1 for DOWN)
        1: volatility_norm (tanh scaled)
        2: rsi_norm ([-1, 1])
        3: market_speed (tanh scaled [0, 1])  <-- CHANGED from regime
        4: momentum (already [-1, 1] from tanh)
        5: price_distance (already [-1, 1])
        6: trend_bias (already [-1, 1])
        7: in_position (0 or 1)
        8: unrealized_pips_norm (tanh scaled)
    """
    return [
        1.0 if event_type == "DC_UP" else -1.0,      # 0: event_type
        float(np.tanh(volatility * vol_scale_factor)), # 1: volatility_norm
        (rsi - 50.0) / 50.0,                          # 2: rsi_norm [-1, 1]
        market_speed,                                 # 3: market_speed [0, 1]
        momentum,                                      # 4: momentum (already normalized)
        price_distance,                               # 5: price_distance
        trend_bias,                                   # 6: trend_bias
        1.0 if in_position else 0.0,                  # 7: in_position
        float(np.tanh(unrealized_pips / pips_norm))   # 8: unrealized_pips_norm
    ]
