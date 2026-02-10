"""
Directional Change Processor
Uses FeatureCalculator from shared_features.py for ALL feature calculations.
This ensures EXACT match between backtest and live.
"""

import pandas as pd
import numpy as np
import os
from shared_features import FeatureCalculator

class DCProcessor:
    """
    Processes tick data into Directional Change events.
    Uses shared FeatureCalculator for feature calculation.
    """
    
    def __init__(self, threshold: float = 0.0001, pip_size: float = 0.0001):
        self.theta = threshold
        self.threshold = threshold
        self.pip_size = pip_size
    
    def process_ticks(self, csv_path: str) -> pd.DataFrame:
        """Convert tick data to DC events with features"""
        print(f"\n{'='*80}")
        print(f"LOADING MARKET DATA: {csv_path}")
        print(f"{'='*80}")
        
        df = pd.read_csv(csv_path)
        df['Timestamp'] = pd.to_datetime(df['Date'])
        
        prices = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        timestamps = df['Timestamp'].values
        
        print(f"Total Ticks:          {len(prices):,}")
        print(f"DC Threshold:         {self.threshold:.5f} ({self.threshold*10000:.1f} pips)")
        
        # Generate DC events with features using SHARED FeatureCalculator
        events = self._generate_dc_events(prices, timestamps, highs, lows)
        
        print(f"\nGenerated Events:     {len(events):,}")
        print(f"Compression Ratio:    {len(prices)/len(events):.1f}:1")
        print(f"{'='*80}\n")
        
        return events
    
    def _generate_dc_events(self, prices: np.ndarray, timestamps: np.ndarray, 
                            highs: np.ndarray, lows: np.ndarray) -> pd.DataFrame:
        """Generate DC events and calculate features using shared calculator"""
        
        # Initialize shared feature calculator - SAME as live
        feature_calc = FeatureCalculator()
        
        current_ext = prices[0]
        mode = 0  # 0=undefined, 1=uptrend, -1=downtrend
        
        events = []
        last_event_time = timestamps[0]
        last_event_price = prices[0]
        
        for i in range(1, len(prices)):
            price = prices[i]
            time = timestamps[i]
            high = highs[i]
            low = lows[i]
            
            event_type = None
            
            # Directional Change Logic
            if mode == 0:
                if price > current_ext * (1 + self.threshold):
                    mode = 1
                    current_ext = price
                    event_type = 'DC_UP'
                elif price < current_ext * (1 - self.threshold):
                    mode = -1
                    current_ext = price
                    event_type = 'DC_DOWN'
            
            elif mode == 1:
                if price > current_ext:
                    current_ext = price
                elif price < current_ext * (1 - self.threshold):
                    event_type = 'DC_DOWN'
                    mode = -1
                    current_ext = price
            
            elif mode == -1:
                if price < current_ext:
                    current_ext = price
                elif price > current_ext * (1 + self.threshold):
                    event_type = 'DC_UP'
                    mode = 1
                    current_ext = price
            
            # Record event if triggered
            if event_type:
                time_gap = (time - last_event_time) / np.timedelta64(1, 's')
                
                # Use SHARED FeatureCalculator - EXACT SAME as live
                dc_event = feature_calc.calculate(price, event_type, time_gap)
                
                # Time features
                dt = pd.Timestamp(time)
                hour_rad = (dt.hour / 24.0) * 2 * np.pi
                
                # Spread from tick data
                spread_pips = (high - low) / self.pip_size
                
                events.append({
                    'Timestamp': time,
                    'Price': price,
                    'High': high,
                    'Low': low,
                    'Type': event_type,
                    'TimeGap': time_gap,
                    'Hour': dt.hour,
                    'Hour_Sin': np.sin(hour_rad),
                    'Hour_Cos': np.cos(hour_rad),
                    'Spread_Pips': spread_pips,
                    # Features from SHARED calculator
                    'Volatility': dc_event.volatility,
                    'RSI': dc_event.rsi,
                    'Market_Speed': dc_event.market_speed,
                    'Momentum': dc_event.momentum,
                    'Price_Distance': dc_event.price_distance,
                    'Trend_Bias': dc_event.trend_bias
                })
                
                last_event_time = time
                last_event_price = price
        
        return pd.DataFrame(events)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = r'C:\Users\thiba\Desktop\MARKET DATA\market_data_GBPAUD.csv'
    
    processor = DCProcessor(threshold=0.0005)
    events = processor.process_ticks(csv_path)
    print(events.head(20))
    print("\nFeatures available:")
    print(events.columns.tolist())
