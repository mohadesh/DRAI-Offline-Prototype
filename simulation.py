"""
موتور شبیه‌سازی — برای شبیه‌سازی جریان داده زنده از داده‌های آفلاین.
"""

import pandas as pd
from typing import Optional, Dict, Any
import numpy as np


class SimulationRunner:
    """
    کلاس برای شبیه‌سازی جریان داده زنده از یک DataFrame آفلاین.
    
    این کلاس داده‌های ادغام شده را به صورت گام به گام (هر 5 دقیقه) ارائه می‌دهد
    تا حالت "زنده" را شبیه‌سازی کند.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize simulation runner with merged dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Merged dataframe with georgian_datetime index or column
        """
        self.df = df.copy()
        
        # Ensure georgian_datetime is a column and set as index for easy access
        if 'georgian_datetime' in self.df.columns:
            self.df['georgian_datetime'] = pd.to_datetime(self.df['georgian_datetime'])
            self.df = self.df.sort_values('georgian_datetime').reset_index(drop=True)
        else:
            raise ValueError("DataFrame must have 'georgian_datetime' column")
        
        # Current position in the dataframe
        self.current_index = 0
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        
        # Store original length
        self.total_rows = len(self.df)
    
    def reset(self):
        """Reset simulation to start from beginning."""
        self.current_index = 0
        self.is_running = False
        self.is_paused = False
    
    def start(self):
        """Start the simulation."""
        self.is_running = True
        self.is_paused = False
    
    def pause(self):
        """Pause the simulation."""
        self.is_paused = True
    
    def resume(self):
        """Resume the simulation."""
        self.is_paused = False
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        self.is_paused = False
    
    def get_next_step(self) -> Optional[Dict[str, Any]]:
        """
        Get the next row of data (simulating a 5-minute step).
        
        Returns:
        --------
        dict with current row data, or None if simulation is complete
        """
        if not self.is_running or self.is_paused:
            return None
        
        if self.current_index >= self.total_rows:
            # Simulation complete
            self.is_running = False
            return None
        
        # Get current row
        row = self.df.iloc[self.current_index].to_dict()
        
        # Convert numpy types to Python native types for JSON serialization
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                row[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                row[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                row[key] = str(value)
            elif isinstance(value, pd.Timedelta):
                row[key] = str(value)
        
        # Increment index for next call
        self.current_index += 1
        
        return row
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """
        Get current row without advancing (for polling).
        
        Returns:
        --------
        dict with current row data, or None if simulation is complete
        """
        if self.current_index >= self.total_rows:
            return None
        
        row = self.df.iloc[self.current_index].to_dict()
        
        # Convert numpy types to Python native types
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (np.integer, np.int64)):
                row[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                row[key] = float(value)
            elif isinstance(value, pd.Timestamp):
                row[key] = str(value)
            elif isinstance(value, pd.Timedelta):
                row[key] = str(value)
        
        return row
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get simulation progress information.
        
        Returns:
        --------
        dict with progress stats
        """
        progress_pct = (self.current_index / self.total_rows * 100) if self.total_rows > 0 else 0
        
        return {
            'current_index': self.current_index,
            'total_rows': self.total_rows,
            'progress_percent': round(progress_pct, 1),
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'is_complete': self.current_index >= self.total_rows
        }
    
    def jump_to_index(self, index: int):
        """
        Jump to a specific index in the simulation.
        
        Parameters:
        -----------
        index : int
            Target index (0-based)
        """
        if 0 <= index < self.total_rows:
            self.current_index = index
        else:
            raise ValueError(f"Index {index} out of range [0, {self.total_rows})")

