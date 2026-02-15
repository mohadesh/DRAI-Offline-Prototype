"""
هسته پردازش داده و اینفرنس — فقط توابع خالص پایتون.
بدون وابستگی به Flask.

این ماژول منطق ادغام سه فایل CSV (Process Tags, Pellet, MD/Quality) را پیاده‌سازی می‌کند
و محاسبه زمان اقامت (Residence Time) را انجام می‌دهد.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
import warnings
import jdatetime

warnings.filterwarnings('ignore')


def parse_jalali_datetime(row):
    """Convert Jalali date columns + time to datetime object"""
    try:
        j_date = jdatetime.date(int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
        g_date = j_date.togregorian()
        t_str = str(row['Time']).strip()
        if len(t_str.split(':')) == 2:
            t_str += ":00"
        return datetime.combine(g_date, datetime.strptime(t_str, "%H:%M:%S").time())
    except:
        return pd.NaT


def datetime_to_jalali_string(dt):
    """Convert datetime to Jalali string format"""
    try:
        j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
        return j_dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None


def load_process_tags(file_path: str) -> pd.DataFrame:
    """
    Load and process Process Tags CSV file.
    Expected format: CSV with Jalali date columns (YEAR, MONTH, DAY, Time) and process tags.
    """
    df = pd.read_csv(file_path, delimiter=';')
    
    # Create georgian_datetime
    df['georgian_datetime'] = df.apply(parse_jalali_datetime, axis=1)
    df = df.dropna(subset=['georgian_datetime'])
    
    # Create Jalali string
    df['jalali_datetime_str'] = df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    # Set index for merging
    df = df.set_index('georgian_datetime')
    
    # Remove metadata columns
    metadata_cols = ['ID', 'DAY', 'MONTH', 'YEAR', 'Date', 'Time']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    return df


def load_pellet_data(file_path: str) -> pd.DataFrame:
    """
    Load Pellet CSV file.
    Expected format: CSV with georgian_datetime column or Jalali date columns.
    """
    df = pd.read_csv(file_path)
    
    # If georgian_datetime exists, use it
    if 'georgian_datetime' in df.columns:
        df['georgian_datetime'] = pd.to_datetime(df['georgian_datetime'])
    # Otherwise, try to create from Jalali columns
    elif all(col in df.columns for col in ['Year', 'Month', 'Day', 'Time']):
        df['georgian_datetime'] = df.apply(
            lambda row: _create_datetime_from_jalali(row), axis=1
        )
        df = df.dropna(subset=['georgian_datetime'])
    else:
        raise ValueError("Pellet file must have 'georgian_datetime' or Jalali date columns")
    
    # Create Jalali string if not exists
    if 'jalali_datetime_str' not in df.columns:
        df['jalali_datetime_str'] = df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    # Set index
    df = df.set_index('georgian_datetime')
    
    # Remove metadata columns
    exclude_cols = ['jalali_datetime_str', 'Year', 'Month', 'Day', 'Time', 'Source_File']
    df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    return df


def load_mdnc_data(file_path: str) -> pd.DataFrame:
    """
    Load MD/Quality CSV file.
    Expected format: CSV with georgian_datetime column.
    """
    df = pd.read_csv(file_path)
    
    if 'georgian_datetime' not in df.columns:
        raise ValueError("MD/Quality file must have 'georgian_datetime' column")
    
    df['georgian_datetime'] = pd.to_datetime(df['georgian_datetime'])
    df = df.dropna(subset=['georgian_datetime'])
    
    # Create Jalali string if not exists
    if 'jalali_datetime_str' not in df.columns:
        df['jalali_datetime_str'] = df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    # Set index
    df = df.set_index('georgian_datetime')
    
    # Remove metadata columns
    exclude_cols = ['jalali_datetime_str', 'time', 'source_file']
    df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    return df


def _create_datetime_from_jalali(row):
    """Helper to create datetime from Jalali date columns"""
    try:
        from persiantools.jdatetime import JalaliDate
        j_date = JalaliDate(int(row['Year']), int(row['Month']), int(row['Day']))
        g_date = j_date.to_gregorian()
        time_str = str(row['Time']).strip()
        time_parts = time_str.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        return datetime(g_date.year, g_date.month, g_date.day, hour, minute)
    except:
        return pd.NaT


def calculate_residence_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate enter and exit times based on WTH15 (Production Rate).
    
    Formulas:
    - enter_deltatime_hours = 600 / ((WTH15 + 270) / 2)
    - exit_deltatime_hours = 1200 / ((WTH15 + 270) / 2)
    
    Returns DataFrame with additional columns:
    - enter_deltatime_hours
    - exit_deltatime_hours
    - enter_georgian_datetime
    - exit_georgian_datetime
    """
    df = df.copy()
    
    if 'WTH15' not in df.columns:
        # If WTH15 not found, try INST_WTH15
        if 'INST_WTH15' in df.columns:
            df['WTH15'] = df['INST_WTH15']
        else:
            raise ValueError("WTH15 column not found. Cannot calculate residence times.")
    
    # Calculate denominators
    denominator = (df['WTH15'] + 270) / 2
    denominator = denominator.where(denominator > 0)  # Protect against division by zero
    
    # Calculate residence times in hours
    df['enter_deltatime_hours'] = 600 / denominator
    df['exit_deltatime_hours'] = 1200 / denominator
    
    # Convert to timedelta
    df['enter_deltatime'] = pd.to_timedelta(df['enter_deltatime_hours'], unit='h', errors='coerce')
    df['exit_deltatime'] = pd.to_timedelta(df['exit_deltatime_hours'], unit='h', errors='coerce')
    
    # Calculate enter and exit datetimes
    df['enter_georgian_datetime'] = df.index - df['enter_deltatime']
    df['exit_georgian_datetime'] = df.index + df['exit_deltatime']
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data based on business rules.
    - If WTH15 < 80, mark as "Shutdown/Startup" (invalid for model)
    
    Returns DataFrame with additional 'status' column:
    - 'Normal': WTH15 >= 80
    - 'Shutdown': WTH15 < 80 or missing
    """
    df = df.copy()
    
    # Determine WTH15 column
    wth15_col = None
    if 'WTH15' in df.columns:
        wth15_col = 'WTH15'
    elif 'INST_WTH15' in df.columns:
        wth15_col = 'INST_WTH15'
    
    if wth15_col:
        df['status'] = df[wth15_col].apply(
            lambda x: 'Normal' if pd.notna(x) and x >= 80 else 'Shutdown'
        )
    else:
        df['status'] = 'Shutdown'  # If WTH15 not found, mark as shutdown
    
    return df


def merge_three_csvs(
    process_tags_path: str,
    pellet_path: str,
    mdnc_path: str,
    time_rate: str = '5T',
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge three CSV files (Process Tags, Pellet, MD/Quality) into a master dataset.
    
    Parameters:
    -----------
    process_tags_path : str
        Path to Process Tags CSV file
    pellet_path : str
        Path to Pellet CSV file
    mdnc_path : str
        Path to MD/Quality CSV file
    time_rate : str
        Time frequency for reference grid (default: '5T' for 5 minutes)
    start_datetime : str, optional
        Start datetime in format 'YYYY-MM-DD HH:MM:SS'
    end_datetime : str, optional
        End datetime in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset with all data aligned to time grid
    """
    print("\n" + "="*80)
    print("MERGING THREE CSV FILES")
    print("="*80)
    
    # Load individual files
    print("\n[Step 1] Loading Process Tags...")
    process_df = load_process_tags(process_tags_path)
    print(f"  ✓ Loaded {len(process_df):,} rows")
    print(f"  ✓ Columns: {len(process_df.columns)}")
    
    print("\n[Step 2] Loading Pellet data...")
    pellet_df = load_pellet_data(pellet_path)
    print(f"  ✓ Loaded {len(pellet_df):,} rows")
    print(f"  ✓ Columns: {len(pellet_df.columns)}")
    
    print("\n[Step 3] Loading MD/Quality data...")
    mdnc_df = load_mdnc_data(mdnc_path)
    print(f"  ✓ Loaded {len(mdnc_df):,} rows")
    print(f"  ✓ Columns: {len(mdnc_df.columns)}")
    
    # Determine time range
    all_dates = []
    if not process_df.empty:
        all_dates.extend([process_df.index.min(), process_df.index.max()])
    if not pellet_df.empty:
        all_dates.extend([pellet_df.index.min(), pellet_df.index.max()])
    if not mdnc_df.empty:
        all_dates.extend([mdnc_df.index.min(), mdnc_df.index.max()])
    
    if not all_dates:
        raise ValueError("No valid datetime data found in any file")
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    # Use provided dates or auto-detect
    if start_datetime:
        start_dt = pd.to_datetime(start_datetime)
    else:
        start_dt = min_date
    
    if end_datetime:
        end_dt = pd.to_datetime(end_datetime)
    else:
        end_dt = max_date
    
    print(f"\n[Step 4] Creating reference time grid...")
    print(f"  • Start: {start_dt}")
    print(f"  • End: {end_dt}")
    print(f"  • Frequency: {time_rate}")
    
    # Create reference time grid
    time_index = pd.date_range(start=start_dt, end=end_dt, freq=time_rate)
    master_df = pd.DataFrame(index=time_index)
    master_df['georgian_datetime'] = time_index
    
    # Merge Process Tags (prefix: INST_)
    print(f"\n[Step 5] Merging Process Tags...")
    for col in process_df.columns:
        master_df[f'INST_{col}'] = process_df[col]
    print(f"  ✓ Merged {len(process_df.columns)} columns with 'INST_' prefix")
    
    # Merge Pellet data (prefix: PELLET_)
    print(f"\n[Step 6] Merging Pellet data...")
    for col in pellet_df.columns:
        master_df[f'PELLET_{col}'] = pellet_df[col]
    print(f"  ✓ Merged {len(pellet_df.columns)} columns with 'PELLET_' prefix")
    
    # Merge MD/Quality data (prefix: MDNC_)
    print(f"\n[Step 7] Merging MD/Quality data...")
    for col in mdnc_df.columns:
        master_df[f'MDNC_{col}'] = mdnc_df[col]
    print(f"  ✓ Merged {len(mdnc_df.columns)} columns with 'MDNC_' prefix")
    
    # Add Jalali datetime string
    if 'jalali_datetime_str' in process_df.columns:
        master_df['jalali_datetime_str'] = process_df['jalali_datetime_str']
    else:
        master_df['jalali_datetime_str'] = master_df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    # Calculate residence times
    print(f"\n[Step 8] Calculating residence times...")
    master_df = calculate_residence_times(master_df)
    print(f"  ✓ Calculated enter and exit times")
    
    # Validate data
    print(f"\n[Step 9] Validating data...")
    master_df = validate_data(master_df)
    print(f"  ✓ Validation complete")
    
    # Reset index to make georgian_datetime a column
    master_df = master_df.reset_index(drop=True)
    
    # Reorder columns
    datetime_cols = ['georgian_datetime', 'jalali_datetime_str']
    other_cols = [col for col in master_df.columns if col not in datetime_cols]
    master_df = master_df[datetime_cols + other_cols]
    
    print(f"\n[Step 10] Final dataset:")
    print(f"  ✓ Shape: {master_df.shape}")
    print(f"  ✓ Total time points: {len(master_df):,}")
    
    # Show status distribution
    if 'status' in master_df.columns:
        status_counts = master_df['status'].value_counts()
        print(f"  ✓ Status distribution:")
        for status, count in status_counts.items():
            print(f"    • {status}: {count:,} ({count/len(master_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80 + "\n")
    
    return master_df


def load_model(model_path: str):
    """Load model from .pkl file using joblib."""
    import joblib
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def run_inference_for_md_c(model_md, model_c, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run inference to predict MD and C values.
    
    Parameters:
    -----------
    model_md : model object or None
        Model for predicting MD
    model_c : model object or None
        Model for predicting C
    df : pd.DataFrame
        Input features dataframe
    
    Returns:
    --------
    dict with 'MD' and 'C' predictions
    """
    predictions = {'MD': None, 'C': None}
    
    # Get feature columns (exclude datetime and status columns)
    exclude_cols = ['georgian_datetime', 'jalali_datetime_str', 'status', 
                   'enter_deltatime_hours', 'exit_deltatime_hours',
                   'enter_deltatime', 'exit_deltatime',
                   'enter_georgian_datetime', 'exit_georgian_datetime']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        return predictions
    
    # Select numeric columns only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return predictions
    
    # Prepare feature matrix (use last row for single prediction)
    X = df[numeric_cols].iloc[-1:].values
    
    # Predict MD
    if model_md is not None:
        try:
            if hasattr(model_md, 'predict'):
                pred_md = model_md.predict(X)[0]
                predictions['MD'] = float(pred_md)
        except Exception as e:
            print(f"Error predicting MD: {e}")
    
    # Predict C
    if model_c is not None:
        try:
            if hasattr(model_c, 'predict'):
                pred_c = model_c.predict(X)[0]
                predictions['C'] = float(pred_c)
        except Exception as e:
            print(f"Error predicting C: {e}")
    
    return predictions


def process_data(
    process_tags_path: str,
    pellet_path: str,
    mdnc_path: str,
    model_md_path: Optional[str] = None,
    model_c_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main processing function: merge CSVs, validate, and optionally run inference.
    
    Returns:
    --------
    dict with 'merged_df', 'stats', and optionally 'predictions'
    """
    # Merge the three CSV files
    merged_df = merge_three_csvs(process_tags_path, pellet_path, mdnc_path)
    
    # Load models if provided
    model_md = None
    model_c = None
    if model_md_path:
        model_md = load_model(model_md_path)
    if model_c_path:
        model_c = load_model(model_c_path)
    
    # Run inference on the full dataset (optional)
    predictions = None
    if model_md or model_c:
        predictions = run_inference_for_md_c(model_md, model_c, merged_df)
    
    stats = {
        'rows': len(merged_df),
        'columns': len(merged_df.columns),
        'date_range': {
            'start': str(merged_df['georgian_datetime'].min()),
            'end': str(merged_df['georgian_datetime'].max())
        }
    }
    
    return {
        'merged_df': merged_df,
        'stats': stats,
        'predictions': predictions
    }
