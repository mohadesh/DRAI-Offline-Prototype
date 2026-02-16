"""
هسته پردازش داده و اینفرنس — فقط توابع خالص پایتون.
بدون وابستگی به Flask.

این ماژول منطق ادغام چندین فایل CSV (Process Tags, Pellet, MD/Quality) را پیاده‌سازی می‌کند
با پشتیبانی از آپلود دسته‌ای و مدیریت خودکار encoding/delimiter.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
import warnings
import jdatetime
from persiantools.jdatetime import JalaliDate
import os

warnings.filterwarnings('ignore')


def read_single_file_robust(file_path: str, delimiter: str = None) -> Optional[pd.DataFrame]:
    """
    Read a single CSV or Excel file with robust encoding and delimiter detection.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV or Excel file
    delimiter : str, optional
        Specific delimiter to try (if None, tries both ',' and ';')
    
    Returns:
    --------
    pd.DataFrame or None
        Loaded dataframe, or None if file couldn't be read
    """
    # Check if it's an Excel file
    if file_path.lower().endswith(('.xlsx', '.xls')):
        try:
            # Try reading as Excel
            df = pd.read_excel(file_path, engine='openpyxl', header=0)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print(f"    ⚠ Could not read as Excel: {e}")
            # Fall through to try as CSV
    
    # Try as CSV
    encodings = ['utf-8', 'cp1256', 'latin1', 'utf-16', 'utf-8-sig', 'windows-1252', 'iso-8859-1']
    delimiters = [delimiter] if delimiter else [',', ';', '\t']
    
    for encoding in encodings:
        for sep in delimiters:
            try:
                # Try with C engine first (faster)
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=sep,
                        engine='c',
                        on_bad_lines='skip',
                        low_memory=False
                    )
                except (TypeError, ValueError):
                    # Fallback for pandas < 2.0
                    try:
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            sep=sep,
                            engine='c',
                            error_bad_lines=False,
                            warn_bad_lines=False,
                            low_memory=False
                        )
                    except Exception:
                        # Try Python engine (more forgiving)
                        try:
                            df = pd.read_csv(
                                file_path,
                                encoding=encoding,
                                sep=sep,
                                engine='python',
                                on_bad_lines='skip',
                                low_memory=False
                            )
                        except (TypeError, ValueError):
                            df = pd.read_csv(
                                file_path,
                                encoding=encoding,
                                sep=sep,
                                engine='python',
                                error_bad_lines=False,
                                warn_bad_lines=False,
                                low_memory=False
                            )
                
                # Check if we got valid data
                if df is not None and len(df) > 0 and len(df.columns) > 0:
                    return df
                    
            except (UnicodeDecodeError, UnicodeError):
                continue
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                continue
            except Exception as e:
                # For other errors, log and continue
                if 'codec' not in str(e).lower() and 'decode' not in str(e).lower():
                    continue
                continue
    
    return None


def load_and_stack_csvs(file_list: List[Any], file_type: str = "unknown") -> pd.DataFrame:
    """
    Load and stack multiple CSV files into one DataFrame.
    
    Parameters:
    -----------
    file_list : List
        List of file objects (from Flask request.files) or file paths
    file_type : str
        Type of files being loaded (for logging): "process", "pellet", "md"
    
    Returns:
    --------
    pd.DataFrame
        Stacked dataframe with all data combined
    """
    dataframes = []
    successful_files = 0
    failed_files = 0
    
    print(f"\n[Loading {file_type.upper()} files]")
    print(f"  • Total files to process: {len(file_list)}")
    
    for idx, file_item in enumerate(file_list, 1):
        # Handle both file objects and file paths
        if hasattr(file_item, 'filename') and hasattr(file_item, 'save'):
            # It's a Flask file object - save temporarily
            temp_path = None
            try:
                temp_path = f"/tmp/drai_upload_{os.getpid()}_{idx}.csv"
                file_item.save(temp_path)
                file_path = temp_path
            except Exception as e:
                print(f"  ✗ File {idx}: Could not save temporary file: {e}")
                failed_files += 1
                continue
        else:
            # It's already a file path
            file_path = str(file_item)
        
        # Try to read the file
        df = read_single_file_robust(file_path)
        
        # Clean up temp file if created
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        if df is not None and len(df) > 0:
            # Add source file column for tracking
            filename = getattr(file_item, 'filename', str(file_item))
            df['_source_file'] = filename
            dataframes.append(df)
            successful_files += 1
            print(f"  ✓ File {idx}/{len(file_list)}: {filename} ({len(df)} rows, {len(df.columns)} cols)")
        else:
            failed_files += 1
            filename = getattr(file_item, 'filename', str(file_item))
            print(f"  ✗ File {idx}/{len(file_list)}: {filename} (empty or unreadable)")
    
    if not dataframes:
        raise ValueError(f"No valid data found in {file_type} files. All {len(file_list)} files failed to load.")
    
    # Stack all dataframes
    print(f"\n[Stacking {file_type.upper()} data]")
    print(f"  • Successful: {successful_files} files")
    print(f"  • Failed: {failed_files} files")
    
    try:
        stacked_df = pd.concat(dataframes, ignore_index=True, sort=False)
        print(f"  ✓ Stacked into {len(stacked_df):,} rows, {len(stacked_df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Error stacking {file_type} files: {str(e)}")
    
    return stacked_df


def convert_jalali_to_georgian(df: pd.DataFrame, file_type: str = "unknown") -> pd.DataFrame:
    """
    Convert Jalali (Shamsi) date columns to Gregorian datetime.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Jalali date columns
    file_type : str
        Type of file (for logging)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with georgian_datetime column added
    """
    print(f"\n[Converting Jalali dates to Gregorian - {file_type.upper()}]")
    
    # Try to find date/time columns (case-insensitive)
    date_col = None
    time_col = None
    year_col = None
    month_col = None
    day_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['date', 'تاریخ']:
            date_col = col
        elif col_lower in ['time', 'زمان']:
            time_col = col
        elif col_lower in ['year', 'سال']:
            year_col = col
        elif col_lower in ['month', 'ماه']:
            month_col = col
        elif col_lower in ['day', 'روز']:
            day_col = col
    
    # If we have date and time columns, parse them
    if date_col and time_col:
        print(f"  • Found date column: '{date_col}', time column: '{time_col}'")
        
        def parse_jalali_datetime(row):
            try:
                date_str = str(row[date_col]).strip()
                time_str = str(row[time_col]).strip()
                
                # Parse date (could be YYYY/MM/DD or YYYY-MM-DD)
                date_parts = date_str.replace('-', '/').split('/')
                if len(date_parts) == 3:
                    year = int(date_parts[0])
                    month = int(date_parts[1])
                    day = int(date_parts[2])
                else:
                    return pd.NaT
                
                # Parse time (could be HH:MM or HH:MM:SS)
                time_parts = time_str.split(':')
                hour = int(time_parts[0]) if len(time_parts) > 0 and time_parts[0].isdigit() else 0
                minute = int(time_parts[1]) if len(time_parts) > 1 and time_parts[1].isdigit() else 0
                second = int(time_parts[2]) if len(time_parts) > 2 and time_parts[2].isdigit() else 0
                
                # Convert Jalali to Gregorian
                j_date = JalaliDate(year, month, day)
                g_date = j_date.to_gregorian()
                
                return datetime(g_date.year, g_date.month, g_date.day, hour, minute, second)
            except Exception:
                return pd.NaT
        
        df['georgian_datetime'] = df.apply(parse_jalali_datetime, axis=1)
        df = df.dropna(subset=['georgian_datetime'])
        print(f"  ✓ Converted {len(df):,} rows")
        
    # If we have Year, Month, Day columns
    elif all([year_col, month_col, day_col]):
        print(f"  • Found Year: '{year_col}', Month: '{month_col}', Day: '{day_col}'")
        if time_col:
            print(f"  • Found Time: '{time_col}'")
        
        def parse_jalali_ymd(row):
            try:
                year = int(row[year_col]) if pd.notna(row[year_col]) else None
                month = int(row[month_col]) if pd.notna(row[month_col]) else None
                day = int(row[day_col]) if pd.notna(row[day_col]) else None
                
                if not all([year, month, day]):
                    return pd.NaT
                
                # Parse time if available
                hour = 0
                minute = 0
                second = 0
                if time_col and pd.notna(row[time_col]):
                    time_str = str(row[time_col]).strip()
                    time_parts = time_str.split(':')
                    hour = int(time_parts[0]) if len(time_parts) > 0 and time_parts[0].isdigit() else 0
                    minute = int(time_parts[1]) if len(time_parts) > 1 and time_parts[1].isdigit() else 0
                    second = int(time_parts[2]) if len(time_parts) > 2 and time_parts[2].isdigit() else 0
                
                # Convert Jalali to Gregorian
                j_date = JalaliDate(year, month, day)
                g_date = j_date.to_gregorian()
                
                return datetime(g_date.year, g_date.month, g_date.day, hour, minute, second)
            except Exception:
                return pd.NaT
        
        df['georgian_datetime'] = df.apply(parse_jalali_ymd, axis=1)
        df = df.dropna(subset=['georgian_datetime'])
        print(f"  ✓ Converted {len(df):,} rows")
    
    # If georgian_datetime already exists, just validate it
    elif 'georgian_datetime' in df.columns:
        print(f"  • georgian_datetime column already exists")
        df['georgian_datetime'] = pd.to_datetime(df['georgian_datetime'], errors='coerce')
        df = df.dropna(subset=['georgian_datetime'])
        print(f"  ✓ Validated {len(df):,} rows")
    
    # Try to auto-detect any date-like column
    else:
        print(f"  • Attempting auto-detection of date columns...")
        for col in df.columns:
            try:
                test_series = pd.to_datetime(df[col], errors='coerce')
                if test_series.notna().sum() > len(df) * 0.5:  # More than 50% valid dates
                    df['georgian_datetime'] = test_series
                    df = df.dropna(subset=['georgian_datetime'])
                    print(f"  ✓ Auto-detected '{col}' as datetime ({len(df):,} rows)")
                    break
            except Exception:
                continue
        else:
            raise ValueError(
                f"Could not find or convert date columns in {file_type} data. "
                f"Available columns: {', '.join(df.columns[:20])}"
            )
    
    # Sort by datetime
    df = df.sort_values('georgian_datetime').reset_index(drop=True)
    
    # Create Jalali string representation
    def datetime_to_jalali_string(dt):
        try:
            j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
            return j_dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None
    
    df['jalali_datetime_str'] = df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    # Deduplicate based on georgian_datetime
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['georgian_datetime'], keep='first')
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"  • Removed {removed:,} duplicate timestamps")
    
    return df


def process_process_tags_files(file_list: List[Any]) -> pd.DataFrame:
    """Load and process Process Tags files."""
    df = load_and_stack_csvs(file_list, "process")
    df = convert_jalali_to_georgian(df, "process")
    
    # Set index for merging
    df = df.set_index('georgian_datetime')
    
    # Remove metadata columns
    metadata_cols = ['_source_file', 'jalali_datetime_str', 'ID', 'DAY', 'MONTH', 'YEAR', 'Date', 'Time', 
                     'Year', 'Month', 'Day', 'date', 'time']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    # Add INST_ prefix to all columns
    df.columns = [f'INST_{col}' if not col.startswith('INST_') else col for col in df.columns]
    
    return df


def process_pellet_files(file_list: List[Any]) -> pd.DataFrame:
    """Load and process Pellet files."""
    df = load_and_stack_csvs(file_list, "pellet")
    df = convert_jalali_to_georgian(df, "pellet")
    
    # Set index for merging
    df = df.set_index('georgian_datetime')
    
    # Remove metadata columns
    metadata_cols = ['_source_file', 'jalali_datetime_str', 'Year', 'Month', 'Day', 'Time', 
                     'Source_File', 'date', 'time']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    # Add PELLET_ prefix to all columns
    df.columns = [f'PELLET_{col}' if not col.startswith('PELLET_') else col for col in df.columns]
    
    return df


def process_md_files(file_list: List[Any]) -> pd.DataFrame:
    """Load and process MD/Quality files."""
    df = load_and_stack_csvs(file_list, "md")
    df = convert_jalali_to_georgian(df, "md")
    
    # Set index for merging
    df = df.set_index('georgian_datetime')
    
    # Remove metadata columns
    metadata_cols = ['_source_file', 'jalali_datetime_str', 'time', 'source_file', 'date']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    # Add MDNC_ prefix to all columns
    df.columns = [f'MDNC_{col}' if not col.startswith('MDNC_') else col for col in df.columns]
    
    return df


def calculate_residence_times(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate enter and exit times based on WTH15."""
    df = df.copy()
    
    # Find WTH15 column
    wth15_col = None
    for col in df.columns:
        if 'WTH15' in col.upper():
            wth15_col = col
            break
    
    if not wth15_col:
        print("  ⚠ WTH15 column not found. Skipping residence time calculation.")
        return df
    
    # Calculate denominators
    denominator = (df[wth15_col] + 270) / 2
    denominator = denominator.where(denominator > 0)
    
    # Calculate residence times
    df['enter_deltatime_hours'] = 600 / denominator
    df['exit_deltatime_hours'] = 1200 / denominator
    
    df['enter_deltatime'] = pd.to_timedelta(df['enter_deltatime_hours'], unit='h', errors='coerce')
    df['exit_deltatime'] = pd.to_timedelta(df['exit_deltatime_hours'], unit='h', errors='coerce')
    
    df['enter_georgian_datetime'] = df.index - df['enter_deltatime']
    df['exit_georgian_datetime'] = df.index + df['exit_deltatime']
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data based on WTH15 threshold."""
    df = df.copy()
    
    wth15_col = None
    for col in df.columns:
        if 'WTH15' in col.upper():
            wth15_col = col
            break
    
    if wth15_col:
        df['status'] = df[wth15_col].apply(
            lambda x: 'Normal' if pd.notna(x) and x >= 80 else 'Shutdown'
        )
    else:
        df['status'] = 'Shutdown'
    
    return df


def merge_three_datasets(process_df: pd.DataFrame, pellet_df: pd.DataFrame, mdnc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge three datasets on georgian_datetime index.
    Uses merge_asof for time-based alignment.
    """
    print("\n" + "="*80)
    print("MERGING THREE DATASETS")
    print("="*80)
    
    # Determine time range
    all_dates = []
    if not process_df.empty:
        all_dates.extend([process_df.index.min(), process_df.index.max()])
    if not pellet_df.empty:
        all_dates.extend([pellet_df.index.min(), pellet_df.index.max()])
    if not mdnc_df.empty:
        all_dates.extend([mdnc_df.index.min(), mdnc_df.index.max()])
    
    if not all_dates:
        raise ValueError("No valid datetime data found in any dataset")
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    print(f"\n[Creating reference time grid]")
    print(f"  • Start: {min_date}")
    print(f"  • End: {max_date}")
    print(f"  • Frequency: 5T (5 minutes)")
    
    # Create reference time grid (5-minute intervals)
    time_index = pd.date_range(start=min_date, end=max_date, freq='5T')
    master_df = pd.DataFrame(index=time_index)
    master_df['georgian_datetime'] = time_index
    
    # Merge Process Tags
    print(f"\n[Merging Process Tags]")
    print(f"  • Process rows: {len(process_df):,}")
    for col in process_df.columns:
        master_df[col] = process_df[col]
    print(f"  ✓ Merged {len(process_df.columns)} columns")
    
    # Merge Pellet data (using merge_asof for time alignment)
    print(f"\n[Merging Pellet data]")
    print(f"  • Pellet rows: {len(pellet_df):,}")
    if not pellet_df.empty:
        for col in pellet_df.columns:
            # Use merge_asof for time-based matching
            merged = pd.merge_asof(
                master_df[['georgian_datetime']].sort_index(),
                pellet_df[[col]].sort_index(),
                left_on='georgian_datetime',
                right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('1H')  # 1 hour tolerance
            )
            master_df[col] = merged[col].values
    print(f"  ✓ Merged {len(pellet_df.columns)} columns")
    
    # Merge MD/Quality data (using merge_asof)
    print(f"\n[Merging MD/Quality data]")
    print(f"  • MD/Quality rows: {len(mdnc_df):,}")
    if not mdnc_df.empty:
        for col in mdnc_df.columns:
            merged = pd.merge_asof(
                master_df[['georgian_datetime']].sort_index(),
                mdnc_df[[col]].sort_index(),
                left_on='georgian_datetime',
                right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('1H')
            )
            master_df[col] = merged[col].values
    print(f"  ✓ Merged {len(mdnc_df.columns)} columns")
    
    # Add Jalali datetime string
    if 'jalali_datetime_str' in process_df.columns:
        master_df['jalali_datetime_str'] = process_df['jalali_datetime_str']
    else:
        def datetime_to_jalali_string(dt):
            try:
                j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
                return j_dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return None
        master_df['jalali_datetime_str'] = master_df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    # Calculate residence times
    print(f"\n[Calculating residence times]")
    master_df = calculate_residence_times(master_df)
    print(f"  ✓ Residence times calculated")
    
    # Validate data
    print(f"\n[Validating data]")
    master_df = validate_data(master_df)
    print(f"  ✓ Validation complete")
    
    # Reset index
    master_df = master_df.reset_index(drop=True)
    
    # Reorder columns
    datetime_cols = ['georgian_datetime', 'jalali_datetime_str']
    other_cols = [col for col in master_df.columns if col not in datetime_cols]
    master_df = master_df[datetime_cols + other_cols]
    
    print(f"\n[Final dataset]")
    print(f"  ✓ Shape: {master_df.shape}")
    print(f"  ✓ Total time points: {len(master_df):,}")
    
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
    """Load model from .pkl file."""
    import joblib
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def run_inference_for_md_c(model_md, model_c, df: pd.DataFrame) -> Dict[str, Any]:
    """Run inference to predict MD and C values."""
    predictions = {'MD': None, 'C': None}
    
    exclude_cols = ['georgian_datetime', 'jalali_datetime_str', 'status', 
                   'enter_deltatime_hours', 'exit_deltatime_hours',
                   'enter_deltatime', 'exit_deltatime',
                   'enter_georgian_datetime', 'exit_georgian_datetime']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return predictions
    
    X = df[numeric_cols].iloc[-1:].values
    
    if model_md is not None:
        try:
            if hasattr(model_md, 'predict'):
                pred_md = model_md.predict(X)[0]
                predictions['MD'] = float(pred_md)
        except Exception as e:
            print(f"Error predicting MD: {e}")
    
    if model_c is not None:
        try:
            if hasattr(model_c, 'predict'):
                pred_c = model_c.predict(X)[0]
                predictions['C'] = float(pred_c)
        except Exception as e:
            print(f"Error predicting C: {e}")
    
    return predictions


def process_data(
    process_files: List[Any],
    pellet_files: List[Any],
    md_files: List[Any],
    model_md_path: Optional[str] = None,
    model_c_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main processing function: load, stack, merge CSVs and optionally run inference.
    
    Parameters:
    -----------
    process_files : List
        List of Process Tags CSV files
    pellet_files : List
        List of Pellet CSV files
    md_files : List
        List of MD/Quality CSV files
    model_md_path : str, optional
        Path to MD model
    model_c_path : str, optional
        Path to C model
    
    Returns:
    --------
    dict with 'merged_df', 'stats', and optionally 'predictions'
    """
    # Validate inputs
    if not process_files:
        raise ValueError("No Process Tags files provided")
    if not pellet_files:
        raise ValueError("No Pellet files provided")
    if not md_files:
        raise ValueError("No MD/Quality files provided")
    
    # Load and process each dataset
    process_df = process_process_tags_files(process_files)
    pellet_df = process_pellet_files(pellet_files)
    mdnc_df = process_md_files(md_files)
    
    # Merge datasets
    merged_df = merge_three_datasets(process_df, pellet_df, mdnc_df)
    
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
        },
        'process_files': len(process_files),
        'pellet_files': len(pellet_files),
        'md_files': len(md_files)
    }
    
    return {
        'merged_df': merged_df,
        'stats': stats,
        'predictions': predictions
    }
