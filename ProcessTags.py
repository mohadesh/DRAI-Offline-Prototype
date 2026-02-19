import pandas as pd
import numpy as np
import jdatetime
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

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


def filter_tags(df, tag_mode='exclude', tag_list=None):
    """
    Filter dataframe columns based on include/exclude tag list.
    
    Args:
        df: Input dataframe
        tag_mode: 'include' or 'exclude'
        tag_list: List of tag names to include or exclude
        
    Returns:
        Filtered dataframe
    """
    if tag_list is None:
        tag_list = []
    
    # Keep metadata columns
    metadata_cols = ['ID', 'DAY', 'MONTH', 'YEAR', 'Date', 'Time']
    
    if tag_mode == 'exclude':
        # Keep all columns except the ones in tag_list
        cols_to_keep = [col for col in df.columns if col in metadata_cols or col not in tag_list]
    elif tag_mode == 'include':
        # Keep only metadata columns and the ones in tag_list
        cols_to_keep = [col for col in df.columns if col in metadata_cols or col in tag_list]
    else:
        raise ValueError("tag_mode must be 'include' or 'exclude'")
    
    return df[cols_to_keep]


# ---------------------------------------------------------
# MAIN PROCESSING FUNCTION
# ---------------------------------------------------------

def process_dri_data(
    file_path,
    output_path=None,
    tag_mode='exclude',
    tag_list=None,
    enable_resampling=False,
    resample_rate='1H',
    delimiter=';'
):
    """
    Process DRI data with flexible tag filtering and optional resampling.
    
    Args:
        file_path: Path to input CSV file
        output_path: Path to save output CSV (optional)
        tag_mode: 'include' or 'exclude' - how to handle tag_list
        tag_list: List of tag names to include/exclude
        enable_resampling: Boolean - whether to resample data
        resample_rate: Resampling frequency (e.g., '1H', '30T', '15T')
        delimiter: CSV delimiter (default ';')
        
    Returns:
        pd.DataFrame: Processed dataframe with all calculated columns
    """
    
    print("\n" + "="*80)
    print("DRI DATA PROCESSING PIPELINE")
    print("="*80)
    
    # =====================================================================
    # STEP 1: LOAD AND FILTER DATA
    # =====================================================================
    
    print("\n[Step 1] Loading and filtering data...")
    file_path = str(file_path)
    file_lower = file_path.lower()
    if file_lower.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception:
            df = pd.read_excel(file_path)
    else:
        # Use the requested delimiter; if we get only 1 column, try common delimiters
        df = pd.read_csv(file_path, sep=delimiter, on_bad_lines='skip', low_memory=False)
        if len(df.columns) <= 1:
            for alt_sep in (',', '\t', ';'):
                if alt_sep == delimiter:
                    continue
                try:
                    df_alt = pd.read_csv(file_path, sep=alt_sep, on_bad_lines='skip', low_memory=False, nrows=1000)
                    if len(df_alt.columns) > len(df.columns):
                        df = pd.read_csv(file_path, sep=alt_sep, on_bad_lines='skip', low_memory=False)
                        print(f"  • Auto-detected delimiter: {repr(alt_sep)} ({len(df.columns)} columns)")
                        break
                except Exception:
                    continue
    # Normalize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"  • Loaded {len(df):,} rows")
    print(f"  • Original columns: {len(df.columns)}")
    
    # Apply tag filtering
    if tag_list:
        df = filter_tags(df, tag_mode=tag_mode, tag_list=tag_list)
        print(f"  ✓ Applied tag filter (mode: {tag_mode})")
        print(f"  ✓ Filtered columns: {len(df.columns)}")
        if tag_mode == 'exclude':
            print(f"  • Excluded tags: {tag_list}")
        else:
            print(f"  • Included tags: {tag_list}")
    
    # =====================================================================
    # STEP 2: CREATE DATETIME COLUMNS
    # =====================================================================
    
    print("\n[Step 2] Creating datetime columns...")
    def norm(s):
        return str(s).strip().upper()
    # Try common Jalali date column names (case-insensitive)
    year_col = next((c for c in df.columns if norm(c) in ('YEAR', 'سال')), None)
    month_col = next((c for c in df.columns if norm(c) in ('MONTH', 'ماه')), None)
    day_col = next((c for c in df.columns if norm(c) in ('DAY', 'روز')), None)
    time_col = next((c for c in df.columns if norm(c) in ('TIME', 'زمان', 'ساعت')), None)
    if year_col and month_col and day_col:
        def make_dt(row):
            try:
                y = int(row[year_col])
                m = int(row[month_col])
                d = int(row[day_col])
                if y < 100:
                    y += 1400 if y < 50 else 1300
                h, min_, s = 0, 0, 0
                if time_col and pd.notna(row.get(time_col)):
                    t_val = row[time_col]
                    if isinstance(t_val, str):
                        t_val = t_val.replace('.', ':').replace(';', ':')
                        parts = t_val.split(':')
                        if len(parts) >= 2:
                            h, min_ = int(float(parts[0])), int(float(parts[1]))
                    elif hasattr(t_val, 'hour'):
                        h, min_, s = t_val.hour, t_val.minute, t_val.second
                g_date = jdatetime.date(y, m, d).togregorian()
                return datetime.combine(g_date, datetime.strptime(f"{h:02d}:{min_:02d}:{s:02d}", "%H:%M:%S").time())
            except Exception:
                return pd.NaT
        df['georgian_datetime'] = df.apply(make_dt, axis=1)
    else:
        df['georgian_datetime'] = df.apply(parse_jalali_datetime, axis=1)
    
    # Remove rows with invalid datetime
    initial_rows = len(df)
    df = df.dropna(subset=['georgian_datetime'])
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"  • Removed {removed_rows:,} rows with invalid datetime")
    
    if df.empty:
        raise ValueError(
            "No valid rows after date parsing. Check that the file has date columns "
            "(e.g. YEAR, MONTH, DAY, Time or سال, ماه, روز, زمان) and the correct delimiter (e.g. ; or ,)."
        )
    
    # Create Jalali string representation
    df['jalali_datetime_str'] = df['georgian_datetime'].apply(datetime_to_jalali_string)
    
    print(f"  ✓ Created georgian_datetime column")
    print(f"  ✓ Created jalali_datetime_str column")
    print(f"  ✓ Date range: {df['georgian_datetime'].min()} to {df['georgian_datetime'].max()}")
    
    # =====================================================================
    # STEP 3: CALCULATE ENTER TIME
    # =====================================================================
    
    print("\n[Step 3] Calculating enter time...")
    
    # Find WTH15 column (case-insensitive, strip spaces)
    wth15_col = next((c for c in df.columns if str(c).strip().upper() == 'WTH15'), None)
    if wth15_col is None:
        wth15_col = next((c for c in df.columns if 'wth15' in str(c).lower()), None)
    if wth15_col is None:
        raise ValueError(
            "WTH15 column not found. This column is required for calculations. "
            "Available columns: " + ", ".join(str(c) for c in df.columns[:30])
            + ("..." if len(df.columns) > 30 else "")
        )
    
    # Calculate enter_deltatime in hours: 600/((WTH15+270)/2)
    df['enter_deltatime_hours'] = 600 / ((df[wth15_col].astype(float).fillna(0) + 270) / 2)
    
    # Convert to timedelta
    df['enter_deltatime'] = pd.to_timedelta(df['enter_deltatime_hours'], unit='h')
    
    # Calculate enter_georgian_datetime: georgian_datetime - enter_deltatime
    df['enter_georgian_datetime'] = df['georgian_datetime'] - df['enter_deltatime']
    
    # Convert to Jalali string format
    df['enter_jalali_datetime_str'] = df['enter_georgian_datetime'].apply(datetime_to_jalali_string)
    
    print(f"  ✓ Calculated enter_deltatime (mean: {df['enter_deltatime_hours'].mean():.2f} hours)")
    print(f"  ✓ Calculated enter_georgian_datetime and enter_jalali_datetime_str")
    
    # =====================================================================
    # STEP 4: CALCULATE EXIT TIME
    # =====================================================================
    
    print("\n[Step 4] Calculating exit time...")
    
    # Calculate exit_deltatime in hours: 1200/((WTH15+270)/2)
    df['exit_deltatime_hours'] = 1200 / ((df[wth15_col].astype(float).fillna(0) + 270) / 2)
    
    # Convert to timedelta
    df['exit_deltatime'] = pd.to_timedelta(df['exit_deltatime_hours'], unit='h')
    
    # Calculate exit_georgian_datetime: georgian_datetime + exit_deltatime
    df['exit_georgian_datetime'] = df['georgian_datetime'] + df['exit_deltatime']
    
    # Convert to Jalali string format
    df['exit_jalali_datetime_str'] = df['exit_georgian_datetime'].apply(datetime_to_jalali_string)
    
    print(f"  ✓ Calculated exit_deltatime (mean: {df['exit_deltatime_hours'].mean():.2f} hours)")
    print(f"  ✓ Calculated exit_georgian_datetime and exit_jalali_datetime_str")
    
    # =====================================================================
    # STEP 5: OPTIONAL RESAMPLING (ROBUST)
    # =====================================================================

    if enable_resampling:
        print(f"\n[Step 5] Resampling data at rate: {resample_rate}...")

        # -----------------------------------------------------------------
        # Set datetime index
        # -----------------------------------------------------------------
        df_resampled = df.set_index('georgian_datetime').copy()

        # -----------------------------------------------------------------
        # Build aggregation rules
        # -----------------------------------------------------------------
        agg_rules = {}

        for col in df_resampled.columns:
            if col in ['enter_georgian_datetime', 'exit_georgian_datetime']:
                agg_rules[col] = 'first'
            elif col in ['enter_deltatime', 'exit_deltatime']:
                agg_rules[col] = 'mean'
            elif col in [
                'jalali_datetime_str',
                'enter_jalali_datetime_str',
                'exit_jalali_datetime_str'
            ]:
                continue
            elif col in ['ID', 'DAY', 'MONTH', 'YEAR', 'Date', 'Time']:
                agg_rules[col] = 'first'
            elif pd.api.types.is_numeric_dtype(df_resampled[col]):
                agg_rules[col] = 'mean'

        # -----------------------------------------------------------------
        # Resample
        # -----------------------------------------------------------------
        df_resampled = df_resampled.resample(resample_rate).agg(agg_rules)

        # -----------------------------------------------------------------
        # Drop empty bins (no WTH15 → physically meaningless)
        # -----------------------------------------------------------------
        df_resampled = df_resampled.dropna(subset=[wth15_col])

        # -----------------------------------------------------------------
        # Recalculate residence times safely
        # -----------------------------------------------------------------
        denom = (df_resampled[wth15_col].astype(float) + 270) / 2
        denom = denom.where(denom > 0)  # protect physics

        df_resampled['enter_deltatime_hours'] = 600 / denom
        df_resampled['exit_deltatime_hours'] = 1200 / denom

        # Remove invalid numeric results explicitly
        df_resampled = df_resampled.replace([np.inf, -np.inf], np.nan)

        # -----------------------------------------------------------------
        # Convert to timedeltas
        # -----------------------------------------------------------------
        df_resampled['enter_deltatime'] = pd.to_timedelta(
            df_resampled['enter_deltatime_hours'], unit='h', errors='coerce'
        )
        df_resampled['exit_deltatime'] = pd.to_timedelta(
            df_resampled['exit_deltatime_hours'], unit='h', errors='coerce'
        )

        # -----------------------------------------------------------------
        # Recalculate datetime columns
        # -----------------------------------------------------------------
        df_resampled['enter_georgian_datetime'] = (
            df_resampled.index - df_resampled['enter_deltatime']
        )
        df_resampled['exit_georgian_datetime'] = (
            df_resampled.index + df_resampled['exit_deltatime']
        )

        # -----------------------------------------------------------------
        # Recalculate Jalali strings
        # -----------------------------------------------------------------
        df_resampled['jalali_datetime_str'] = (
            df_resampled.index.to_series().apply(datetime_to_jalali_string)
        )
        df_resampled['enter_jalali_datetime_str'] = (
            df_resampled['enter_georgian_datetime'].apply(datetime_to_jalali_string)
        )
        df_resampled['exit_jalali_datetime_str'] = (
            df_resampled['exit_georgian_datetime'].apply(datetime_to_jalali_string)
        )

        # -----------------------------------------------------------------
        # Restore datetime column
        # -----------------------------------------------------------------
        df_resampled = df_resampled.reset_index().rename(
            columns={'index': 'georgian_datetime'}
        )

        print(f"  ✓ Resampled from {len(df):,} to {len(df_resampled):,} rows")

        df_final = df_resampled

    else:
        print(f"\n[Step 5] Skipping resampling (disabled)")
        df_final = df

    
    # =====================================================================
    # STEP 6: ORGANIZE COLUMNS AND SAVE
    # =====================================================================
    
    print("\n[Step 6] Organizing output...")
    
    # Organize columns in logical order
    time_cols = [
        'georgian_datetime',
        'jalali_datetime_str',
        'enter_deltatime_hours',
        'enter_deltatime',
        'enter_georgian_datetime',
        'enter_jalali_datetime_str',
        'exit_deltatime_hours',
        'exit_deltatime',
        'exit_georgian_datetime',
        'exit_jalali_datetime_str'
    ]
    
    metadata_cols = ['ID', 'DAY', 'MONTH', 'YEAR', 'Date', 'Time']
    
    # Get process tag columns (everything else)
    process_cols = [col for col in df_final.columns 
                   if col not in time_cols and col not in metadata_cols]
    
    # Reorder columns
    final_cols = time_cols + process_cols + metadata_cols
    final_cols = [col for col in final_cols if col in df_final.columns]
    
    df_final = df_final[final_cols]
    
    print(f"  ✓ Final dataset shape: {df_final.shape}")
    print(f"  ✓ Time columns: {len([c for c in time_cols if c in df_final.columns])}")
    print(f"  ✓ Process tag columns: {len(process_cols)}")
    
    # Save if output path provided
    if output_path:
        df_final.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved to: {output_path}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutput columns:")
    print(f"  • georgian_datetime: Process timestamp (Gregorian)")
    print(f"  • jalali_datetime_str: Process timestamp (Jalali string)")
    print(f"  • enter_deltatime: Time delta to calculate entry time")
    print(f"  • enter_georgian_datetime: Calculated entry time (Gregorian)")
    print(f"  • enter_jalali_datetime_str: Calculated entry time (Jalali string)")
    print(f"  • exit_deltatime: Time delta to calculate exit time")
    print(f"  • exit_georgian_datetime: Calculated exit time (Gregorian)")
    print(f"  • exit_jalali_datetime_str: Calculated exit time (Jalali string)")
    print(f"  • Process tags: {len(process_cols)} columns")
    print(f"\nFormulas used:")
    print(f"  • Enter time: 600/((WTH15+270)/2) hours before process time")
    print(f"  • Exit time: 1200/((WTH15+270)/2) hours after process time")
    if enable_resampling:
        print(f"  • Resampling rate: {resample_rate}")
    print("\n")
    
    return df_final


# ---------------------------------------------------------
# CLI: when run as script (e.g. by core_logic with --input / --output)
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process DRI/Process Tags data.")
    parser.add_argument("--input", "-i", help="Input CSV/Excel file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--resample-rate", default="5T", help="Resample rate (e.g. 5T, 1H)")
    parser.add_argument("--no-resample", action="store_true", help="Disable resampling")
    args = parser.parse_args()

    if args.input:
        # Invoked by app or user with --input: use provided paths
        df = process_dri_data(
            file_path=args.input,
            output_path=args.output,
            enable_resampling=not args.no_resample,
            resample_rate=args.resample_rate,
        )
    else:
        # No --input: run example (for local testing only)
        print("\n" + "="*80)
        print("EXAMPLE: Include specific tags, with 5-Minute resampling")
        print("="*80)
        df = process_dri_data(
            file_path=r'data\datasets\for_data_engineering\raw_from_source\27081404\combined_data.csv',
            output_path=r'data\datasets\for_data_engineering\to_merge\Instruments_5T_Extended.csv',
            tag_mode='include',
            tag_list=["WTH15", "AITC10", "AITC09", "AITA331", "AITA311", "AITA321",
                       "AITA18", "FTA19", "FTA33", "FTA22A36",
                       "FTA82", "TTA341", "TTA351", "TTA19",
                       "TTA521", "TTA522", "TTA523", "TTA524", "TTA525",
                       "TTA526", "TTA527", "TTA528", "TTA529", "TTA5210",
                       "PTA44", "PTD11", "PTD15", "PDIA48", "PTA45",
                       "TTA251", "TTA842", "FTA201"],
            enable_resampling=True,
            resample_rate='5T'
        )
        print(df.head())