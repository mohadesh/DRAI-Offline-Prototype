# merging.py
import pandas as pd
import argparse
from pathlib import Path

def normalize_freq(freq_str):
    """
    Fix for Pandas >= 2.2.0 compatibility.
    Converts 'H' -> 'h' and 'T' -> 'min'
    """
    if not freq_str:
        return "30min"
    f = str(freq_str).strip()
    f = f.replace("H", "h").replace("T", "min")
    return f

def create_master_dataset(process_path, pellet_path, md_path, output_path, rate='30min'):
    """
    Creates a master dataset by merging three data sources onto a dynamic time grid.
    The time grid is determined by the min/max timestamps in the process_path file.
    """
    # Fix pandas frequency string before doing anything
    safe_rate = normalize_freq(rate)
    
    print("\n" + "="*80)
    print(f"MASTER DATASET CREATION | RATE: {safe_rate}")
    print("="*80)

    # --- STEP 1: LOAD MAIN FILE & DEFINE DYNAMIC TIME GRID ---
    print(f"\n[Step 1] Loading main process file to determine date range: {Path(process_path).name}")
    try:
        df_proc = pd.read_csv(process_path, usecols=['georgian_datetime'])
        df_proc['georgian_datetime'] = pd.to_datetime(df_proc['georgian_datetime'])
        start_dt = df_proc['georgian_datetime'].min()
        end_dt = df_proc['georgian_datetime'].max()

        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ValueError("Could not determine a valid date range from the process file.")

        print(f"  • Detected Start: {start_dt}")
        print(f"  • Detected End:   {end_dt}")

    except Exception as e:
        print(f"  ✗ FATAL: Could not read date range from process file. Error: {e}")
        raise

    print(f"\n[Step 2] Creating reference time grid at '{safe_rate}' frequency...")
    time_index = pd.date_range(start=start_dt, end=end_dt, freq=safe_rate)
    master_df = pd.DataFrame(index=time_index)
    master_df['georgian_datetime'] = time_index
    print(f"  ✓ Created {len(master_df):,} time points.")

    # --- STEP 3: LOAD AND MERGE ALL DATA SOURCES ---
    data_sources = {
        "INST": {"path": process_path, "df": None, "cols": []},
        "PELLET": {"path": pellet_path, "df": None, "cols": []},
        "MDNC": {"path": md_path, "df": None, "cols": []},
    }

    for prefix, source in data_sources.items():
        print(f"\n[Processing {prefix}] Loading data from: {Path(source['path']).name if source['path'] else 'N/A'}")
        if not source['path'] or not Path(source['path']).exists():
            print("  • Skipping: File path is empty or does not exist.")
            continue
        try:
            df = pd.read_csv(source['path'])
            if 'georgian_datetime' not in df.columns:
                print("  ✗ Error: 'georgian_datetime' column not found. Cannot merge.")
                continue

            df['georgian_datetime'] = pd.to_datetime(df['georgian_datetime'], errors='coerce')
            df.dropna(subset=['georgian_datetime'], inplace=True)

            # Aggregate duplicates by taking the mean for numeric columns
            if df['georgian_datetime'].duplicated().any():
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                df = df.groupby('georgian_datetime')[numeric_cols].mean().reset_index()
                print(f"  • Aggregated duplicate timestamps by mean.")

            df = df.set_index('georgian_datetime')
            source['df'] = df

            # Select columns to merge, excluding datetime-related ones
            cols_to_merge = [c for c in df.columns if 'date' not in c.lower() and 'time' not in c.lower()]
            source['cols'] = cols_to_merge
            print(f"  ✓ Loaded {len(df):,} rows and found {len(cols_to_merge)} data columns.")

            # Merge into master_df with prefix
            for col in cols_to_merge:
                master_df[f'{prefix}_{col}'] = source['df'][col]

            print(f"  ✓ Merged {len(cols_to_merge)} columns with '{prefix}_' prefix.")

        except Exception as e:
            print(f"  ✗ Error processing {prefix} file: {e}")

    # --- STEP 4: ORGANIZE AND SAVE ---
    print("\n[Step 4] Finalizing and saving the dataset...")

    # Organize Jalali Datetime if available
    if 'INST_jalali_datetime_str' in master_df.columns:
        master_df['jalali_datetime'] = master_df['INST_jalali_datetime_str']
        jalali_cols = [c for c in master_df.columns if 'jalali' in c.lower()]
        master_df = master_df.drop(columns=jalali_cols)
        cols = master_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('jalali_datetime')))
        master_df = master_df[cols]

    # ---------------------------------------------------------------------
    # CRITICAL FIX: Forward Fill (ffill) missing values
    # This ensures sparse data (like daily lab tests) are carried forward
    # so they don't show up as 0.00 on the UI dashboard.
    # ---------------------------------------------------------------------
    master_df.ffill(inplace=True)
    master_df.bfill(inplace=True) # Backfill just in case the first rows are NaN
    
    # Drop rows where all data columns are empty
    data_cols = [c for c in master_df.columns if c not in ['georgian_datetime', 'jalali_datetime']]
    initial_rows = len(master_df)
    master_df.dropna(subset=data_cols, how='all', inplace=True)
    print(f"  • Removed {initial_rows - len(master_df)} empty rows.")

    master_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved final dataset with shape {master_df.shape} to: {Path(output_path).name}")
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge processed data sources into a master dataset.")
    parser.add_argument("--process", required=True, help="Path to the cleaned ProcessTags CSV file.")
    parser.add_argument("--pellet", required=True, help="Path to the cleaned Pellet CSV file.")
    parser.add_argument("--md", required=True, help="Path to the cleaned MD/Quality CSV file.")
    parser.add_argument("--output", "-o", required=True, help="Path for the output merged CSV file.")
    parser.add_argument("--rate", default="30min", help="Time frequency for the master time grid (e.g., '30min', '1h').")
    args = parser.parse_args()

    create_master_dataset(
        process_path=args.process,
        pellet_path=args.pellet,
        md_path=args.md,
        output_path=args.output,
        rate=args.rate
    )
