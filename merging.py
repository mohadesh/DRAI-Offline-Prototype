import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_master_dataset(
    instrumentation_file,
    mdnc_file,
    pellet_file,
    output_file,
    start_datetime='2024-03-20 00:00:00',
    end_datetime='2025-11-15 00:00:00',
    time_rate='5T',
    mdnc_tags_to_include=None,
    pellet_tags_to_include=None
):
    """
    Create a master dataset with a reference time grid and merge data from three sources.
    
    Parameters:
    -----------
    instrumentation_file : str
        Path to instrumentation_resample.csv
    mdnc_file : str
        Path to MDnC.csv
    pellet_file : str
        Path to Pellet.csv
    output_file : str
        Path for the output merged CSV file
    start_datetime : str
        Start datetime in format 'YYYY-MM-DD HH:MM:SS' (default: '2024-03-20 00:00:00')
    end_datetime : str
        End datetime in format 'YYYY-MM-DD HH:MM:SS' (default: '2025-11-15 00:00:00')
    time_rate : str
        Time frequency for the reference grid (default: '30T' for 30 minutes)
    mdnc_tags_to_include : list
        List of column names to include from MDnC.csv (default: all numeric columns)
    pellet_tags_to_include : list
        List of column names to include from Pellet.csv (default: all numeric columns)
    
    Returns:
    --------
    pandas.DataFrame
        Merged dataset with reference time grid
    """
    
    print("\n" + "="*80)
    print("MASTER DATASET CREATION PIPELINE")
    print("="*80)
    
    # =====================================================================
    # STEP 1: CREATE REFERENCE TIME GRID
    # =====================================================================
    
    print(f"\n[Step 1] Creating reference time grid...")
    print(f"  • Start: {start_datetime}")
    print(f"  • End: {end_datetime}")
    print(f"  • Frequency: {time_rate}")
    
    # Parse datetime strings
    start_dt = pd.to_datetime(start_datetime)
    end_dt = pd.to_datetime(end_datetime)
    
    # Create time range
    time_index = pd.date_range(start=start_dt, end=end_dt, freq=time_rate)
    
    # Create master dataframe
    master_df = pd.DataFrame(index=time_index)
    master_df['georgian_datetime'] = time_index
    
    print(f"  ✓ Created {len(master_df):,} time points")
    print(f"  ✓ Duration: {(end_dt - start_dt).days} days")
    
    # =====================================================================
    # STEP 2: LOAD AND MERGE INSTRUMENTATION DATA
    # =====================================================================
    
    print(f"\n[Step 2] Loading instrumentation data...")
    try:
        inst_df = pd.read_csv(instrumentation_file)
        drop_cols = [c for c in ['ID', 'DAY', 'MONTH', 'YEAR', 'Date', 'Time'] if c in inst_df.columns]
        if drop_cols:
            inst_df = inst_df.drop(drop_cols, axis=1)
        # Convert georgian_datetime to datetime if it's not already
        if 'georgian_datetime' in inst_df.columns:
            inst_df['georgian_datetime'] = pd.to_datetime(inst_df['georgian_datetime'])
            inst_df = inst_df.set_index('georgian_datetime')
        else:
            print("  ✗ Error: 'georgian_datetime' column not found in instrumentation file")
            return None
        
        print(f"  ✓ Loaded {len(inst_df):,} rows")
        print(f"  ✓ Date range: {inst_df.index.min()} to {inst_df.index.max()}")
        print(f"  ✓ Columns: {len(inst_df.columns)}")
        
        # Merge with master dataframe (using index)
        # Get all columns except jalali_datetime_str (we'll keep only one version)
        inst_cols = [col for col in inst_df.columns if col != 'jalali_datetime_str']
        
        for col in inst_cols:
            master_df[f'INST_{col}'] = inst_df[col]
        
        # Keep one jalali_datetime_str from instrumentation
        if 'jalali_datetime_str' in inst_df.columns:
            master_df['jalali_datetime_str'] = inst_df['jalali_datetime_str']
        
        print(f"  ✓ Merged {len(inst_cols)} columns with 'INST_' prefix")
        
    except Exception as e:
        print(f"  ✗ Error loading instrumentation file: {e}")
        return None
    
    # =====================================================================
    # STEP 3: LOAD AND MERGE MDnC DATA
    # =====================================================================
    
    print(f"\n[Step 3] Loading MDnC data...")
    try:
        mdnc_df = pd.read_csv(mdnc_file)
        
        # Convert georgian_datetime to datetime
        if 'georgian_datetime' in mdnc_df.columns:
            mdnc_df['georgian_datetime'] = pd.to_datetime(mdnc_df['georgian_datetime'])
        else:
            print("  ✗ Error: 'georgian_datetime' column not found in MDnC file")
            return None
        
        print(f"  ✓ Loaded {len(mdnc_df):,} rows")
        print(f"  ✓ Date range: {mdnc_df['georgian_datetime'].min()} to {mdnc_df['georgian_datetime'].max()}")
        
        # Determine which columns to include
        exclude_cols = ['jalali_datetime_str', 'time', 'source_file', 'georgian_datetime']
        
        if mdnc_tags_to_include is None:
            # Include all numeric columns except excluded ones
            mdnc_cols = [col for col in mdnc_df.columns 
                        if col not in exclude_cols]
        else:
            # Use specified tags
            mdnc_cols = [col for col in mdnc_tags_to_include if col in mdnc_df.columns]
        
        # Convert selected columns to numeric, coercing errors to NaN
        for col in mdnc_cols:
            mdnc_df[col] = pd.to_numeric(mdnc_df[col], errors='coerce')
        
        # Filter to only keep columns that are actually numeric after conversion
        mdnc_cols = [col for col in mdnc_cols if pd.api.types.is_numeric_dtype(mdnc_df[col])]
        
        print(f"  ✓ Selected columns: {mdnc_cols}")
        
        # Check for duplicates
        duplicates = mdnc_df['georgian_datetime'].duplicated().sum()
        if duplicates > 0:
            print(f"  • Found {duplicates} duplicate timestamps - aggregating by mean")
            
            # Aggregate duplicates by taking mean
            agg_dict = {col: 'mean' for col in mdnc_cols}
            mdnc_df = mdnc_df.groupby('georgian_datetime').agg(agg_dict).reset_index()
            print(f"  ✓ After aggregation: {len(mdnc_df):,} unique timestamps")
        
        # Set index after handling duplicates
        mdnc_df = mdnc_df.set_index('georgian_datetime')
        
        # Merge with master dataframe
        for col in mdnc_cols:
            master_df[f'MDNC_{col}'] = mdnc_df[col]
        
        print(f"  ✓ Merged {len(mdnc_cols)} columns with 'MDNC_' prefix")
        
    except Exception as e:
        print(f"  ✗ Error loading MDnC file: {e}")
        import traceback
        traceback.print_exc()
        # Continue without MDnC data
        print(f"  • Continuing without MDnC data...")
    
    # =====================================================================
    # STEP 4: LOAD AND MERGE PELLET DATA
    # =====================================================================
    
    print(f"\n[Step 4] Loading Pellet data...")
    try:
        pellet_df = pd.read_csv(pellet_file)
        
        # Convert georgian_datetime to datetime
        if 'georgian_datetime' in pellet_df.columns:
            pellet_df['georgian_datetime'] = pd.to_datetime(pellet_df['georgian_datetime'])
        else:
            print("  ✗ Error: 'georgian_datetime' column not found in Pellet file")
            return None
        
        print(f"  ✓ Loaded {len(pellet_df):,} rows")
        print(f"  ✓ Date range: {pellet_df['georgian_datetime'].min()} to {pellet_df['georgian_datetime'].max()}")
        
        # Determine which columns to include
        exclude_cols = ['jalali_datetime_str', 'Year', 'Month', 'Day', 'Time', 'Source_File', 'georgian_datetime']
        
        if pellet_tags_to_include is None:
            # Include all columns except excluded ones
            pellet_cols = [col for col in pellet_df.columns 
                          if col not in exclude_cols]
        else:
            # Use specified tags
            pellet_cols = [col for col in pellet_tags_to_include if col in pellet_df.columns]
        
        # Convert selected columns to numeric, coercing errors to NaN
        for col in pellet_cols:
            pellet_df[col] = pd.to_numeric(pellet_df[col], errors='coerce')
        
        # Filter to only keep columns that are actually numeric after conversion
        pellet_cols = [col for col in pellet_cols if pd.api.types.is_numeric_dtype(pellet_df[col])]
        
        print(f"  ✓ Selected columns: {pellet_cols}")
        
        # Check for duplicates
        duplicates = pellet_df['georgian_datetime'].duplicated().sum()
        if duplicates > 0:
            print(f"  • Found {duplicates} duplicate timestamps - aggregating by mean")
            
            # Aggregate duplicates by taking mean
            agg_dict = {col: 'mean' for col in pellet_cols}
            pellet_df = pellet_df.groupby('georgian_datetime').agg(agg_dict).reset_index()
            print(f"  ✓ After aggregation: {len(pellet_df):,} unique timestamps")
        
        # Set index after handling duplicates
        pellet_df = pellet_df.set_index('georgian_datetime')
        
        # Merge with master dataframe
        for col in pellet_cols:
            master_df[f'PELLET_{col}'] = pellet_df[col]
        
        print(f"  ✓ Merged {len(pellet_cols)} columns with 'PELLET_' prefix")
        
    except Exception as e:
        print(f"  ✗ Error loading Pellet file: {e}")
        import traceback
        traceback.print_exc()
        # Continue without Pellet data
        print(f"  • Continuing without Pellet data...")
    
    # =====================================================================
    # STEP 5: ORGANIZE AND SAVE
    # =====================================================================
    
    print(f"\n[Step 5] Organizing and saving dataset...")
    
    # Reset index to make georgian_datetime a column
    master_df = master_df.reset_index(drop=True)
    
    # Reorder columns: datetime columns first, then data columns
    datetime_cols = ['georgian_datetime']
    if 'jalali_datetime_str' in master_df.columns:
        datetime_cols.append('jalali_datetime_str')
    
    other_cols = [col for col in master_df.columns if col not in datetime_cols]
    master_df = master_df[datetime_cols + other_cols]
    
    # Calculate data availability
    total_rows = len(master_df)
    non_null_counts = master_df.notna().sum()
    
    print(f"  ✓ Final dataset shape: {master_df.shape}")
    print(f"  ✓ Total time points: {total_rows:,}")
    
    # Show data coverage statistics
    print(f"\n  Data Coverage:")
    inst_cols_in_master = [col for col in master_df.columns if col.startswith('INST_')]
    mdnc_cols_in_master = [col for col in master_df.columns if col.startswith('MDNC_')]
    pellet_cols_in_master = [col for col in master_df.columns if col.startswith('PELLET_')]
    
    if inst_cols_in_master:
        inst_coverage = (master_df[inst_cols_in_master].notna().any(axis=1).sum() / total_rows) * 100
        print(f"    • Instrumentation: {inst_coverage:.1f}% of time points have data")
    
    if mdnc_cols_in_master:
        mdnc_coverage = (master_df[mdnc_cols_in_master].notna().any(axis=1).sum() / total_rows) * 100
        print(f"    • MDnC: {mdnc_coverage:.1f}% of time points have data")
    
    if pellet_cols_in_master:
        pellet_coverage = (master_df[pellet_cols_in_master].notna().any(axis=1).sum() / total_rows) * 100
        print(f"    • Pellet: {pellet_coverage:.1f}% of time points have data")
    
    # Save to CSV
    master_df.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved to: {output_file}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"\nDataset Structure:")
    print(f"  • Reference time grid: {len(master_df):,} points at {time_rate} intervals")
    print(f"  • Date range: {start_datetime} to {end_datetime}")
    print(f"  • Total columns: {len(master_df.columns)}")
    print(f"    - Instrumentation: {len(inst_cols_in_master)} columns")
    print(f"    - MDnC: {len(mdnc_cols_in_master)} columns")
    print(f"    - Pellet: {len(pellet_cols_in_master)} columns")
    print(f"\nColumn Naming Convention:")
    print(f"  • INST_* : Instrumentation data")
    print(f"  • MDNC_* : MDnC data")
    print(f"  • PELLET_* : Pellet data")
    print("\n")
    
    return master_df


# =====================================================================
# USAGE EXAMPLE
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge processed data into master dataset.")
    parser.add_argument("--process", help="Path to processed ProcessTags CSV")
    parser.add_argument("--pellet", help="Path to processed Pellet CSV")
    parser.add_argument("--md", help="Path to processed MD/Quality CSV")
    parser.add_argument("--output", "-o", help="Output merged CSV path")
    parser.add_argument("--rate", default="5T", help="Time frequency (default: 5T)")
    args = parser.parse_args()

    if args.process and args.output:
        # Detect date range from process file
        try:
            df_proc = pd.read_csv(args.process, usecols=['georgian_datetime'], nrows=None)
            df_proc['georgian_datetime'] = pd.to_datetime(df_proc['georgian_datetime'])
            start_dt = str(df_proc['georgian_datetime'].min())
            end_dt = str(df_proc['georgian_datetime'].max())
        except Exception:
            start_dt = '2024-03-20 00:00:00'
            end_dt = '2025-11-15 00:00:00'

        mdnc_tags = ['M_D', 'C']
        pellet_tags = ['%FeO', 'CCS']

        master_df = create_master_dataset(
            instrumentation_file=args.process,
            mdnc_file=args.md or '',
            pellet_file=args.pellet or '',
            output_file=args.output,
            start_datetime=start_dt,
            end_datetime=end_dt,
            time_rate=args.rate,
            mdnc_tags_to_include=mdnc_tags,
            pellet_tags_to_include=pellet_tags
        )
    else:
        # Legacy: hardcoded example for local testing
        instrumentation_file = r'data\datasets\for_data_engineering\to_merge\Instruments_5T_Extended.csv'
        mdnc_file = r'data\datasets\for_data_engineering\to_merge\MDnC.csv'
        pellet_file = r'data\datasets\for_data_engineering\to_merge\Pellet.csv'
        output_file = r'data\datasets\for_preprocessing\master_dataset_5T_Extended.csv'

        mdnc_tags = ['M_D', 'C']
        pellet_tags = ['%FeO', 'CCS']

        print("\n" + "="*80)
        print("Include specific columns")
        print("="*80)

        master_df = create_master_dataset(
            instrumentation_file=instrumentation_file,
            mdnc_file=mdnc_file,
            pellet_file=pellet_file,
            output_file=output_file,
            start_datetime='2024-03-20 00:00:00',
            end_datetime='2025-11-15 00:00:00',
            time_rate='5T',
            mdnc_tags_to_include=mdnc_tags,
            pellet_tags_to_include=pellet_tags
        )