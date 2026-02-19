import pandas as pd
from datetime import datetime
from persiantools.jdatetime import JalaliDate
import os
from pathlib import Path
import glob

def create_datetime_from_ymd(df):
    """
    Create jalali_datetime_str and georgian_datetime columns from Year, Month, Day, Time columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: Year, Month, Day (in Jalali), Time (in format 'HH:MM'), 
        and other custom columns
    
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with two new columns: georgian_datetime and jalali_datetime_str
        Format: 'yyyy-mm-dd hh:mm:ss'
    """
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Drop rows with missing values in date/time columns
    result_df = result_df.dropna(subset=['Year', 'Month', 'Day', 'Time'])
    
    # Reset index after dropping rows
    result_df = result_df.reset_index(drop=True)
    
    # Parse Year, Month, Day, Time to create datetime columns
    def create_datetimes(row):
        try:
            # Parse Jalali date
            year = int(row['Year'])
            month = int(row['Month'])
            day = int(row['Day'])
            
            # Parse time (handle both string 'HH:MM' and datetime objects)
            time_value = row['Time']
            
            if pd.isna(time_value):
                raise ValueError("Time is NaN")
            
            # If Time is a datetime object (Excel format)
            if isinstance(time_value, pd.Timestamp) or isinstance(time_value, datetime):
                hour = time_value.hour
                minute = time_value.minute
                second = time_value.second
            else:
                # If Time is a string 'HH:MM' or 'HH:MM:SS'
                time_str = str(time_value)
                time_parts = time_str.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
            
            # Create Jalali datetime string (format: YYYY-MM-DD HH:MM:SS)
            jalali_datetime_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            
            # Convert to Gregorian
            jalali_date = JalaliDate(year, month, day)
            gregorian_date = jalali_date.to_gregorian()
            
            # Create Georgian datetime object
            georgian_datetime = datetime(
                gregorian_date.year, 
                gregorian_date.month, 
                gregorian_date.day, 
                hour, 
                minute, 
                second
            )
            
            return jalali_datetime_str, georgian_datetime
            
        except Exception as e:
            print(f"Error converting row: {row}")
            raise e
    
    # Apply the function to create both datetime columns
    datetime_results = result_df.apply(create_datetimes, axis=1, result_type='expand')
    result_df['jalali_datetime_str'] = datetime_results[0]
    result_df['georgian_datetime'] = datetime_results[1]
    
    return result_df


def process_directory_excel_files(directory_path, file_pattern="*.xlsx", output_dir=None, sheet_name='Data', columns_to_keep=None):
    """
    Process all Excel files in a directory and add jalali_datetime_str and georgian_datetime columns.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing Excel files to process
    file_pattern : str, optional
        Pattern to match Excel files (default: "*.xlsx")
    output_dir : str, optional
        Directory to save processed files. If None, creates 'processed' subdirectory
    sheet_name : str, optional
        Name of the sheet to read from Excel files (default: 'Data')
    columns_to_keep : list, optional
        List of columns to keep. If None, uses default ['Year', 'Month', 'Day', 'Time', '%FeO', 'CCS', '%S']
    
    Returns:
    --------
    dict
        Dictionary with file paths as keys and processed DataFrames as values
    """
    if columns_to_keep is None:
        columns_to_keep = ['Year', 'Month', 'Day', 'Time', '%FeO', 'CCS', '%S']
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(directory_path, 'processed')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Excel files matching the pattern
    search_pattern = os.path.join(directory_path, file_pattern)
    excel_files = glob.glob(search_pattern)
    
    # Filter out temporary Excel files (starting with ~)
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~')]
    
    if not excel_files:
        print(f"No Excel files found in {directory_path} matching pattern '{file_pattern}'")
        return {}
    
    print(f"Found {len(excel_files)} Excel files to process:")
    for file in excel_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    processed_files = {}
    
    for file_path in excel_files:
        try:
            print(f"Processing: {os.path.basename(file_path)}")
            
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=1, engine='openpyxl')
            
            # Keep only specified columns (if they exist)
            existing_columns = [col for col in columns_to_keep if col in df.columns]
            if not existing_columns:
                print(f"  Warning: None of the expected columns found in {file_path}")
                continue
                
            df = df[existing_columns]
            
            # Process the dataframe
            processed_df = create_datetime_from_ymd(df)
            
            # Store in dictionary
            processed_files[file_path] = processed_df
            
            # Save processed file
            output_filename = Path(file_path).stem + '_processed.xlsx'
            output_path = os.path.join(output_dir, output_filename)
            
            processed_df.to_excel(output_path, index=False)
            print(f"  ✓ Saved to: {output_path}")
            print(f"  ✓ Rows processed: {len(processed_df)}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(file_path)}: {e}")
            print()
            continue
    
    return processed_files


def stack_processed_files(directory_path, output_filename='stacked_data.xlsx', sort_by='georgian_datetime'):
    """
    Stack all processed Excel files (with '_processed' in filename) into a single DataFrame.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing processed Excel files
    output_filename : str, optional
        Name of the output file to save stacked data. Default: 'stacked_data.xlsx'
    sort_by : str, optional
        Column to sort by. Default: 'georgian_datetime'
    
    Returns:
    --------
    pandas.DataFrame
        Stacked DataFrame containing all processed files
    """
    # Find all processed Excel files
    search_pattern = os.path.join(directory_path, '*_processed*.xlsx')
    processed_files = glob.glob(search_pattern)
    
    # Filter out temporary files
    processed_files = [f for f in processed_files if not os.path.basename(f).startswith('~')]
    
    if not processed_files:
        print(f"No processed files found in {directory_path}")
        return None
    
    print(f"Found {len(processed_files)} processed files:")
    for file in processed_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    # Read and stack all files
    dataframes = []
    for file_path in processed_files:
        try:
            df = pd.read_excel(file_path)
            # Add source file column to track origin
            df['Source_File'] = os.path.basename(file_path)
            dataframes.append(df)
            print(f"Loaded: {os.path.basename(file_path)} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not dataframes:
        print("No data to stack!")
        return None
    
    # Stack all dataframes
    stacked_df = pd.concat(dataframes, ignore_index=True)

    # Sort by georgian_datetime
    if sort_by in stacked_df.columns:
        if sort_by == 'georgian_datetime':
            # Sort by Georgian datetime object
            stacked_df = stacked_df.sort_values('georgian_datetime')
        elif sort_by == 'jalali_datetime_str':
            # For Jalali, we can sort by Georgian for reliable chronological sorting
            if 'georgian_datetime' in stacked_df.columns:
                stacked_df = stacked_df.sort_values('georgian_datetime')
            else:
                # Fallback: sort by jalali_datetime_str as string
                stacked_df = stacked_df.sort_values('jalali_datetime_str')
    
    # Reset index after sorting
    stacked_df = stacked_df.reset_index(drop=True)
    
    # Save stacked data
    # Determine file format and save accordingly
    if output_filename.endswith('.csv'):
        stacked_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    else:
        stacked_df.to_excel(output_filename, index=False)
    
    print(f"\n{'='*80}")
    print(f"Stacked {len(dataframes)} files into one DataFrame")
    print(f"Total rows: {len(stacked_df)}")
    print(f"Total columns: {len(stacked_df.columns)}")
    print(f"Sorted by: {sort_by}")
    # Show date range
    if 'georgian_datetime' in stacked_df.columns:
        dates = pd.to_datetime(stacked_df['georgian_datetime'])
        print(f"Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print(f"Saved to: {output_filename}")
    print(f"{'='*80}\n")
    
    return stacked_df


def create_complete_dataset(input_directory, output_file='complete_pellet_dataset.xlsx', 
                           sheet_name='Data', columns_to_keep=None, sort_by='georgian_datetime'):
    """
    Complete pipeline: Process all Excel files in directory and create a unified dataset.
    
    Parameters:
    -----------
    input_directory : str
        Path to directory containing raw Excel files
    output_file : str, optional
        Path for the final output file
    sheet_name : str, optional
        Sheet name to read from Excel files
    columns_to_keep : list, optional
        Columns to extract from each file
    sort_by : str, optional
        Column to sort by (default: 'georgian_datetime')
    
    Returns:
    --------
    pandas.DataFrame
        Complete stacked dataset
    """
    print("="*80)
    print("PELLET DATA PROCESSING PIPELINE")
    print("="*80)
    
    # Step 1: Process all Excel files in directory
    print(f"\n[Step 1] Processing Excel files in: {input_directory}")
    processed_data = process_directory_excel_files(
        directory_path=input_directory,
        sheet_name=sheet_name,
        columns_to_keep=columns_to_keep
    )
    
    if not processed_data:
        print("No files were processed successfully!")
        return None
    
    # Step 2: Stack all processed files with sorting
    print(f"\n[Step 2] Stacking and sorting processed files...")
    output_dir = os.path.join(input_directory, 'processed')
    stacked_df = stack_processed_files(output_dir, output_filename=output_file, sort_by=sort_by)
    
    if stacked_df is not None:
        # Step 3: Final summary
        print(f"\n[Step 3] Dataset Summary:")
        print(f"  • Total samples: {len(stacked_df):,}")
        print(f"  • Date range:")
        print(f"  • Sorted by: {sort_by}")
        
        # Verify sorting
        if 'georgian_datetime' in stacked_df.columns:
            dates = pd.to_datetime(stacked_df['georgian_datetime'])
            is_sorted = dates.is_monotonic_increasing
            print(f"  • Chronologically sorted: {'✓ YES' if is_sorted else '✗ NO'}")
        
        print(f"  • Columns: {list(stacked_df.columns)}")
        print(f"  • Source files: {stacked_df['Source_File'].nunique()}")
        
        print(f"\n✓ Complete dataset saved to: {output_file}")
    
    return stacked_df


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process Pellet Excel data.")
    parser.add_argument("--input", "-i", help="Input Excel file or directory path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--sheet", default="Data", help="Sheet name in Excel (default: Data)")
    args = parser.parse_args()

    if args.input and args.output:
        input_path = Path(args.input)
        if input_path.is_file():
            # Single file: process it directly
            print("="*80)
            print("PELLET DATA PROCESSING (Single File)")
            print("="*80)
            try:
                if input_path.suffix.lower() in ('.xlsx', '.xls'):
                    df = pd.read_excel(str(input_path), sheet_name=args.sheet, header=1, engine='openpyxl')
                else:
                    df = pd.read_csv(str(input_path))
                print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
                # Keep expected columns if they exist
                columns_to_extract = ['Year', 'Month', 'Day', 'Time', '%FeO', 'CCS', '%S']
                existing = [c for c in columns_to_extract if c in df.columns]
                if existing:
                    df = df[existing]
                df_processed = create_datetime_from_ymd(df)
                df_processed.to_csv(args.output, index=False, encoding='utf-8-sig')
                print(f"  ✓ Saved {len(df_processed)} rows to {args.output}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                # Write an empty CSV so downstream merging doesn't crash on missing file
                pd.DataFrame(columns=['georgian_datetime', 'jalali_datetime_str']).to_csv(args.output, index=False)
                print(f"  ✓ Wrote empty placeholder to {args.output}")
        elif input_path.is_dir():
            create_complete_dataset(
                input_directory=str(input_path),
                output_file=args.output,
                sheet_name=args.sheet,
                sort_by='georgian_datetime'
            )
        else:
            print(f"  ✗ Input path does not exist: {args.input}")
            pd.DataFrame(columns=['georgian_datetime', 'jalali_datetime_str']).to_csv(args.output, index=False)
    else:
        # Legacy: hardcoded example for local testing
        input_dir = "data/datasets/for_data_engineering/raw_from_source/27081404/pellet/PelletAnalysis/"
        output_file = "data/datasets/for_data_engineering/to_merge/Pellet.csv"
        columns_to_extract = ['Year', 'Month', 'Day', 'Time', '%FeO', 'CCS']
        create_complete_dataset(
            input_directory=input_dir,
            output_file=output_file,
            sheet_name='Data',
            columns_to_keep=columns_to_extract,
            sort_by='georgian_datetime'
        )