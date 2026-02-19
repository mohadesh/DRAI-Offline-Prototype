import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import re
from persiantools.jdatetime import JalaliDate

def jalali_to_gregorian(j_year, j_month, j_day):
    """Convert Jalali date to Gregorian date using persiantools library"""
    try:
        jalali_date = JalaliDate(j_year, j_month, j_day)
        gregorian_date = jalali_date.to_gregorian()
        return gregorian_date.year, gregorian_date.month, gregorian_date.day
    except Exception as e:
        print(f"Error converting Jalali date {j_year}/{j_month}/{j_day}: {e}")
        # Return None to indicate failure
        return None, None, None


def parse_jalali_date_string(date_str):
    """
    Parse Jalali date string in format 'YYYY/MM/DD' or 'DD/MM/YYYY'
    Returns (year, month, day)
    """
    try:
        parts = date_str.strip().split('/')
        if len(parts) != 3:
            return None, None, None
        
        # Try YYYY/MM/DD format first (year > 1000)
        if int(parts[0]) > 1000:
            return int(parts[0]), int(parts[1]), int(parts[2])
        # Try DD/MM/YYYY format
        else:
            return int(parts[2]), int(parts[1]), int(parts[0])
    except:
        return None, None, None

def parse_time(time_str):
    """Parse time string and return hours, minutes, seconds"""
    if pd.isna(time_str):
        return 0, 0, 0
    
    time_str = str(time_str).strip()
    if ':' in time_str:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours, minutes, seconds
    else:
        # Try to parse as integer
        try:
            hours = int(time_str)
            return hours, 0, 0
        except:
            return 0, 0, 0

def adjust_date_for_time_crossing(df, base_gregorian_date, j_year, j_month, j_day):
    """
    Adjust dates when time crosses midnight.
    If time decreases from previous row, increment date by one day.
    Returns both Georgian dates and Jalali date strings.
    """
    gregorian_dates = []
    jalali_datetime_strs = []
    current_date = base_gregorian_date
    current_j_year, current_j_month, current_j_day = j_year, j_month, j_day
    
    for i, row in df.iterrows():
        time_parts = parse_time(row['time'])
        hours, minutes, seconds = time_parts
        
        if i == 0:
            # First row uses base date
            gregorian_dates.append(current_date)
            jalali_datetime_strs.append(
                f"{current_j_year:04d}-{current_j_month:02d}-{current_j_day:02d} {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
            continue
            
        current_time = time_parts
        prev_time = parse_time(df.iloc[i-1]['time'])
        
        current_hours = current_time[0]
        prev_hours = prev_time[0]
        
        # If current time is less than previous time, we've crossed midnight
        if current_hours < prev_hours:
            current_date += timedelta(days=1)
            
            # Increment Jalali date too
            current_j_day += 1
            # Simple day overflow handling for Jalali calendar
            if current_j_month <= 6 and current_j_day > 31:
                current_j_day = 1
                current_j_month += 1
            elif current_j_month > 6 and current_j_day > 30:
                current_j_day = 1
                current_j_month += 1
            
            if current_j_month > 12:
                current_j_month = 1
                current_j_year += 1
        
        gregorian_dates.append(current_date)
        jalali_datetime_strs.append(
            f"{current_j_year:04d}-{current_j_month:02d}-{current_j_day:02d} {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
    
    return gregorian_dates, jalali_datetime_strs

def aggregate_jalali_data(input_folder, additional_file=None, output_file='aggregated_data.csv'):
    """
    Aggregate multiple data files with Jalali dates into one CSV file.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing data files (Excel, CSV, etc.)
    additional_file : str, optional
        Path to additional CSV/Excel file with Date and Time columns (instead of filename-based dates)
    output_file : str
        Name of output CSV file (default: 'aggregated_data.csv')
    """
    
    all_data = []
    
    # Define column mapping from Persian to English
    column_mapping = {
        'زمان': 'time',
        '%Femetal': 'FeMetal',
        '%Fetotal': 'FeTotal',
        '%MD': 'M_D',
        '%C': 'C',
        '%S': 'S',
        '%Fine': 'Fine',
    }
    
    # Get all Excel and CSV files from the folder
    input_path = Path(input_folder)
    file_patterns = ['*.xlsx', '*.xls', '*.csv']
    files = []
    for pattern in file_patterns:
        files.extend(input_path.glob(pattern))
    
    print(f"Found {len(files)} files to process...")
    
    # =====================================================================
    # PROCESS ADDITIONAL FILE (if provided)
    # =====================================================================
    
    if additional_file:
        print(f"\n[Processing additional file: {Path(additional_file).name}]")
        try:
            # Determine file type and read
            file_ext = Path(additional_file).suffix.lower()
            if file_ext == '.csv':
                add_df = pd.read_csv(additional_file)
            else:
                add_df = pd.read_excel(additional_file)
            
            print(f"  ✓ Loaded {len(add_df)} rows")
            print(f"  • Columns: {list(add_df.columns)}")
            
            # Process rows with Date and Time columns
            if 'Date' in add_df.columns and 'Time' in add_df.columns:
                processed_rows = []
                
                for idx, row in add_df.iterrows():
                    try:
                        # Parse Date column
                        j_year, j_month, j_day = parse_jalali_date_string(str(row['Date']))
                        
                        if j_year is None:
                            continue
                        
                        # Convert to Gregorian
                        g_year, g_month, g_day = jalali_to_gregorian(j_year, j_month, j_day)
                        
                        if g_year is None:
                            continue
                        
                        # Parse time
                        time_parts = parse_time(row['Time'])
                        hours, minutes, seconds = time_parts
                        
                        # Create Georgian datetime
                        georgian_dt = datetime(g_year, g_month, g_day, hours, minutes, seconds)
                        
                        # Create Jalali datetime string
                        jalali_dt_str = f"{j_year:04d}-{j_month:02d}-{j_day:02d} {hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        # Create row dictionary
                        row_dict = {
                            'georgian_datetime': georgian_dt,
                            'jalali_datetime_str': jalali_dt_str,
                            'time': row['Time']
                        }
                        
                        # Add all numeric columns
                        for col in add_df.columns:
                            if col not in ['Date', 'Time']:
                                row_dict[col] = row[col]
                        
                        processed_rows.append(row_dict)
                        
                    except Exception as e:
                        print(f"  • Warning: Could not process row {idx}: {e}")
                        continue
                
                if processed_rows:
                    add_df_processed = pd.DataFrame(processed_rows)
                    add_df_processed['source_file'] = Path(additional_file).name
                    all_data.append(add_df_processed)
                    print(f"  ✓ Processed {len(add_df_processed)} rows from additional file")
                else:
                    print(f"  • Warning: No valid rows processed from additional file")
            else:
                print(f"  • Warning: 'Date' or 'Time' column not found in additional file")
                
        except Exception as e:
            print(f"  ✗ Error processing additional file: {e}")
    
    # =====================================================================
    # PROCESS FOLDER FILES
    # =====================================================================
    
    for file_path in files:
        try:
            # Read the file
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, header=2).iloc[:-2]
            else:
                df = pd.read_excel(file_path, header=2).iloc[:-2]
            
            # Extract date from filename (format: 1403.01.01)
            filename = file_path.stem
            date_pattern = r'(14\d{2})[.\-_/](\d{2})[.\-_/](\d{2})'
            match = re.search(date_pattern, filename)
            
            if match:
                j_year = int(match.group(1))
                j_month = int(match.group(2))
                j_day = int(match.group(3))
                
                # Convert to Gregorian for base date
                g_year, g_month, g_day = jalali_to_gregorian(j_year, j_month, j_day)
                
                # Check if conversion was successful
                if g_year is None:
                    print(f"Warning: Could not convert Jalali date {j_year}/{j_month}/{j_day} from {file_path.name}")
                    continue
                
                base_gregorian_date = datetime(g_year, g_month, g_day)
                
            else:
                print(f"Warning: Could not extract date from {file_path.name}")
                continue
            
            # Rename columns from Persian to English
            df = df.rename(columns=column_mapping)
            
            # Create datetime columns if time column exists
            if 'time' in df.columns and base_gregorian_date is not None:
                # Adjust dates for time crossing midnight
                gregorian_dates, jalali_datetime_strs = adjust_date_for_time_crossing(
                    df, base_gregorian_date, j_year, j_month, j_day
                )
                
                # Create Georgian datetime with proper date adjustment
                df['georgian_datetime'] = [
                    datetime(g_date.year, g_date.month, g_date.day, 
                            parse_time(row['time'])[0], 
                            parse_time(row['time'])[1],
                            parse_time(row['time'])[2])
                    if pd.notna(row['time']) 
                    else g_date
                    for g_date, (_, row) in zip(gregorian_dates, df.iterrows())
                ]
                
                # Add Jalali datetime string
                df['jalali_datetime_str'] = jalali_datetime_strs
                
            else:
                df['georgian_datetime'] = base_gregorian_date
                df['jalali_datetime_str'] = f"{j_year:04d}-{j_month:02d}-{j_day:02d} 00:00:00"
            
            df['source_file'] = file_path.name
            
            all_data.append(df)
            print(f"✓ Processed: {file_path.name}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")
            continue
    
    if not all_data:
        print("No data found to aggregate!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to have datetime first
    cols = combined_df.columns.tolist()
    priority_cols = ['georgian_datetime', 'jalali_datetime_str', 'time']
    other_cols = [col for col in cols if col not in priority_cols + ['source_file']]
    final_cols = priority_cols + other_cols + ['source_file']
    
    # Keep only columns that exist
    final_cols = [col for col in final_cols if col in cols]
    combined_df = combined_df[final_cols]
    
    # Sort by datetime
    if 'georgian_datetime' in combined_df.columns:
        combined_df = combined_df.sort_values('georgian_datetime')
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ Successfully created {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {', '.join(combined_df.columns)}")
    
    return combined_df


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process MD/Quality data files.")
    parser.add_argument("--input", "-i", help="Input Excel/CSV file or folder path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    args = parser.parse_args()

    if args.input and args.output:
        input_path = Path(args.input)
        if input_path.is_file():
            # Single file with Date and Time columns
            print(f"Processing single MD file: {input_path.name}")
            try:
                if input_path.suffix.lower() in ('.xlsx', '.xls'):
                    df = pd.read_excel(str(input_path))
                else:
                    df = pd.read_csv(str(input_path))
                print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

                column_mapping = {
                    'زمان': 'time', '%Femetal': 'FeMetal', '%Fetotal': 'FeTotal',
                    '%MD': 'M_D', '%C': 'C', '%S': 'S', '%Fine': 'Fine',
                }
                df = df.rename(columns=column_mapping)

                if 'Date' in df.columns and 'Time' in df.columns:
                    processed_rows = []
                    for idx, row in df.iterrows():
                        try:
                            j_year, j_month, j_day = parse_jalali_date_string(str(row['Date']))
                            if j_year is None:
                                continue
                            g_year, g_month, g_day = jalali_to_gregorian(j_year, j_month, j_day)
                            if g_year is None:
                                continue
                            hours, minutes, seconds = parse_time(row['Time'])
                            georgian_dt = datetime(g_year, g_month, g_day, hours, minutes, seconds)
                            jalali_dt_str = f"{j_year:04d}-{j_month:02d}-{j_day:02d} {hours:02d}:{minutes:02d}:{seconds:02d}"
                            row_dict = {'georgian_datetime': georgian_dt, 'jalali_datetime_str': jalali_dt_str, 'time': row.get('Time', '')}
                            for col in df.columns:
                                if col not in ('Date', 'Time', 'time'):
                                    row_dict[col] = row[col]
                            processed_rows.append(row_dict)
                        except Exception:
                            continue
                    if processed_rows:
                        result_df = pd.DataFrame(processed_rows)
                        result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
                        print(f"  ✓ Saved {len(result_df)} rows to {args.output}")
                    else:
                        print("  No valid rows found.")
                        pd.DataFrame(columns=['georgian_datetime', 'jalali_datetime_str', 'M_D', 'C']).to_csv(args.output, index=False)
                else:
                    print(f"  File columns: {list(df.columns)} — 'Date' and 'Time' not found, trying as folder file...")
                    aggregate_jalali_data(str(input_path.parent), output_file=args.output)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                pd.DataFrame(columns=['georgian_datetime', 'jalali_datetime_str', 'M_D', 'C']).to_csv(args.output, index=False)
        elif input_path.is_dir():
            aggregate_jalali_data(str(input_path), output_file=args.output)
        else:
            print(f"  ✗ Input path does not exist: {args.input}")
            pd.DataFrame(columns=['georgian_datetime', 'jalali_datetime_str', 'M_D', 'C']).to_csv(args.output, index=False)
    else:
        # Legacy: hardcoded example for local testing
        input_folder = "data/datasets/for_data_engineering/raw_from_source/27081404/MD/"
        output_file = "data/datasets/for_data_engineering/to_merge/MDnC.csv"
        additional_file = "data/datasets/for_data_engineering/raw_from_source/27081404/MD_additional.xlsx"
        df = aggregate_jalali_data(input_folder, additional_file=additional_file, output_file=output_file)