import pandas as pd
import numpy as np
import jdatetime
import io
import gc
import joblib
import warnings
import re

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TARGET_FREQ = '5min'   
MIN_VALID_YEAR = 2020  # حذف داده‌های نویز (مثل سال 1990)

# ==============================================================================
# 1. HELPER FUNCTIONS: FILE READING & CLEANING
# ==============================================================================

def clean_column_names(df):
    """Standardize column names to avoid whitespace issues."""
    df.columns = [str(c).strip() for c in df.columns]
    return df

def optimize_floats(df):
    """Downcast float64 to float32 to save memory."""
    floats = df.select_dtypes(include=['float64']).columns
    df[floats] = df[floats].astype('float32')
    return df

def read_single_file_robust(file_storage):
    """
    Reads a file (Excel/CSV) robustly handling various encodings.
    """
    filename = ""
    file_bytes = None
    
    # Handle Flask FileStorage vs Local Path
    if hasattr(file_storage, 'filename'):
        filename = file_storage.filename.lower()
        file_bytes = file_storage.read()
        file_storage.seek(0)
    else:
        filename = str(file_storage).lower()
        with open(file_storage, 'rb') as f:
            file_bytes = f.read()

    df = None

    # --- A) HANDLE EXCEL ---
    if filename.endswith(('.xlsx', '.xls')):
        try:
            temp_io = io.BytesIO(file_bytes)
            # Find header row dynamically
            df_temp = pd.read_excel(temp_io, engine='openpyxl', header=None, nrows=20)
            header_row_idx = 0
            keywords = ['year', 'date', 'time', 'sample', 'tag_name', 'value', 'تاریخ', 'wth15', 'زمان']
            
            for i, row in df_temp.iterrows():
                row_str = row.astype(str).str.lower().values
                if sum(1 for k in keywords if any(k in str(s) for s in row_str)) >= 2:
                    header_row_idx = i
                    break
            
            temp_io.seek(0)
            df = pd.read_excel(temp_io, engine='openpyxl', header=header_row_idx)
        except Exception as e:
            print(f"    [ERROR] Excel read failed: {e}")

    # --- B) HANDLE CSV ---
    elif filename.endswith('.csv') or filename.endswith('.txt'):
        # Try 'utf-8-sig' first as seen in Armin's code
        encodings = ['utf-8-sig', 'utf-8', 'cp1256', 'latin1']
        delimiters = [',', ';', '\t']
        
        temp_io = io.BytesIO(file_bytes)
        
        for enc in encodings:
            if df is not None: break
            for sep in delimiters:
                try:
                    temp_io.seek(0)
                    df = pd.read_csv(temp_io, encoding=enc, sep=sep, on_bad_lines='skip', low_memory=False)
                    if len(df) > 1 and len(df.columns) > 1:
                        break 
                    else:
                        df = None
                except:
                    continue

    if df is not None and not df.empty:
        df = clean_column_names(df)
        df = optimize_floats(df)
        return df
    
    return None

def convert_jalali_to_gregorian(df):
    """
    Converts Jalali dates to Gregorian.
    Handles separated columns (Year, Month, Day, Time) or single string column.
    """
    if df is None or df.empty: return pd.DataFrame()
    
    df = df.copy()
    cols_map = {c.lower(): c for c in df.columns}
    
    def validate(dt):
        if pd.isna(dt): return pd.NaT
        if dt.year < MIN_VALID_YEAR: return pd.NaT
        return dt

    # Strategy 1: Separate Columns (Year, Month, Day, Time)
    # Common in Process Tags
    y_key = next((k for k in cols_map if k in ['year', 'سال']), None)
    m_key = next((k for k in cols_map if k in ['month', 'ماه']), None)
    d_key = next((k for k in cols_map if k in ['day', 'روز']), None)
    t_key = next((k for k in cols_map if k in ['time', 'زمان', 'ساعت']), None)

    if y_key and m_key and d_key:
        try:
            def make_dt(row):
                try:
                    y = int(row[cols_map[y_key]])
                    m = int(row[cols_map[m_key]])
                    d = int(row[cols_map[d_key]])
                    
                    # Fix 2-digit years
                    if y < 100: y += 1400 if y < 50 else 1300
                    
                    h, min_, s = 0, 0, 0
                    if t_key:
                        t_val = row[cols_map[t_key]]
                        if pd.notna(t_val):
                            if isinstance(t_val, str):
                                t_val = t_val.replace('.', ':').replace(';', ':')
                                parts = t_val.split(':')
                                if len(parts) >= 2:
                                    h, min_ = int(parts[0]), int(parts[1])
                            elif hasattr(t_val, 'hour'):
                                h, min_, s = t_val.hour, t_val.minute, t_val.second
                    
                    g_date = jdatetime.date(y, m, d).togregorian()
                    return validate(pd.Timestamp(year=g_date.year, month=g_date.month, day=g_date.day, 
                                                 hour=h, minute=min_, second=s))
                except:
                    return pd.NaT

            df['georgian_datetime'] = df.apply(make_dt, axis=1)
            df = df.dropna(subset=['georgian_datetime'])
            if not df.empty: return df
        except:
            pass

    # Strategy 2: Single String Column (YYYY/MM/DD)
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'تاریخ' in c.lower()), None)
    if date_col:
        try:
             def parse_jalali(val):
                 s = str(val).replace('/', '-').split(' ')[0]
                 match = re.search(r'(\d{4})[-](\d{1,2})[-](\d{1,2})', s)
                 if match:
                     y, m, d = map(int, match.groups())
                     try:
                         g = jdatetime.date(y, m, d).togregorian()
                         return validate(pd.Timestamp(g))
                     except: pass
                 return pd.NaT
             
             df['georgian_datetime'] = df[date_col].apply(parse_jalali)
             df = df.dropna(subset=['georgian_datetime'])
             if not df.empty: return df
        except:
            pass

    return df

def load_and_process_batch(files, prefix=""):
    """
    Loads multiple files, concatenates them, and handles date conversion.
    """
    df_list = []
    for f in files:
        df = read_single_file_robust(f)
        if df is not None:
            df = convert_jalali_to_gregorian(df)
            if 'georgian_datetime' in df.columns and not df.empty:
                # Add prefix to distinguish Pellet/MD columns
                if prefix:
                    rename_map = {c: f"{prefix}{c}" for c in df.columns if c != 'georgian_datetime'}
                    df = df.rename(columns=rename_map)
                
                df = df.set_index('georgian_datetime').sort_index()
                # Remove duplicate timestamps
                df = df[~df.index.duplicated(keep='first')]
                df_list.append(df)
        gc.collect()
        
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list).sort_index()

# ==============================================================================
# 2. MODEL INTERFACE
# ==============================================================================
# Track failed paths to avoid repeating the same error (e.g. missing 'darts')
_load_model_failed_paths = set()

def load_model(path):
    global _load_model_failed_paths
    try:
        if not path: return None
        obj = joblib.load(path)
        # If we had failed before for this path, it's now OK (e.g. after installing darts)
        _load_model_failed_paths.discard(path)
        return obj
    except Exception as e:
        if path not in _load_model_failed_paths:
            _load_model_failed_paths.add(path)
            print(f"Error loading model {path}: {e}")
            if "darts" in str(e).lower():
                print("  → Install the darts library: pip install darts  (required for pipeline .pkl models)")
        return None

def run_inference_for_md_c(model_md, model_c, df_row):
    predictions = {'MD': None, 'C': None}
    if df_row is None or df_row.empty:
        return predictions
    
    # Preprocessing for inference
    X = df_row.select_dtypes(include=[np.number]).fillna(0)
    
    if len(X.shape) == 1:
        X = X.values.reshape(1, -1)
    
    try:
        if model_md:
            pred = model_md.predict(X)
            predictions['MD'] = float(pred[0]) if len(pred) > 0 else None
            
        if model_c:
            pred = model_c.predict(X)
            predictions['C'] = float(pred[0]) if len(pred) > 0 else None
    except Exception as e:
        # Avoid crashing app on inference error
        pass
        
    return predictions

# ==============================================================================
# 3. CORE LOGIC: ARMIN'S SYNCHING IMPLEMENTATION
# ==============================================================================

def process_data(process_files, pellet_files, md_files, model_md_path=None, model_c_path=None):
    """
    Implements the exact logic from ProcessTags.py and synching.py:
    1. Master Grid = Process Data.
    2. Calc Residence Time (Enter Time) based on WTH15 formula.
    3. Find Pellet sample at (Process Time - Residence Time).
    4. Match MD sample at Process Time.
    """
    gc.collect()
    print("[Core] Starting Processing (Armin's Logic)...")

    # --- Step 1: Process Data (Master Grid) ---
    df_process = load_and_process_batch(process_files, prefix="")
    
    if df_process.empty:
         return {'merged_df': pd.DataFrame(), 'stats': {'rows':0}, 'predictions': None}

    # Create regular time grid (5min)
    min_date, max_date = df_process.index.min(), df_process.index.max()
    time_index = pd.date_range(start=min_date, end=max_date, freq=TARGET_FREQ, name='georgian_datetime')
    
    # Align process data to this grid
    master_df = df_process.reindex(time_index, method='nearest', tolerance=pd.Timedelta('10min'))
    master_df = master_df.reset_index()
    
    # Free memory
    del df_process
    gc.collect()

    # --- Step 2: Calculate Residence Time & Enter Time ---
    # Based on ProcessTags.py formula
    wth_col = next((c for c in master_df.columns if 'wth15' in c.lower()), None)
    
    if wth_col:
        # Formula: Residence Time (hr) = 600 / ((WTH15 + 270) / 2)
        # We handle NaNs by filling with a default (e.g., 80 for temp) to avoid division errors initially
        
        # [FIX] .fillna(method='ffill') is deprecated in pandas 2.1+
        # Replaced with .ffill()
        wth_values = master_df[wth_col].ffill().fillna(80) 
        
        den = (wth_values + 270) / 2
        res_hours = (600 / den).replace([np.inf, -np.inf], np.nan)
        
        # Clip reasonable bounds (e.g., between 2 and 24 hours) and fill missing
        res_hours = res_hours.clip(2, 24).fillna(8.0)
        
        master_df['calculated_residence_hours'] = res_hours.astype('float32')
        
        # Enter Time = Output Time (Current) - Residence Time
        # This tells us: "For the material coming out NOW, when did it go IN?"
        master_df['enter_georgian_datetime'] = master_df['georgian_datetime'] - pd.to_timedelta(master_df['calculated_residence_hours'], unit='h')
    else:
        # Fallback if WTH15 missing
        print("[Warning] WTH15 column not found. Using default 8h residence time.")
        master_df['calculated_residence_hours'] = 8.0
        master_df['enter_georgian_datetime'] = master_df['georgian_datetime'] - pd.to_timedelta(8, unit='h')

    # --- Step 3: Sync Pellet (Input) ---
    # Logic: Find the Pellet sample that existed at 'enter_georgian_datetime'
    df_pellet = load_and_process_batch(pellet_files, prefix="PELLET_")
    
    if not df_pellet.empty:
        df_pellet = df_pellet.reset_index().sort_values('georgian_datetime')
        
        # CRITICAL FIX FOR PANDAS 2.x & "Merge keys contain null":
        # We must drop rows where 'enter_georgian_datetime' is NaT before merging.
        # We'll do this on a temporary copy to preserve the master grid structure if possible, 
        # or just filter master_df if those rows are useless without pellet data.
        
        # Let's keep master_df intact but only merge on valid times
        # To do this efficiently with merge_asof, we need a clean key column.
        
        # Filter Master to only valid enter times for the merge process
        valid_merge_mask = master_df['enter_georgian_datetime'].notna()
        temp_master = master_df.loc[valid_merge_mask].copy()
        temp_master = temp_master.sort_values('enter_georgian_datetime')
        
        # Normalize datetime dtypes (merge_asof requires same type: ns vs us can differ from Excel/parsers)
        temp_master['enter_georgian_datetime'] = temp_master['enter_georgian_datetime'].astype('datetime64[ns]')
        df_pellet = df_pellet.copy()
        df_pellet['georgian_datetime'] = df_pellet['georgian_datetime'].astype('datetime64[ns]')
        
        # Perform Merge ASOF
        merged_pellet = pd.merge_asof(
            temp_master,
            df_pellet,
            left_on='enter_georgian_datetime',
            right_on='georgian_datetime',
            direction='nearest', 
            tolerance=pd.Timedelta('120min'), # Pellet sampling is every 2 hours roughly
            suffixes=('', '_pellet_idx')
        )
        
        # Now we need to put this data back into master_df.
        # Since temp_master was a subset, we can merge back on 'georgian_datetime' (the original index)
        # But simplify: If we just want the result, we can use the merged result as the new master
        # (assuming rows with invalid time are not needed for ML)
        master_df = merged_pellet.sort_values('georgian_datetime')
        
        # Cleanup extra columns
        cols_to_drop = ['georgian_datetime_pellet_idx']
        master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns])

    # --- Step 4: Sync MD/Quality (Output) ---
    # Logic: Match MD sample at Process Time (Output Side) - No Lag
    df_md = load_and_process_batch(md_files, prefix="MDNC_")
    
    if not df_md.empty:
        df_md = df_md.sort_index()
        
        # Ensure master is sorted by the key we are merging on
        master_df = master_df.sort_values('georgian_datetime')
        
        # Normalize datetime dtypes for merge_asof (ns vs us)
        master_df = master_df.copy()
        master_df['georgian_datetime'] = master_df['georgian_datetime'].astype('datetime64[ns]')
        df_md = df_md.copy()
        df_md.index = df_md.index.astype('datetime64[ns]')
        
        master_df = pd.merge_asof(
            master_df,
            df_md,
            left_on='georgian_datetime',
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('120min')
        )

    # --- Step 5: Final Polish ---
    # Add Jalali String for Dashboard display
    master_df['jalali_datetime_str'] = master_df['georgian_datetime'].apply(
        lambda x: jdatetime.datetime.fromgregorian(datetime=x).strftime('%Y/%m/%d %H:%M') if pd.notna(x) else ''
    )
    
    # Remove duplicates
    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    
    # Generate Stats
    stats = {
        'rows': len(master_df),
        'columns': len(master_df.columns),
        'start_date': str(master_df['georgian_datetime'].min()) if not master_df.empty else "-",
        'end_date': str(master_df['georgian_datetime'].max()) if not master_df.empty else "-",
        'process_files': len(process_files),
        'pellet_files': len(pellet_files),
        'md_files': len(md_files)
    }

    # Initial Prediction (Test run)
    predictions = None
    model_md = load_model(model_md_path)
    model_c = load_model(model_c_path)
    if (model_md or model_c) and not master_df.empty:
        try:
            predictions = run_inference_for_md_c(model_md, model_c, master_df.iloc[[-1]])
        except:
            pass

    gc.collect()
    
    return {
        'merged_df': master_df,
        'stats': stats,
        'predictions': predictions
    }
