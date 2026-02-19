# ProcessTags.py — Process, clean, and calculate residence times for DRI data.
# Uses argparse, consistent column names (enter_deltatime_hours), pandas 'min' frequency, numeric-only resample.

import argparse
import csv
import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import jdatetime

warnings.filterwarnings("ignore")

# Attempt to import chardet for robust encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

def detect_file_properties(file_path):
    """
    Detects file encoding and delimiter. Uses chardet if available for better accuracy.
    Defaults to utf-8 and comma if detection fails.
    """
    encoding = "utf-8"
    delimiter = ","
    try:
        # Read a sample of the file for detection
        with open(file_path, "rb") as f:
            raw_data = f.read(50000) # Read 50KB

        # 1. Detect Encoding
        if HAS_CHARDET and raw_data:
            enc_result = chardet.detect(raw_data)
            encoding = enc_result.get("encoding") or "utf-8"
            print(f"  • Detected encoding: {encoding} (confidence: {enc_result.get('confidence', 0):.2f})", flush=True)

        # 2. Detect Delimiter
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            # Get a clean sample of lines without comments
            sample = ""
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    sample += line
                    if len(sample) > 4096: # 4KB sample is plenty
                        break
            if sample:
                try:
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    print(f"  • Detected delimiter: '{delimiter}'", flush=True)
                except csv.Error:
                    print("  • Sniffer could not determine delimiter, defaulting to ','", flush=True)

    except Exception as e:
        print(f"  [Warning] Auto-detection failed: {e}. Using defaults (utf-8, ',').", file=sys.stderr, flush=True)

    return encoding, delimiter


def parse_jalali_datetime(row):
    """
    Safely converts Jalali YEAR, MONTH, DAY, Time columns to a single Gregorian datetime object.
    Handles various time formats and potential errors gracefully.
    """
    try:
        j_date = jdatetime.date(int(row["YEAR"]), int(row["MONTH"]), int(row["DAY"]))
        g_date = j_date.togregorian()

        # Clean and parse time string
        t_str = str(row["Time"]).strip().replace(".", ":").replace(";", ":")
        parts = t_str.split(":")
        if len(parts) == 2:
            parts.append("00") # Add seconds if missing

        h = int(float(parts[0])) if parts else 0
        m = int(float(parts[1])) if len(parts) > 1 else 0
        s = int(float(parts[2])) if len(parts) > 2 else 0

        # Combine Gregorian date and parsed time
        return datetime.combine(g_date, datetime.strptime(f"{h:02d}:{m:02d}:{s:02d}", "%H:%M:%S").time())
    except (ValueError, TypeError, IndexError):
        return pd.NaT


def datetime_to_jalali_string(dt):
    """Converts a Gregorian datetime object back to a formatted Jalali string."""
    if pd.isna(dt):
        return None
    try:
        j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
        return j_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def normalize_resample_freq(freq):
    """
    Standardizes frequency string for pandas. Replaces 'T' with 'min' to avoid future warnings.
    e.g., '30T' becomes '30min'.
    """
    if not freq:
        return None
    # Ensure it's a string, clean it, and make replacement
    return str(freq).strip().upper().replace("T", "min")


def main():
    parser = argparse.ArgumentParser(
        description="Process DRI/Process Tags Data: Calculates residence time, cleans, and resamples."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV or Excel file.")
    parser.add_argument("--output", "-o", required=True, help="Path for the output cleaned CSV file.")
    parser.add_argument(
        "--resample-rate",
        default=None,
        help="Resampling frequency (e.g., '30min', '1H'). If not provided, no resampling is done."
    )
    args = parser.parse_args()

    print("\n" + "=" * 80, flush=True)
    print("DRI DATA PROCESSING PIPELINE (ProcessTags.py)", flush=True)
    print("=" * 80, flush=True)

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        # --- Step 1: Load Data ---
        print(f"\n[Step 1] Loading data from: {input_path.name}", flush=True)
        if input_path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(input_path, engine="openpyxl")
        else:
            encoding, delimiter = detect_file_properties(args.input)
            df = pd.read_csv(args.input, encoding=encoding, sep=delimiter, low_memory=False, on_bad_lines="warn")

        df.columns = df.columns.astype(str).str.strip()
        print(f"  ✓ Loaded {len(df):,} rows and {len(df.columns)} columns.", flush=True)

        # --- Step 2: Create Datetime from Jalali Columns ---
        print("\n[Step 2] Parsing Jalali datetime...", flush=True)
        # Find date/time columns, allowing for variations in naming
        col_map = {c.upper().strip(): c for c in df.columns}
        year_col = col_map.get("YEAR")
        month_col = col_map.get("MONTH")
        day_col = col_map.get("DAY")
        time_col = col_map.get("TIME")

        if not all([year_col, month_col, day_col, time_col]):
            raise ValueError(f"Missing required date/time columns (YEAR, MONTH, DAY, Time). Found: {list(df.columns)[:20]}")

        # Standardize column names for processing
        df = df.rename(columns={
            year_col: "YEAR", month_col: "MONTH", day_col: "DAY", time_col: "Time"
        })

        df["georgian_datetime"] = df.apply(parse_jalali_datetime, axis=1)
        initial_rows = len(df)
        df = df.dropna(subset=["georgian_datetime"])
        if len(df) < initial_rows:
            print(f"  • Removed {initial_rows - len(df):,} rows with invalid datetime values.", flush=True)
        if df.empty:
            raise ValueError("No valid data remaining after datetime parsing.")

        df["jalali_datetime_str"] = df["georgian_datetime"].apply(datetime_to_jalali_string)
        print(f"  ✓ Date range: {df['georgian_datetime'].min()} to {df['georgian_datetime'].max()}", flush=True)

        # --- Step 3: Calculate Residence Times ---
        print("\n[Step 3] Calculating residence times...", flush=True)
        wth_col = next((c for c in df.columns if str(c).strip().upper() == "WTH15"), None)
        if wth_col is None:
            raise ValueError(f"Column 'WTH15' not found. This column is required for residence time calculations. Available columns: {', '.join(df.columns[:30])}")

        denom = (pd.to_numeric(df[wth_col], errors='coerce').fillna(0) + 270) / 2
        df["enter_deltatime_hours"] = (600 / denom).replace([np.inf, -np.inf], np.nan)
        df["exit_deltatime_hours"] = (1200 / denom).replace([np.inf, -np.inf], np.nan)
        print(f"  ✓ Calculated 'enter_deltatime_hours' (avg: {df['enter_deltatime_hours'].mean():.2f}h)", flush=True)

        # --- Step 4: Resample Data (if requested) ---
        resample_freq = normalize_resample_freq(args.resample_rate)
        if resample_freq:
            print(f"\n[Step 4] Resampling data at '{resample_freq}' frequency...", flush=True)
            df_indexed = df.set_index("georgian_datetime").sort_index()

            # Resample only numeric columns to avoid errors
            numeric_cols = df_indexed.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found to resample.")

            df_res = df_indexed[numeric_cols].resample(resample_freq).mean()

            # After resampling, recalculate residence time for the new aggregated points
            if wth_col in df_res.columns:
                df_res = df_res.dropna(subset=[wth_col])
                denom_r = (pd.to_numeric(df_res[wth_col], errors='coerce') + 270) / 2
                df_res["enter_deltatime_hours"] = (600 / denom_r).replace([np.inf, -np.inf], np.nan)
                df_res["exit_deltatime_hours"] = (1200 / denom_r).replace([np.inf, -np.inf], np.nan)

            df_res["jalali_datetime_str"] = df_res.index.to_series().apply(datetime_to_jalali_string)
            df_final = df_res.reset_index()
            print(f"  ✓ Resampled data from {len(df):,} to {len(df_final):,} rows.", flush=True)
        else:
            print("\n[Step 4] No resample rate provided; skipping resampling step.", flush=True)
            df_final = df.copy()

        # --- Step 5: Save Output ---
        print("\n[Step 5] Saving final dataset...", flush=True)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Select and reorder columns for a clean output
        datetime_cols = ['georgian_datetime', 'jalali_datetime_str']
        data_cols = [c for c in df_final.columns if c not in datetime_cols]
        df_final = df_final[datetime_cols + sorted(data_cols)]

        df_final.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  ✓ Successfully saved cleaned data to: {out_path}", flush=True)
        print("=" * 80 + "\n", flush=True)

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred in ProcessTags.py: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
