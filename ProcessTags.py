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

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


def detect_file_properties(file_path):
    """Detect encoding and delimiter. Uses chardet if available."""
    encoding = "utf-8"
    delimiter = ","
    try:
        with open(file_path, "rb") as f:
            raw = f.read(50000)
        if HAS_CHARDET:
            enc_result = chardet.detect(raw)
            encoding = enc_result.get("encoding") or "utf-8"
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            sample = ""
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    sample += line
                    if len(sample) > 2048:
                        break
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
            except Exception:
                pass
    except Exception as e:
        print(f"[Warning] Auto-detect failed: {e}. Using utf-8 and ','.", file=sys.stderr)
    return encoding, delimiter


def parse_jalali_datetime(row):
    """Convert Jalali YEAR, MONTH, DAY, Time to Gregorian datetime."""
    try:
        j_date = jdatetime.date(int(row["YEAR"]), int(row["MONTH"]), int(row["DAY"]))
        g_date = j_date.togregorian()
        t_str = str(row["Time"]).strip().replace(".", ":").replace(";", ":")
        parts = t_str.split(":")
        if len(parts) == 2:
            parts.append("00")
        h = int(float(parts[0])) if parts else 0
        m = int(float(parts[1])) if len(parts) > 1 else 0
        s = int(float(parts[2])) if len(parts) > 2 else 0
        return datetime.combine(g_date, datetime.strptime(f"{h:02d}:{m:02d}:{s:02d}", "%H:%M:%S").time())
    except Exception:
        return pd.NaT


def datetime_to_jalali_string(dt):
    """Gregorian datetime to Jalali string."""
    if pd.isna(dt):
        return None
    try:
        j_dt = jdatetime.datetime.fromgregorian(datetime=dt)
        return j_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def normalize_resample_freq(freq):
    """Replace 'T' with 'min' to avoid Pandas FutureWarning."""
    if not freq:
        return None
    s = str(freq).strip().upper()
    return s.replace("T", "min")


def main():
    parser = argparse.ArgumentParser(description="Process DRI/Process Tags: residence time, resample.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV or Excel path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--resample-rate", default=None, help="Resample frequency, e.g. 30min, 1H (use 'min' not 'T')")
    args = parser.parse_args()

    print("\n" + "=" * 80, flush=True)
    print("DRI DATA PROCESSING PIPELINE", flush=True)
    print("=" * 80, flush=True)

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        # --- Step 1: Load ---
        print(f"\n[Step 1] Loading: {args.input}", flush=True)
        if input_path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(input_path, engine="openpyxl")
        else:
            encoding, delimiter = detect_file_properties(args.input)
            df = pd.read_csv(args.input, encoding=encoding, sep=delimiter, low_memory=False, on_bad_lines="skip")
        df.columns = df.columns.astype(str).str.strip()
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.", flush=True)

        # --- Step 2: Datetime from Jalali ---
        print("\n[Step 2] Creating datetime columns...", flush=True)
        required = {"YEAR", "MONTH", "DAY", "Time"}
        col_upper = {c.upper(): c for c in df.columns}
        year_col = col_upper.get("YEAR") or next((c for c in df.columns if "year" in c.lower() or "سال" in c.lower()), None)
        month_col = col_upper.get("MONTH") or next((c for c in df.columns if "month" in c.lower() or "ماه" in c.lower()), None)
        day_col = col_upper.get("DAY") or next((c for c in df.columns if "day" in c.lower() or "روز" in c.lower()), None)
        time_col = next((c for c in df.columns if c == "Time" or "time" in c.lower() or "زمان" in c.lower()), None)
        if not all([year_col, month_col, day_col]):
            raise ValueError(f"Missing date columns. Need YEAR/MONTH/DAY (or Persian equivalents). Found: {list(df.columns)[:20]}")

        rename = {}
        if year_col != "YEAR":
            rename[year_col] = "YEAR"
        if month_col != "MONTH":
            rename[month_col] = "MONTH"
        if day_col != "DAY":
            rename[day_col] = "DAY"
        if time_col and time_col != "Time":
            rename[time_col] = "Time"
        if rename:
            df = df.rename(columns=rename)

        df["georgian_datetime"] = df.apply(parse_jalali_datetime, axis=1)
        before = len(df)
        df = df.dropna(subset=["georgian_datetime"])
        if len(df) < before:
            print(f"  Removed {before - len(df):,} rows with invalid datetime.", flush=True)
        if df.empty:
            raise ValueError("No valid rows after date parsing.")
        df["jalali_datetime_str"] = df["georgian_datetime"].apply(datetime_to_jalali_string)
        print(f"  Date range: {df['georgian_datetime'].min()} to {df['georgian_datetime'].max()}", flush=True)

        # --- Step 3: Residence time (always use enter_deltatime_hours / exit_deltatime_hours) ---
        print("\n[Step 3] Calculating residence times...", flush=True)
        wth_col = next((c for c in df.columns if str(c).strip().upper() == "WTH15"), None) or next(
            (c for c in df.columns if "wth15" in str(c).lower()), None
        )
        if wth_col is None:
            raise ValueError("Column WTH15 not found. Required for residence time. Columns: " + ", ".join(str(c) for c in df.columns[:30]))

        denom = (df[wth_col].astype(float).fillna(0) + 270) / 2
        df["enter_deltatime_hours"] = (600 / denom).replace([np.inf, -np.inf], np.nan)
        df["exit_deltatime_hours"] = (1200 / denom).replace([np.inf, -np.inf], np.nan)
        df["enter_deltatime"] = pd.to_timedelta(df["enter_deltatime_hours"], unit="h", errors="coerce")
        df["exit_deltatime"] = pd.to_timedelta(df["exit_deltatime_hours"], unit="h", errors="coerce")
        df["enter_georgian_datetime"] = df["georgian_datetime"] - df["enter_deltatime"]
        df["exit_georgian_datetime"] = df["georgian_datetime"] + df["exit_deltatime"]
        df["enter_jalali_datetime_str"] = df["enter_georgian_datetime"].apply(datetime_to_jalali_string)
        df["exit_jalali_datetime_str"] = df["exit_georgian_datetime"].apply(datetime_to_jalali_string)
        print(f"  enter_deltatime_hours mean: {df['enter_deltatime_hours'].mean():.2f}h", flush=True)

        # --- Step 4: Resample (only numeric columns, use 'min' not 'T') ---
        resample_freq = normalize_resample_freq(args.resample_rate)
        if resample_freq:
            print(f"\n[Step 4] Resampling at {resample_freq}...", flush=True)
            df_indexed = df.set_index("georgian_datetime")
            numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns to resample.")
            df_res = df_indexed[numeric_cols].resample(resample_freq).mean()
            df_res = df_res.dropna(subset=[c for c in [wth_col] if c in df_res.columns], how="all")
            if wth_col in df_res.columns:
                df_res = df_res.dropna(subset=[wth_col])
            denom_r = (df_res[wth_col].astype(float) + 270) / 2
            df_res["enter_deltatime_hours"] = (600 / denom_r).replace([np.inf, -np.inf], np.nan)
            df_res["exit_deltatime_hours"] = (1200 / denom_r).replace([np.inf, -np.inf], np.nan)
            df_res["enter_deltatime"] = pd.to_timedelta(df_res["enter_deltatime_hours"], unit="h", errors="coerce")
            df_res["exit_deltatime"] = pd.to_timedelta(df_res["exit_deltatime_hours"], unit="h", errors="coerce")
            df_res["enter_georgian_datetime"] = df_res.index - df_res["enter_deltatime"]
            df_res["exit_georgian_datetime"] = df_res.index + df_res["exit_deltatime"]
            df_res["jalali_datetime_str"] = df_res.index.to_series().apply(datetime_to_jalali_string)
            df_res["enter_jalali_datetime_str"] = df_res["enter_georgian_datetime"].apply(datetime_to_jalali_string)
            df_res["exit_jalali_datetime_str"] = df_res["exit_georgian_datetime"].apply(datetime_to_jalali_string)
            df_final = df_res.reset_index()
            print(f"  Resampled to {len(df_final):,} rows.", flush=True)
        else:
            print("\n[Step 4] No resample-rate; skipping resampling.", flush=True)
            df_final = df.copy()

        # --- Step 5: Save ---
        print("\n[Step 5] Saving output...", flush=True)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  Saved to {out_path}", flush=True)
        print("=" * 80 + "\n", flush=True)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
