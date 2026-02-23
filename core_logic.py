# core_logic.py
import os
import sys
import uuid
import shutil
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Subprocess logging limits
MAX_LINES_LOGGED = 5000
LINES_AFTER_CAP = 500
LOG_STDERR_LINES = 200

_DEBUG_START_TIME = datetime.now().strftime("%Y%m%d_%H%M")

# Full list of 43 features expected by the model
PAST_COVARIATES_COLS = [
    "PELLET_CCS", "PELLET_%FeO", "INST_WTH15", "INST_AITC10", "INST_AITC09",
    "INST_AITA331", "INST_AITA321", "INST_AITA311", "INST_AITA18", "INST_FTA19",
    "INST_FTA33", "INST_FTA22A36", "INST_FTA82", "INST_TTA341", "INST_TTA351",
    "INST_TTA19", "INST_TTA521", "INST_TTA522", "INST_TTA523", "INST_TTA524",
    "INST_TTA525", "INST_TTA526", "INST_TTA527", "INST_TTA528", "INST_TTA529",
    "INST_TTA5210", "INST_PTA45", "INST_PTD11", "INST_PTD15", "INST_PDIA48",
    "INST_PTA44", "INST_TTA251", "INST_TTA842", "INST_FTA201", "INST_TEMP_UPPER",
    "INST_TEMP_MIDDLE", "INST_TEMP_LOWER", "INST_DELTA_T1", "INST_DELTA_T2",
    "INST_DELTA_T_DIFF", "INST_FLOW_VAR_6H", "INST_BPR_SLOPE", "INST_BUSTLE_TEMP_RATE"
]

# --- Feature Engineering ---
def _safe_mean(cols, df):
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(index=df.index, dtype=float)
    return df[existing].mean(axis=1)

def apply_feature_engineering(df):
    df = df.copy()
    middle = ['INST_TTA521', 'INST_TTA522', 'INST_TTA523', 'INST_TTA524']
    upper = ['INST_TTA525', 'INST_TTA526', 'INST_TTA527']
    lower = ['INST_TTA528', 'INST_TTA529', 'INST_TTA5210']
    
    df['INST_TEMP_UPPER'] = _safe_mean(upper, df)
    df['INST_TEMP_MIDDLE'] = _safe_mean(middle, df)
    df['INST_TEMP_LOWER'] = _safe_mean(lower, df)

    if 'INST_TEMP_MIDDLE' in df.columns and 'INST_TEMP_UPPER' in df.columns:
        df['INST_DELTA_T1'] = np.abs((df['INST_TEMP_MIDDLE'] - df['INST_TEMP_UPPER'])) / 2
    if 'INST_TEMP_MIDDLE' in df.columns and 'INST_TEMP_LOWER' in df.columns:
        df['INST_DELTA_T2'] = np.abs((df['INST_TEMP_MIDDLE'] - df['INST_TEMP_LOWER'])) / 2
    if 'INST_DELTA_T1' in df.columns and 'INST_DELTA_T2' in df.columns:
        df['INST_DELTA_T_DIFF'] = (df['INST_DELTA_T1'] - df['INST_DELTA_T2']).abs()

    if 'INST_FTA19' in df.columns:
        df['INST_FLOW_VAR_6H'] = df['INST_FTA19'].rolling(6, min_periods=1).std().fillna(0)

    if 'INST_PTA45' in df.columns:
        df['INST_BPR_SLOPE'] = df['INST_PTA45'].diff().fillna(0)
    if 'INST_TTA341' in df.columns:
        df['INST_BUSTLE_TEMP_RATE'] = df['INST_TTA341'].diff().fillna(0)
        
    for col in ('INST_BPR_SLOPE', 'INST_BUSTLE_TEMP_RATE'):
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

TARGET_MD_COL = "MDNC_M_D"
TARGET_C_COL = "MDNC_C"

def _get_actual_scaler(scaler_target, target_col_name):
    if scaler_target is None: return None
    if not isinstance(scaler_target, dict): return scaler_target
    if target_col_name in scaler_target: return scaler_target[target_col_name]
    
    is_md = target_col_name in [TARGET_MD_COL, "MDNC_M_D", "M_D", "MD"]
    is_c = target_col_name in [TARGET_C_COL, "MDNC_C", "C"]

    if is_md:
        for k in ["MDNC_M_D", "M_D", "MD"]:
            if k in scaler_target: return scaler_target[k]
    if is_c:
        for k in ["MDNC_C", "C"]:
            if k in scaler_target: return scaler_target[k]

    keys = list(scaler_target.keys())
    if len(keys) >= 2:
        if is_md: return scaler_target[keys[0]]
        if is_c: return scaler_target[keys[1]]
    return list(scaler_target.values())[0] if scaler_target else None

def _preprocess_target_like_training(series, target_name):
    s = series.copy().ffill().bfill()
    if s.isna().any():
        mean_val = s.mean()
        if pd.isna(mean_val):
            fallback = 92.0 if "M_D" in target_name else 2.0
            s = s.fillna(fallback)
        else:
            s = s.fillna(mean_val)
    return s
def _prepare_covariates(df, model_covs, scaler_inst, scaler_pellet, safe_freq, frequency_for_debug="30T"):
    from darts import TimeSeries

    # 1. Apply Feature Engineering
    df_eng = apply_feature_engineering(df).copy()

    # 2. Fill missing columns with NaN initially
    for col in model_covs:
        if col not in df_eng.columns:
            df_eng[col] = np.nan

    cov_df = df_eng[model_covs].copy()

    # 3. Forward-fill and Backward-fill covariates
    for c in cov_df.columns:
        cov_df[c] = cov_df[c].ffill().bfill()
    cov_df = cov_df.fillna(0.0)

    # 4. Define exactly which features MUST NOT be scaled
    UNSCALED_FEATS = [
        "INST_DELTA_T_DIFF",
        "INST_FLOW_VAR_6H",
        "INST_BPR_SLOPE",
        "INST_BUSTLE_TEMP_RATE"
    ]

    # --- 5. SMART SCALING LOGIC (CRITICAL FIX) ---
    def _apply_scaler_safely(data_df, default_cols, scaler, prefix=""):
        if scaler is None or not default_cols:
            return data_df

        # Step A: Try to extract the EXACT column order the scaler was trained on
        ordered_cols = default_cols
        try:
            if hasattr(scaler, '_transformer') and hasattr(scaler._transformer, 'feature_names_in_'):
                expected = list(scaler._transformer.feature_names_in_)
                # Ensure we only use it if it matches our expected features length
                if len(expected) == len(default_cols) and all(c in data_df.columns for c in expected):
                    ordered_cols = expected
                    logger.info(f"[{prefix}] Applied exact training feature order from scaler brain.")
        except Exception:
            pass

        # Step B: Ensure all columns in ordered_cols exist
        for c in ordered_cols:
            if c not in data_df.columns:
                data_df[c] = 0.0

        # Step C: Create TimeSeries in the STRICT ORDER required by the scaler
        try:
            ts = TimeSeries.from_dataframe(data_df[ordered_cols], freq=safe_freq)
            scaled_ts = scaler.transform(ts)
            
            # Step D: Map the transformed values perfectly back to the DataFrame
            vals = scaled_ts.values()
            if vals.ndim == 3:
                vals = vals[:, :, 0]
                
            data_df[ordered_cols] = vals
        except Exception as e:
            logger.error(f"[{prefix}] Scaling transformation failed: {e}")

        return data_df

    # 6. Extract raw natural columns (fallback order)
    natural_pellet_cols = [c for c in df_eng.columns if c.startswith('PELLET_') and c in model_covs]
    natural_inst_cols = [c for c in df_eng.columns if c.startswith('INST_') and c in model_covs and c not in UNSCALED_FEATS]

    # 7. Apply smart scaling
    cov_df = _apply_scaler_safely(cov_df, natural_inst_cols, scaler_inst, prefix="INST")
    cov_df = _apply_scaler_safely(cov_df, natural_pellet_cols, scaler_pellet, prefix="PELLET")

    # 8. Safe Auditing
    try:
        save_debug_data(cov_df, prefix=f"inference_input_{frequency_for_debug}")
    except Exception:
        pass

    # 9. Reconstruct final covariate series in the model's required order
    covariate_series = TimeSeries.from_dataframe(cov_df[model_covs], freq=safe_freq)

    return covariate_series, df_eng


def _get_scaled_target_series(df, target_name, is_md, scaler_target, safe_freq):
    try: from darts import TimeSeries
    except ImportError: return None

    actual_scaler = _get_actual_scaler(scaler_target, target_name)
    df_clean = df.copy()

    if target_name in df_clean.columns:
        df_clean[target_name] = _preprocess_target_like_training(df_clean[target_name], target_name)
    else:
        df_clean[target_name] = 92.0 if is_md else 2.0

    try:
        ts_1d = TimeSeries.from_dataframe(df_clean[[target_name]], freq=safe_freq)
        if actual_scaler: return actual_scaler.transform(ts_1d)
        return ts_1d
    except Exception:
        pass

    try:
        ts_df = pd.DataFrame(index=df_clean.index)
        ts_df[TARGET_MD_COL] = _preprocess_target_like_training(df_clean.get(TARGET_MD_COL, pd.Series(92.0, index=df_clean.index)), TARGET_MD_COL)
        ts_df[TARGET_C_COL] = _preprocess_target_like_training(df_clean.get(TARGET_C_COL, pd.Series(2.0, index=df_clean.index)), TARGET_C_COL)
        ts_combined = TimeSeries.from_dataframe(ts_df, freq=safe_freq)

        if actual_scaler:
            scaled_ts_combined = actual_scaler.transform(ts_combined)
            return scaled_ts_combined[target_name]
    except Exception:
        pass

    return TimeSeries.from_dataframe(df_clean[[target_name]], freq=safe_freq)

def _inverse_transform_single(pred_ts, scaler_target, is_md, safe_freq):
    col_name = TARGET_MD_COL if is_md else TARGET_C_COL
    pred_val = float(pred_ts.values()[-1][0])
    actual_scaler = _get_actual_scaler(scaler_target, col_name)
    if actual_scaler is None: return pred_val
    try:
        inv = actual_scaler.inverse_transform(pred_ts)
        return float(inv.values()[-1][0])
    except: pass
    try:
        from darts import TimeSeries
        dummy_df = pd.DataFrame(0.0, index=pred_ts.time_index, columns=[TARGET_MD_COL, TARGET_C_COL])
        dummy_df[col_name] = pred_val
        dummy_ts = TimeSeries.from_dataframe(dummy_df, freq=safe_freq)
        inv = actual_scaler.inverse_transform(dummy_ts)
        col_idx = 0 if is_md else 1
        return float(inv.values()[-1][col_idx])
    except: pass
    try:
        arr = np.array([[pred_val]])
        inv = actual_scaler.inverse_transform(arr)
        return float(inv[0][0])
    except: pass
    try:
        arr = np.zeros((1, 2))
        arr[0, 0 if is_md else 1] = pred_val
        inv = actual_scaler.inverse_transform(arr)
        return float(inv[0][0 if is_md else 1])
    except: return pred_val

def _inverse_transform_horizon(pred_ts, scaler_target, is_md, horizon, safe_freq):
    try: from darts import TimeSeries
    except ImportError: return None
    if pred_ts is None or len(pred_ts) != horizon: return None
    col_name = TARGET_MD_COL if is_md else TARGET_C_COL
    actual_scaler = _get_actual_scaler(scaler_target, col_name)
    if actual_scaler is None: return [_to_python_float(pred_ts.values()[i][0]) for i in range(horizon)]

    try:
        inv = actual_scaler.inverse_transform(pred_ts)
        vals = inv.values()
        if vals is not None and len(vals) >= horizon:
            return [_to_python_float(vals[i][0]) for i in range(horizon)]
    except: pass
    try:
        dummy_df = pd.DataFrame(0.0, index=pred_ts.time_index, columns=[TARGET_MD_COL, TARGET_C_COL])
        dummy_df[col_name] = pred_ts.values().flatten()
        dummy_ts = TimeSeries.from_dataframe(dummy_df, freq=safe_freq)
        inv = actual_scaler.inverse_transform(dummy_ts)
        col_idx = 0 if is_md else 1
        vals = inv.values()
        if vals is not None and len(vals) >= horizon:
            return [_to_python_float(vals[i][col_idx]) for i in range(horizon)]
    except: pass
    return [_to_python_float(pred_ts.values()[i][0]) for i in range(horizon)]

def _to_python_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) if hasattr(np, "isnan") else (f != f)) or np.isinf(f) else round(f, 3)
    except: return None

def save_debug_data(data, prefix="model_input"):
    is_enabled = os.getenv("SAVE_FINAL_MODEL_INPUT", "False").lower() in ("true", "1", "t")
    if not is_enabled: return
    try:
        folder_name = os.getenv("DEBUG_DATA_FOLDER", "debug_model_inputs")
        debug_dir = BASE_DIR / folder_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        df_to_save = data.pd_dataframe() if hasattr(data, 'pd_dataframe') else pd.DataFrame(data)
        accumulated_file = debug_dir / f"{prefix}_accumulated_{_DEBUG_START_TIME}.csv"
        last_row = df_to_save.iloc[[-1]]
        last_row.to_csv(accumulated_file, mode='a' if accumulated_file.exists() else 'w', header=not accumulated_file.exists(), index=True)
        df_to_save.to_csv(debug_dir / f"{prefix}_latest_window.csv", mode='w', header=True, index=True)
    except: pass

def run_script(script_name, args):
    script_path = BASE_DIR / script_name
    if not script_path.exists(): raise FileNotFoundError(f"Script not found: {script_path}")
    command = [sys.executable, str(script_path)] + [str(a) for a in args]
    logger.info("Running %s", script_name)
    stderr_lines = []
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, universal_newlines=True, cwd=str(BASE_DIR)
        )
        lines_logged = [0]
        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ""):
                    if line and lines_logged[0] < MAX_LINES_LOGGED:
                        print(f"[{script_name}] " + line.rstrip(), flush=True)
                        lines_logged[0] += 1
            except: pass
        def read_stderr():
            try:
                for line in iter(process.stderr.readline, ""):
                    if line: stderr_lines.append(line.rstrip())
            except: pass
        out_thread, err_thread = threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)
        out_thread.start(); err_thread.start()
        returncode = process.wait()
        out_thread.join(timeout=5.0); err_thread.join(timeout=2.0)
        if returncode != 0:
            tail = "\n".join(stderr_lines[-LOG_STDERR_LINES:]) if stderr_lines else ""
            raise RuntimeError(f"{script_name} failed. Stderr:\n{tail}")
    except Exception as e:
        logger.exception("Error running %s", script_name)
        raise e

def save_uploaded_files(files, folder):
    saved = []
    for f in files:
        if f and getattr(f, "filename", None):
            path = folder / f.filename
            f.save(str(path))
            saved.append(str(path))
    return saved

def process_data(process_files, pellet_files, md_files, resample_rate="30T", model_md_path=None, model_c_path=None):
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    logger.info("Starting session: %s with frequency: %s", session_id, resample_rate)
    try:
        process_paths = save_uploaded_files(process_files, session_dir)
        pellet_paths = save_uploaded_files(pellet_files, session_dir)
        md_paths = save_uploaded_files(md_files, session_dir)
        
        output_process = session_dir / "Process_Cleaned.csv"
        output_pellet = session_dir / "Pellet_Cleaned.csv"
        output_md = session_dir / "MD_Cleaned.csv"
        output_merged = session_dir / "Merged_Final.csv"

        run_script("ProcessTags.py", ["--input", process_paths[0], "--output", str(output_process), "--resample-rate", resample_rate])
        run_script("Pellet.py", ["--input", pellet_paths[0], "--output", str(output_pellet)])
        run_script("MDnC.py", ["--input", md_paths[0], "--output", str(output_md)])
        run_script("merging.py", ["--process", str(output_process), "--pellet", str(output_pellet), "--md", str(output_md), "--output", str(output_merged), "--rate", resample_rate])

        final_df = pd.read_csv(output_merged, parse_dates=['georgian_datetime'])
        return {"success": True, "merged_df": final_df, "stats": {"rows": len(final_df), "columns": list(final_df.columns)}}
    except Exception as e:
        logger.exception("Pipeline failed")
        raise e

def run_inference_for_md_c(model_md, model_c, df_window, frequency="30T"):
    import joblib
    try: from darts import TimeSeries
    except ImportError: return {"MD": None, "C": None}

    if df_window is None or df_window.empty or len(df_window) < 24: return {"MD": None, "C": None}
    df = df_window.copy()
    if "georgian_datetime" in df.columns:
        df["georgian_datetime"] = pd.to_datetime(df["georgian_datetime"], errors="coerce")
        df = df.set_index("georgian_datetime")

    out = {"MD": None, "C": None}
    safe_freq = frequency.replace("T", "min").replace("H", "h")
    scalers_dir = BASE_DIR / "scalers"
    try:
        scaler_inst = joblib.load(scalers_dir / f"scaler_inst_{frequency}.pkl")
        scaler_pellet = joblib.load(scalers_dir / f"scaler_pellet_{frequency}.pkl")
        scaler_target = joblib.load(scalers_dir / f"scaler_target_{frequency}.pkl")
    except Exception as e:
        logger.error("Failed to load scalers: %s", e)
        return {"MD": None, "C": None}

    active_model = model_md if model_md is not None else model_c
    model_covs = list(active_model.past_covariate_components) if (active_model and getattr(active_model, "past_covariate_components", None)) else PAST_COVARIATES_COLS
    covariate_series, df_eng = _prepare_covariates(df, model_covs, scaler_inst, scaler_pellet, safe_freq, frequency_for_debug=frequency)
    
    if covariate_series is None: return {"MD": None, "C": None}

    if model_md is not None:
        try:
            scaled_target_md = _get_scaled_target_series(df_eng, TARGET_MD_COL, True, scaler_target, safe_freq)
            if scaled_target_md is not None:
                pred_md_scaled = model_md.predict(n=1, series=scaled_target_md, past_covariates=covariate_series)
                real_md = _inverse_transform_single(pred_md_scaled, scaler_target, True, safe_freq)
                if real_md is not None: out["MD"] = round(real_md, 2)
        except Exception as e: logger.error("MD prediction failed: %s", e)

    if model_c is not None:
        try:
            scaled_target_c = _get_scaled_target_series(df_eng, TARGET_C_COL, False, scaler_target, safe_freq)
            if scaled_target_c is not None:
                pred_c_scaled = model_c.predict(n=1, series=scaled_target_c, past_covariates=covariate_series)
                real_c = _inverse_transform_single(pred_c_scaled, scaler_target, False, safe_freq)
                if real_c is not None: out["C"] = round(real_c, 2)
        except Exception as e: logger.error("C prediction failed: %s", e)

    return out

def run_inference_horizon_8(model_md, model_c, df_window, frequency="30T"):
    import joblib
    try: from darts import TimeSeries
    except ImportError: return {"MD": None, "C": None}

    if df_window is None or df_window.empty or len(df_window) < 24: return {"MD": None, "C": None}
    df = df_window.copy()
    HORIZON = 8
    out = {"MD": None, "C": None}

    if "georgian_datetime" in df.columns:
        df["georgian_datetime"] = pd.to_datetime(df["georgian_datetime"], errors="coerce")
        df = df.set_index("georgian_datetime")

    safe_freq = frequency.replace("T", "min").replace("H", "h")
    scalers_dir = BASE_DIR / "scalers"
    try:
        scaler_inst = joblib.load(scalers_dir / f"scaler_inst_{frequency}.pkl")
        scaler_pellet = joblib.load(scalers_dir / f"scaler_pellet_{frequency}.pkl")
        scaler_target = joblib.load(scalers_dir / f"scaler_target_{frequency}.pkl")
    except Exception as e:
        logger.error("Horizon-8 scalers failed: %s", e)
        return out

    active_model = model_md if model_md is not None else model_c
    model_covs = list(active_model.past_covariate_components) if (active_model and getattr(active_model, "past_covariate_components", None)) else PAST_COVARIATES_COLS

    covariate_series, df_eng = _prepare_covariates(df, model_covs, scaler_inst, scaler_pellet, safe_freq)
    if covariate_series is None: return out

    if model_md is not None:
        try:
            scaled_target_md = _get_scaled_target_series(df_eng, TARGET_MD_COL, True, scaler_target, safe_freq)
            if scaled_target_md is not None:
                pred_md_scaled = model_md.predict(n=HORIZON, series=scaled_target_md, past_covariates=covariate_series)
                out["MD"] = _inverse_transform_horizon(pred_md_scaled, scaler_target, True, HORIZON, safe_freq)
        except Exception as e: logger.error("MD horizon-8 prediction failed: %s", e)

    if model_c is not None:
        try:
            scaled_target_c = _get_scaled_target_series(df_eng, TARGET_C_COL, False, scaler_target, safe_freq)
            if scaled_target_c is not None:
                pred_c_scaled = model_c.predict(n=HORIZON, series=scaled_target_c, past_covariates=covariate_series)
                out["C"] = _inverse_transform_horizon(pred_c_scaled, scaler_target, False, HORIZON, safe_freq)
        except Exception as e: logger.error("C horizon-8 prediction failed: %s", e)

    return out
