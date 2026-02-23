# core_logic.py
import os
import sys
import uuid
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Configure standard logging (INFO level for production)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

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
    """
    Calculates features on RAW data. 
    The new scaler_inst incorporates these engineered features natively.
    """
    df = df.copy()

    # 1. Zone temperatures
    middle = ['INST_TTA521', 'INST_TTA522', 'INST_TTA523', 'INST_TTA524']
    upper = ['INST_TTA525', 'INST_TTA526', 'INST_TTA527']
    lower = ['INST_TTA528', 'INST_TTA529', 'INST_TTA5210']
    df['INST_TEMP_UPPER'] = _safe_mean(upper, df)
    df['INST_TEMP_MIDDLE'] = _safe_mean(middle, df)
    df['INST_TEMP_LOWER'] = _safe_mean(lower, df)

    # 2. Delta features
    if 'INST_TEMP_MIDDLE' in df.columns and 'INST_TEMP_UPPER' in df.columns:
        df['INST_DELTA_T1'] = np.abs((df['INST_TEMP_MIDDLE'] - df['INST_TEMP_UPPER'])) / 2
    if 'INST_TEMP_MIDDLE' in df.columns and 'INST_TEMP_LOWER' in df.columns:
        df['INST_DELTA_T2'] = np.abs((df['INST_TEMP_MIDDLE'] - df['INST_TEMP_LOWER'])) / 2
    if 'INST_DELTA_T1' in df.columns and 'INST_DELTA_T2' in df.columns:
        df['INST_DELTA_T_DIFF'] = (df['INST_DELTA_T1'] - df['INST_DELTA_T2']).abs()

    # 3. Flow variance
    if 'INST_FTA19' in df.columns:
        df['INST_FLOW_VAR_6H'] = df['INST_FTA19'].rolling(6, min_periods=1).std()

    # 4. Slopes
    if 'INST_PTA45' in df.columns:
        df['INST_BPR_SLOPE'] = df['INST_PTA45'].diff()
    if 'INST_TTA341' in df.columns:
        df['INST_BUSTLE_TEMP_RATE'] = df['INST_TTA341'].diff()

    return df

# --- Shared inference helpers ---
TARGET_MD_COL = "MDNC_M_D"
TARGET_C_COL = "MDNC_C"

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

def _extract_scaler_if_dict(scaler_obj):
    """If the loaded pickle is a dictionary, extract the actual scaler object."""
    if isinstance(scaler_obj, dict) and len(scaler_obj) > 0:
        return list(scaler_obj.values())[0]
    return scaler_obj

def _prepare_covariates(df, model_covs, scaler_inst, scaler_pellet, safe_freq, frequency_for_debug="30T"):
    from darts import TimeSeries

    # 1. Feature Engineering on RAW data
    df_eng = apply_feature_engineering(df).copy()

    # 2. Ensure columns exist
    for col in model_covs:
        if col not in df_eng.columns:
            df_eng[col] = np.nan

    cov_df = df_eng[model_covs].copy()

    # 3. Interpolate & Fill
    inst_cols = [c for c in model_covs if c.startswith("INST_")]
    if inst_cols:
        existing_inst = [c for c in inst_cols if c in cov_df.columns]
        if existing_inst:
            cov_df[existing_inst] = cov_df[existing_inst].interpolate(method="linear", limit=1, limit_direction="both")

    cov_df = cov_df.ffill().bfill().fillna(0.0)

    # 4. Smooth INST features (Rolling 5)
    if inst_cols:
        cov_df[inst_cols] = cov_df[inst_cols].rolling(window=5, min_periods=1).mean().bfill().fillna(0.0)

    # 5. SPLIT: Base vs Pellet
    pellet_cols = [c for c in model_covs if c.startswith("PELLET_")]

    # 6. Scale ALL Inst features (Base 37 + 4 Eng) using the new scaler_inst
    if len(inst_cols) > 0:
        ts_inst = TimeSeries.from_dataframe(cov_df[inst_cols], freq=safe_freq)
        try:
            actual_inst_scaler = _extract_scaler_if_dict(scaler_inst)
            scaled_ts_inst = actual_inst_scaler.transform(ts_inst)
            vals = scaled_ts_inst.values()
            if vals.ndim == 3: vals = vals[:, :, 0]
            cov_df[inst_cols] = vals
        except Exception as e:
            logger.error(f"Scaler INST mismatch/error: {e}")

    # 7. Scale Pellet
    if len(pellet_cols) > 0:
        ts_pellet = TimeSeries.from_dataframe(cov_df[pellet_cols], freq=safe_freq)
        try:
            actual_pellet_scaler = _extract_scaler_if_dict(scaler_pellet)
            scaled_ts_pellet = actual_pellet_scaler.transform(ts_pellet)
            vals = scaled_ts_pellet.values()
            if vals.ndim == 3: vals = vals[:, :, 0]
            cov_df[pellet_cols] = vals
        except Exception as e:
            logger.error(f"Scaler PELLET mismatch/error: {e}")

    try:
        save_debug_data(cov_df, prefix=f"inference_input_{frequency_for_debug}")
    except Exception:
        pass

    covariate_series = TimeSeries.from_dataframe(cov_df[model_covs], freq=safe_freq)
    return covariate_series, df_eng

def _get_scaled_target_series(df, target_name, is_md, specific_scaler, safe_freq):
    try: from darts import TimeSeries
    except ImportError: return None

    df_clean = df.copy()
    if target_name in df_clean.columns:
        df_clean[target_name] = _preprocess_target_like_training(df_clean[target_name], target_name)
    else:
        fallback = 92.0 if is_md else 2.0
        df_clean[target_name] = fallback

    ts_1d = TimeSeries.from_dataframe(df_clean[[target_name]], freq=safe_freq)
    
    if specific_scaler:
        try:
            actual_scaler = _extract_scaler_if_dict(specific_scaler)
            return actual_scaler.transform(ts_1d)
        except Exception as e: 
            logger.error(f"Error scaling target {target_name}: {e}")
            
    return ts_1d

def _inverse_transform_single(pred_ts, specific_scaler, is_md, safe_freq):
    pred_val = float(pred_ts.values()[-1][0])
    if specific_scaler is None: return pred_val
    
    try:
        actual_scaler = _extract_scaler_if_dict(specific_scaler)
        inv = actual_scaler.inverse_transform(pred_ts)
        return float(inv.values()[-1][0])
    except Exception as e:
        logger.debug(f"Single Inverse fallback triggered: {e}")
        return pred_val

def _inverse_transform_horizon(pred_ts, specific_scaler, is_md, horizon, safe_freq):
    try: from darts import TimeSeries
    except ImportError: return None
    if pred_ts is None or len(pred_ts) != horizon: return None
    
    if specific_scaler is None: 
        return [_to_python_float(pred_ts.values()[i][0]) for i in range(horizon)]

    try:
        actual_scaler = _extract_scaler_if_dict(specific_scaler)
        inv = actual_scaler.inverse_transform(pred_ts)
        vals = inv.values()
        if vals is not None and len(vals) >= horizon:
            return [_to_python_float(vals[i][0]) for i in range(horizon)]
    except Exception as e: 
        logger.debug(f"Horizon Inverse fallback triggered: {e}")
        
    return [_to_python_float(pred_ts.values()[i][0]) for i in range(horizon)]

def _to_python_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) if hasattr(np, "isnan") else (f != f)) or np.isinf(f) else round(f, 3)
    except (TypeError, ValueError): return None

def save_debug_data(data, prefix="model_input"):
    is_enabled = os.getenv("SAVE_FINAL_MODEL_INPUT", "False").lower() in ("true", "1", "t")
    if not is_enabled: return
    try:
        folder_name = os.getenv("DEBUG_DATA_FOLDER", "debug_model_inputs")
        debug_dir = BASE_DIR / folder_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(data, 'pd_dataframe'): df_to_save = data.pd_dataframe()
        elif isinstance(data, pd.DataFrame): df_to_save = data
        else: df_to_save = pd.DataFrame(data)
        accumulated_file = debug_dir / f"{prefix}_accumulated_{_DEBUG_START_TIME}.csv"
        last_row = df_to_save.iloc[[-1]]
        if not accumulated_file.exists(): last_row.to_csv(accumulated_file, mode='w', header=True, index=True)
        else: last_row.to_csv(accumulated_file, mode='a', header=False, index=True)
        latest_window_file = debug_dir / f"{prefix}_latest_window.csv"
        df_to_save.to_csv(latest_window_file, mode='w', header=True, index=True)
    except Exception: pass

def run_script(script_name, args):
    script_path = BASE_DIR / script_name
    if not script_path.exists(): raise FileNotFoundError(f"Script not found: {script_path}")
    command = [sys.executable, str(script_path)] + [str(a) for a in args]
    logger.info("Running %s", script_name)
    process = None
    stderr_lines = []
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True, encoding="utf-8", errors="replace", cwd=str(BASE_DIR))
        lines_logged = [0]
        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ""):
                    if not line: break
                    if lines_logged[0] < MAX_LINES_LOGGED:
                        try: print(f"[{script_name}] " + line.rstrip(), flush=True)
                        except: pass
                        lines_logged[0] += 1
            except: pass
        def read_stderr():
            try:
                for line in iter(process.stderr.readline, ""):
                    if line: stderr_lines.append(line.rstrip())
            except: pass
        out_thread = threading.Thread(target=read_stdout, daemon=True)
        err_thread = threading.Thread(target=read_stderr, daemon=True)
        out_thread.start()
        err_thread.start()
        returncode = process.wait()
        out_thread.join(timeout=5.0)
        err_thread.join(timeout=2.0)
        if returncode != 0:
            tail = "\n".join(stderr_lines[-LOG_STDERR_LINES:]) if stderr_lines else "(no stderr)"
            raise RuntimeError(f"{script_name} failed with exit code {returncode}. Stderr:\n{tail}")
    except Exception as e:
        logger.exception("Error running %s", script_name)
        raise RuntimeError(f"Subprocess error: {e}") from e

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
    
    # Load 4 distinct scalers based on the DS's new structure
    try:
        scaler_inst = joblib.load(scalers_dir / f"scaler_inst_{frequency}.pkl")
        scaler_pellet = joblib.load(scalers_dir / f"scaler_pellet_{frequency}.pkl")
        scaler_target_md = joblib.load(scalers_dir / f"scaler_target_MDNC_M_D_{frequency}.pkl")
        scaler_target_c = joblib.load(scalers_dir / f"scaler_target_MDNC_C_{frequency}.pkl")
    except Exception as e:
        logger.error(f"Failed to load new scalers: {e}")
        return {"MD": None, "C": None}
        
    active_model = model_md if model_md is not None else model_c
    model_covs = list(active_model.past_covariate_components) if (active_model and getattr(active_model, "past_covariate_components", None)) else PAST_COVARIATES_COLS
    
    covariate_series, df_eng = _prepare_covariates(df, model_covs, scaler_inst, scaler_pellet, safe_freq, frequency_for_debug=frequency)
    if covariate_series is None: return {"MD": None, "C": None}
    
    if model_md is not None:
        try:
            scaled_target_md = _get_scaled_target_series(df_eng, TARGET_MD_COL, is_md=True, specific_scaler=scaler_target_md, safe_freq=safe_freq)
            if scaled_target_md is not None:
                pred_md_scaled = model_md.predict(n=1, series=scaled_target_md, past_covariates=covariate_series)
                real_md = _inverse_transform_single(pred_md_scaled, specific_scaler=scaler_target_md, is_md=True, safe_freq=safe_freq)
                if real_md is not None: out["MD"] = round(real_md, 2)
        except Exception as e: logger.error("MD pred failed: %s", e)
        
    if model_c is not None:
        try:
            scaled_target_c = _get_scaled_target_series(df_eng, TARGET_C_COL, is_md=False, specific_scaler=scaler_target_c, safe_freq=safe_freq)
            if scaled_target_c is not None:
                pred_c_scaled = model_c.predict(n=1, series=scaled_target_c, past_covariates=covariate_series)
                real_c = _inverse_transform_single(pred_c_scaled, specific_scaler=scaler_target_c, is_md=False, safe_freq=safe_freq)
                if real_c is not None: out["C"] = round(real_c, 2)
        except Exception as e: logger.error("C pred failed: %s", e)
        
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
        scaler_target_md = joblib.load(scalers_dir / f"scaler_target_MDNC_M_D_{frequency}.pkl")
        scaler_target_c = joblib.load(scalers_dir / f"scaler_target_MDNC_C_{frequency}.pkl")
    except Exception as e:
        logger.error(f"Failed to load new scalers for horizon: {e}")
        return out
        
    active_model = model_md if model_md is not None else model_c
    model_covs = list(active_model.past_covariate_components) if (active_model and getattr(active_model, "past_covariate_components", None)) else PAST_COVARIATES_COLS
    
    covariate_series, df_eng = _prepare_covariates(df, model_covs, scaler_inst, scaler_pellet, safe_freq)
    if covariate_series is None: return out
    
    if model_md is not None:
        try:
            scaled_target_md = _get_scaled_target_series(df_eng, TARGET_MD_COL, is_md=True, specific_scaler=scaler_target_md, safe_freq=safe_freq)
            if scaled_target_md is not None:
                pred_md_scaled = model_md.predict(n=HORIZON, series=scaled_target_md, past_covariates=covariate_series)
                out["MD"] = _inverse_transform_horizon(pred_md_scaled, specific_scaler=scaler_target_md, is_md=True, horizon=HORIZON, safe_freq=safe_freq)
        except Exception: pass
        
    if model_c is not None:
        try:
            scaled_target_c = _get_scaled_target_series(df_eng, TARGET_C_COL, is_md=False, specific_scaler=scaler_target_c, safe_freq=safe_freq)
            if scaled_target_c is not None:
                pred_c_scaled = model_c.predict(n=HORIZON, series=scaled_target_c, past_covariates=covariate_series)
                out["C"] = _inverse_transform_horizon(pred_c_scaled, specific_scaler=scaler_target_c, is_md=False, horizon=HORIZON, safe_freq=safe_freq)
        except Exception: pass
        
    return out
