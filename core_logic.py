# core_logic.py
import os
import sys
import uuid
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import joblib

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

MAX_LINES_LOGGED = 5000
LINES_AFTER_CAP = 500
LOG_STDERR_LINES = 200

_DEBUG_START_TIME = datetime.now().strftime("%Y%m%d_%H%M")

# MLOps debug: save final model inputs when SAVE_FINAL_MODEL_INPUT=true
_save_final_input = os.environ.get("SAVE_FINAL_MODEL_INPUT", "").strip().lower() in ("true", "1", "yes", "on")
_debug_data_folder = (os.environ.get("DEBUG_DATA_FOLDER", "").strip() or "debug_model_inputs").strip()
DEBUG_DATA_DIR = BASE_DIR / _debug_data_folder

def _save_debug_model_inputs(df_eng, covariate_series, frequency, suffix="single"):
    """Save model input data to DEBUG_DATA_FOLDER when SAVE_FINAL_MODEL_INPUT is true."""
    if not _save_final_input or df_eng is None:
        return
    try:
        DEBUG_DATA_DIR.mkdir(exist_ok=True, parents=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"model_input_{frequency}_{suffix}_{ts_str}"
        path_eng = DEBUG_DATA_DIR / f"{prefix}_df_eng.csv"
        df_eng.to_csv(path_eng, index=True)
        logger.info("Saved debug model input: %s", path_eng)
        if covariate_series is not None:
            try:
                cov_df = covariate_series.pd_dataframe()
                path_cov = DEBUG_DATA_DIR / f"{prefix}_covariates.csv"
                cov_df.to_csv(path_cov, index=True)
                logger.info("Saved debug covariates: %s", path_cov)
            except Exception as e:
                logger.warning("Could not save covariates to CSV: %s", e)
    except Exception as e:
        logger.warning("Failed to save debug model inputs: %s", e)

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

    if any(c in df.columns for c in upper):
        df['INST_TEMP_UPPER'] = _safe_mean(upper, df)
    if any(c in df.columns for c in middle):
        df['INST_TEMP_MIDDLE'] = _safe_mean(middle, df)
    if any(c in df.columns for c in lower):
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
    if isinstance(scaler_obj, dict) and len(scaler_obj) > 0:
        return list(scaler_obj.values())[0]
    return scaler_obj

# --- ROBUST SCALING WRAPPERS ---
def _transform_robust(df_subset, scaler_obj, safe_freq, name=""):
    """
    Robust scaling that handles both raw Scikit-Learn and Darts Scalers.
    Logs Verification info to prove Scaling happened.
    """
    actual_scaler = _extract_scaler_if_dict(scaler_obj)
    mean_before = df_subset.iloc[:,0].mean() if not df_subset.empty else 0
    logger.info(f"[{name}] SCALING BEFORE -> Shape: {df_subset.shape}, Mean(Col 0): {mean_before:.3f}")

    is_darts = "darts" in str(type(actual_scaler)).lower()

    if is_darts:
        from darts import TimeSeries
        ts = TimeSeries.from_dataframe(df_subset, freq=safe_freq)
        scaled_ts = actual_scaler.transform(ts)
        vals = scaled_ts.values()
    else:
        vals = actual_scaler.transform(df_subset.values)

    if vals.ndim == 3: vals = vals[:, :, 0]

    mean_after = np.mean(vals[:,0]) if len(vals) > 0 else 0
    std_after = np.std(vals[:,0]) if len(vals) > 0 else 0
    logger.info(f"[{name}] SCALING AFTER  -> Shape: {vals.shape}, Mean: {mean_after:.3f}, Std: {std_after:.3f}")

    return vals

def _inverse_transform_robust(pred_ts, scaler_obj, safe_freq, name=""):
    """Robust inverse scaling supporting both sklearn and Darts."""
    actual_scaler = _extract_scaler_if_dict(scaler_obj)
    is_darts = "darts" in str(type(actual_scaler)).lower()

    if is_darts:
        inv = actual_scaler.inverse_transform(pred_ts)
        vals = inv.values()
    else:
        vals_in = pred_ts.values()
        if vals_in.ndim == 3: vals_in = vals_in[:,:,0] # (n, 1)
        # sklearn standardscaler requires 2D
        if vals_in.ndim == 1: vals_in = vals_in.reshape(-1, 1)
        vals = actual_scaler.inverse_transform(vals_in)

    if vals.ndim == 3: vals = vals[:, :, 0]
    return vals

# --- COVARIATE PREPARATION ---
def _prepare_covariates_exact(df, inst_cols_list, pellet_cols_list, scaler_inst, scaler_pellet, safe_freq):
    from darts import TimeSeries

    df_eng = apply_feature_engineering(df).copy()

    all_needed_cols = list(inst_cols_list) + list(pellet_cols_list)
    for c in all_needed_cols:
        if c not in df_eng.columns:
            df_eng[c] = np.nan

    df_eng = df_eng.interpolate(method="linear", limit=1, limit_direction="both")
    df_eng = df_eng.ffill().bfill().fillna(0.0)

    cols_to_smooth = [c for c in inst_cols_list if c in df_eng.columns]
    if cols_to_smooth:
        df_eng[cols_to_smooth] = df_eng[cols_to_smooth].rolling(window=5, min_periods=1).mean().bfill().fillna(0.0)

    cov_df = df_eng.copy()

    if len(inst_cols_list) > 0 and scaler_inst is not None:
        try:
            df_inst_subset = cov_df[inst_cols_list]
            vals = _transform_robust(df_inst_subset, scaler_inst, safe_freq, "INST_Features")
            cov_df[inst_cols_list] = vals
        except Exception as e:
            logger.error(f"CRITICAL: Scaler INST failed! Data will not be fed to model. Error: {e}")
            return None, None # Force failure rather than silent bad prediction

    if len(pellet_cols_list) > 0 and scaler_pellet is not None:
        try:
            df_pellet_subset = cov_df[pellet_cols_list]
            vals = _transform_robust(df_pellet_subset, scaler_pellet, safe_freq, "PELLET_Features")
            cov_df[pellet_cols_list] = vals
        except Exception as e:
            logger.error(f"CRITICAL: Scaler PELLET failed! Error: {e}")
            return None, None

    full_cov_cols = list(inst_cols_list) + list(pellet_cols_list)
    covariate_series = TimeSeries.from_dataframe(cov_df[full_cov_cols], freq=safe_freq)

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

    if specific_scaler:
        try:
            df_subset = df_clean[[target_name]]
            vals = _transform_robust(df_subset, specific_scaler, safe_freq, f"Target_{target_name}")
            df_vals = pd.DataFrame(vals, columns=[target_name], index=df_clean.index)
            return TimeSeries.from_dataframe(df_vals, freq=safe_freq)
        except Exception as e:
            logger.error(f"Error scaling target {target_name}: {e}")
            return None

    return TimeSeries.from_dataframe(df_clean[[target_name]], freq=safe_freq)

def _inverse_transform_single(pred_ts, specific_scaler, is_md, safe_freq):
    pred_val = float(pred_ts.values()[-1][0])
    if specific_scaler is None: return pred_val

    try:
        vals = _inverse_transform_robust(pred_ts, specific_scaler, safe_freq, "Inv_Single")
        return float(vals[-1][0])
    except Exception as e:
        logger.error(f"Single Inverse fallback triggered: {e}")
        return pred_val

def _inverse_transform_horizon(pred_ts, specific_scaler, is_md, horizon, safe_freq):
    if pred_ts is None or len(pred_ts) != horizon: return None

    if specific_scaler is None:
        return [_to_python_float(pred_ts.values()[i][0]) for i in range(horizon)]

    try:
        vals = _inverse_transform_robust(pred_ts, specific_scaler, safe_freq, "Inv_Horizon")
        if vals is not None and len(vals) >= horizon:
            return [_to_python_float(vals[i][0]) for i in range(horizon)]
    except Exception as e:
        logger.error(f"Horizon Inverse fallback triggered: {e}")

    return [_to_python_float(pred_ts.values()[i][0]) for i in range(horizon)]

def _to_python_float(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) if hasattr(np, "isnan") else (f != f)) or np.isinf(f) else round(f, 3)
    except (TypeError, ValueError): return None

# --- Process / Pipeline Functions ---
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
    session_dir.mkdir(exist_ok=True, parents=True)
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

# --- Inference Functions ---

def run_inference_for_md_c(model_md, model_c, df_window, frequency="30T"):
    try: from darts import TimeSeries
    except ImportError: return {"MD": None, "C": None}

    # CHANGE HERE: Increased from 6 to 35 to satisfy Darts min_lags=24
    if df_window is None or df_window.empty or len(df_window) < 24:
        return {"MD": None, "C": None}

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
        scaler_target_md = joblib.load(scalers_dir / f"scaler_target_MDNC_M_D_{frequency}.pkl")
        scaler_target_c = joblib.load(scalers_dir / f"scaler_target_MDNC_C_{frequency}.pkl")

        inst_cols_list = joblib.load(scalers_dir / f"inst_columns_{frequency}.pkl")
        pellet_cols_list = joblib.load(scalers_dir / f"pellet_columns_{frequency}.pkl")

    except Exception as e:
        logger.error(f"Failed to load scalers or column definitions: {e}")
        return {"MD": None, "C": None}

    covariate_series, df_eng = _prepare_covariates_exact(
        df, inst_cols_list, pellet_cols_list, scaler_inst, scaler_pellet, safe_freq
    )

    if covariate_series is None: return {"MD": None, "C": None}

    _save_debug_model_inputs(df_eng, covariate_series, frequency, "single")

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
    try: from darts import TimeSeries
    except ImportError: return {"MD": None, "C": None}

    # CHANGE HERE: Increased from 6 to 35 to satisfy Darts min_lags=24
    if df_window is None or df_window.empty or len(df_window) < 35:
        return {"MD": None, "C": None}

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

        inst_cols_list = joblib.load(scalers_dir / f"inst_columns_{frequency}.pkl")
        pellet_cols_list = joblib.load(scalers_dir / f"pellet_columns_{frequency}.pkl")
    except Exception as e:
        logger.error(f"Failed to load scalers for horizon: {e}")
        return out

    covariate_series, df_eng = _prepare_covariates_exact(
        df, inst_cols_list, pellet_cols_list, scaler_inst, scaler_pellet, safe_freq
    )

    if covariate_series is None: return out

    _save_debug_model_inputs(df_eng, covariate_series, frequency, "horizon8")

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
