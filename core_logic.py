# core_logic.py
import os
import sys
import uuid
import shutil
import logging
import subprocess
import threading
from pathlib import Path

import pandas as pd
import numpy as np

# Configure standard logging (INFO level for production)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Subprocess logging limits to prevent memory overflow
MAX_LINES_LOGGED = 5000
LINES_AFTER_CAP = 500
LOG_STDERR_LINES = 200

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

# The 4 specific features that do not have corresponding scalers
UNSCALED_FEATURES = [
    "INST_DELTA_T_DIFF", 
    "INST_FLOW_VAR_6H", 
    "INST_BPR_SLOPE", 
    "INST_BUSTLE_TEMP_RATE"
]

def run_script(script_name, args):
    """
    Run a Python script as a subprocess. 
    Streams output with a line cap to avoid OOM issues in production.
    """
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path)] + [str(a) for a in args]
    logger.info("Running %s", script_name)

    process = None
    stderr_lines = []
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
        )

        lines_logged = [0]
        lines_skipped = [0]
        prefix = f"[{script_name}] "

        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ""):
                    if not line: break
                    line = line.rstrip()
                    if lines_logged[0] < MAX_LINES_LOGGED:
                        try: print(prefix + line, flush=True)
                        except: pass
                        lines_logged[0] += 1
                    else:
                        lines_skipped[0] += 1
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
    """
    Handles the uploaded files, runs preprocessing scripts, and creates the merged dataset.
    """
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
    """
    Runs inference for MD and C models. Scales the input correctly and handles dynamic scalers.
    """
    import joblib
    import pandas as pd
    
    try:
        from darts import TimeSeries
    except ImportError:
        logger.warning("darts not installed; inference skipped.")
        return {"MD": None, "C": None}

    if df_window is None or df_window.empty:
        return {"MD": None, "C": None}

    df = df_window.copy()
    
    # Check minimum required lags for inference
    if len(df) < 24:
        return {"MD": None, "C": None}

    if "georgian_datetime" in df.columns:
        df["georgian_datetime"] = pd.to_datetime(df["georgian_datetime"], errors="coerce")
        df = df.set_index("georgian_datetime")

    TARGET_MD_COL = "MDNC_M_D"
    TARGET_C_COL = "MDNC_C"
    out = {"MD": None, "C": None}
    
    # Standardize frequency string for Darts compatibility
    safe_freq = frequency.replace("T", "min").replace("H", "h")

    # Load required scalers
    scalers_dir = BASE_DIR / "scalers"
    try:
        scaler_inst = joblib.load(scalers_dir / f"scaler_inst_{frequency}.pkl")
        scaler_pellet = joblib.load(scalers_dir / f"scaler_pellet_{frequency}.pkl")
        scaler_target = joblib.load(scalers_dir / f"scaler_target_{frequency}.pkl")
    except Exception as e:
        logger.error(f"Failed to load scalers: {e}")
        return {"MD": None, "C": None}

    # Extract required covariates (Priority: Model components -> Default list)
    active_model = model_md if model_md is not None else model_c
    if active_model is not None and hasattr(active_model, 'past_covariate_components') and active_model.past_covariate_components is not None:
        model_covs = list(active_model.past_covariate_components)
    else:
        model_covs = PAST_COVARIATES_COLS

    # Pad missing columns with zero
    for col in model_covs:
        if col not in df.columns:
            df[col] = 0.0

    cov_df = df[model_covs].fillna(0.0).copy()

    # Segregate columns for scaling
    pellet_cols = [c for c in model_covs if c.startswith('PELLET_')]
    inst_cols_to_scale = [c for c in model_covs if c.startswith('INST_') and c not in UNSCALED_FEATURES]

    logger.info(f"Model needs {len(model_covs)} cols total. Scalers will transform: {len(inst_cols_to_scale)} INST, {len(pellet_cols)} PELLET.")

    try:
        # Scale INST features
        if len(inst_cols_to_scale) > 0:
            ts_inst = TimeSeries.from_dataframe(cov_df[inst_cols_to_scale], freq=safe_freq)
            scaled_ts_inst = scaler_inst.transform(ts_inst)
            cov_df[inst_cols_to_scale] = scaled_ts_inst.values()

        # Scale PELLET features
        if len(pellet_cols) > 0:
            ts_pellet = TimeSeries.from_dataframe(cov_df[pellet_cols], freq=safe_freq)
            scaled_ts_pellet = scaler_pellet.transform(ts_pellet)
            cov_df[pellet_cols] = scaled_ts_pellet.values()
            
        # Reconstruct final covariate series
        covariate_series = TimeSeries.from_dataframe(cov_df[model_covs], freq=safe_freq)
    except Exception as e:
        logger.error(f"Failed to transform input covariates: {e}")
        return {"MD": None, "C": None}

    def inverse_transform_target(pred_ts, is_md=True):
        """
        Robustly reverse-scales the prediction output by trying multiple scaler formats.
        """
        col_name = TARGET_MD_COL if is_md else TARGET_C_COL
        pred_val = float(pred_ts.values()[-1][0])

        # 1. Extract the actual scaler if the loaded object is a dictionary
        actual_scaler = scaler_target
        if isinstance(scaler_target, dict):
            if col_name in scaler_target:
                actual_scaler = scaler_target[col_name]
            elif is_md and "M_D" in scaler_target:
                actual_scaler = scaler_target["M_D"]
            elif not is_md and "C" in scaler_target:
                actual_scaler = scaler_target["C"]
            else:
                # Fallback to the first available value in the dictionary
                actual_scaler = list(scaler_target.values())[0] if scaler_target else None

        if actual_scaler is None:
            return pred_val

        # 2. First attempt: Direct Darts TimeSeries inversion
        try:
            inv = actual_scaler.inverse_transform(pred_ts)
            return float(inv.values()[-1][0])
        except:
            pass

        # 3. Second attempt: Create a 2-column dummy TimeSeries (if scaler expects both)
        try:
            dummy_df = pd.DataFrame(0.0, index=pred_ts.time_index, columns=[TARGET_MD_COL, TARGET_C_COL])
            dummy_df[col_name] = pred_val
            dummy_ts = TimeSeries.from_dataframe(dummy_df, freq=safe_freq)
            inv = actual_scaler.inverse_transform(dummy_ts)
            col_idx = 0 if is_md else 1
            return float(inv.values()[-1][col_idx])
        except:
            pass

        # 4. Third attempt: Scikit-Learn 1D array
        try:
            arr = np.array([[pred_val]])
            inv = actual_scaler.inverse_transform(arr)
            return float(inv[0][0])
        except:
            pass

        # 5. Fourth attempt: Scikit-Learn 2D array
        try:
            arr = np.zeros((1, 2))
            col_idx = 0 if is_md else 1
            arr[0, col_idx] = pred_val
            inv = actual_scaler.inverse_transform(arr)
            return float(inv[0][col_idx])
        except Exception as e:
            logger.error(f"All inverse transform attempts failed: {e}")
            return pred_val  # Return the scaled value rather than crashing

    def get_target_series(target_name):
        ts_df = df[[target_name]].fillna(0.0) if target_name in df.columns else pd.DataFrame({target_name: [0.0] * len(df)}, index=df.index)
        try: return TimeSeries.from_dataframe(ts_df, freq=safe_freq)
        except: return TimeSeries.from_dataframe(ts_df)

    # Execute MD Inference
    if model_md is not None:
        try:
            target_series_md = get_target_series(TARGET_MD_COL)
            pred_md_scaled = model_md.predict(n=1, series=target_series_md, past_covariates=covariate_series)
            real_md = inverse_transform_target(pred_md_scaled, is_md=True)
            if real_md is not None: out["MD"] = round(real_md, 2)
        except Exception as e:
            logger.error(f"MD prediction failed: {e}")

    # Execute C Inference
    if model_c is not None:
        try:
            target_series_c = get_target_series(TARGET_C_COL)
            pred_c_scaled = model_c.predict(n=1, series=target_series_c, past_covariates=covariate_series)
            real_c = inverse_transform_target(pred_c_scaled, is_md=False)
            if real_c is not None: out["C"] = round(real_c, 2)
        except Exception as e:
            logger.error(f"C prediction failed: {e}")

    return out
