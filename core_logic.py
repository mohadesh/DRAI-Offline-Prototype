# core_logic.py — Orchestrates pipeline scripts. Safe subprocess execution to avoid OOM/crash.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ---- Debug FileHandler: persist all debug logs to a dedicated file ----
_debug_fh = logging.FileHandler(BASE_DIR / "debug_inference.log", mode="a", encoding="utf-8")
_debug_fh.setLevel(logging.DEBUG)
_debug_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_debug_fh)
logger.setLevel(logging.DEBUG)

_MODEL_INPUT_EXPORTED = False

# Limit lines printed from subprocess to avoid IDE buffer OOM
MAX_LINES_LOGGED = 5000
LINES_AFTER_CAP = 500
LOG_STDERR_LINES = 200

# =========================================================================
# Extracted features for the model (fallback in case of failure to read from the model file)
# =========================================================================
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

def run_script(script_name, args):
    """
    Run a Python script as a subprocess. Streams output with a line cap to avoid OOM.
    """
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path)] + [str(a) for a in args]
    logger.info("Running %s (args: %s)", script_name, args[:6])

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
                        try:
                            print(prefix + line, flush=True)
                        except (OSError, UnicodeEncodeError): pass
                        lines_logged[0] += 1
                    else:
                        lines_skipped[0] += 1
                        if lines_skipped[0] % LINES_AFTER_CAP == 0:
                            try:
                                print(prefix + f"... ({lines_skipped[0]} more lines suppressed)", flush=True)
                            except (OSError, UnicodeEncodeError): pass
            except (BrokenPipeError, OSError, ValueError): pass
            finally:
                try: process.stdout.close()
                except Exception: pass

        def read_stderr():
            try:
                for line in iter(process.stderr.readline, ""):
                    if line: stderr_lines.append(line.rstrip())
            except (BrokenPipeError, OSError, ValueError): pass
            finally:
                try: process.stderr.close()
                except Exception: pass

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

        if lines_skipped[0] > 0:
            logger.info("%s produced %d extra lines (not printed).", script_name, lines_skipped[0])
        logger.info("%s completed successfully.", script_name)

    except FileNotFoundError: raise
    except RuntimeError: raise
    except Exception as e:
        logger.exception("Error running %s", script_name)
        raise RuntimeError(f"Subprocess error running {script_name}: {e}") from e
    finally:
        if process is not None:
            try: process.stdout.close()
            except Exception: pass
            try: process.stderr.close()
            except Exception: pass
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try: process.kill()
                except Exception: pass

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
        if not process_paths or not pellet_paths or not md_paths:
            raise ValueError("Missing one or more required input files.")

        output_process = session_dir / "Process_Cleaned.csv"
        output_pellet = session_dir / "Pellet_Cleaned.csv"
        output_md = session_dir / "MD_Cleaned.csv"
        output_merged = session_dir / "Merged_Final.csv"

        run_script("ProcessTags.py", [
            "--input", process_paths[0],
            "--output", str(output_process),
            "--resample-rate", resample_rate,
        ])

        run_script("Pellet.py", ["--input", pellet_paths[0], "--output", str(output_pellet)])

        run_script("MDnC.py", ["--input", md_paths[0], "--output", str(output_md)])

        run_script("merging.py", [
            "--process", str(output_process),
            "--pellet", str(output_pellet),
            "--md", str(output_md),
            "--output", str(output_merged),
            "--rate", resample_rate,
        ])

        if not output_merged.exists():
            raise FileNotFoundError("Merged file was not created. Check script outputs above.")

        final_df = pd.read_csv(output_merged, parse_dates=['georgian_datetime'])
        logger.info("Session %s done. Rows: %d", session_id, len(final_df))

        # ---- DEBUG STEP 1: Inspect merged data ----
        logger.debug(
            "DEBUG_01 | final_df shape: %s | columns: %s",
            final_df.shape, list(final_df.columns),
        )
        missing_counts = final_df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if cols_with_missing.empty:
            logger.debug("DEBUG_01 | No missing values in final_df.")
        else:
            logger.debug(
                "DEBUG_01 | Columns with missing values:\n%s", cols_with_missing.to_string(),
            )
        try:
            debug_csv_path = UPLOADS_DIR / "DEBUG_01_merged_sample.csv"
            final_df.head(500).to_csv(debug_csv_path, index=False)
            logger.debug("DEBUG_01 | Exported first 500 rows -> %s", debug_csv_path)
        except Exception as exc:
            logger.warning("DEBUG_01 | Failed to export merged sample: %s", exc)

        return {
            "success": True,
            "merged_df": final_df,
            "stats": {"rows": len(final_df), "columns": list(final_df.columns)},
        }

    except Exception as e:
        logger.exception("Pipeline failed for session %s", session_id)
        try: shutil.rmtree(session_dir, ignore_errors=True)
        except Exception: pass
        raise e

def run_inference_for_md_c(model_md, model_c, df_window, frequency="30T"):
    """
    Run Darts inference for MD and C. Returns dict with MD and C values or None on error.
    """
    try:
        from darts import TimeSeries
    except ImportError:
        logger.warning("darts not installed; inference skipped.")
        return {"MD": None, "C": None}

    if df_window is None or (hasattr(df_window, "empty") and df_window.empty):
        return {"MD": None, "C": None}

    df = df_window.copy()

    # --- Check length of input data ---
    REQUIRED_LAGS = 24
    if len(df) < REQUIRED_LAGS:
        return {"MD": None, "C": None}

    if "georgian_datetime" in df.columns:
        df["georgian_datetime"] = pd.to_datetime(df["georgian_datetime"], errors="coerce")
        df = df.set_index("georgian_datetime")
    else:
        logger.warning("Inference skipped: 'georgian_datetime' not in window.")
        return {"MD": None, "C": None}

    TARGET_MD_COL = "MDNC_M_D"
    TARGET_C_COL = "MDNC_C"
    out = {"MD": None, "C": None}

    # Replace 'T' with 'min' to avoid pandas/darts FutureWarning
    safe_freq = frequency.replace("T", "min")

    # =========================================================================
    # FEATURE ENGINEERING (Translating raw tags to model-expected features)
    # Added to prevent critical values from becoming zero
    # =========================================================================
    # 1. Map main temperatures
    if "INST_ROOFTEMP" in df.columns:
        df["INST_TEMP_UPPER"] = df["INST_ROOFTEMP"]
    
    if "INST_TTA525" in df.columns: # TTA525 is assumed as Middle Temp
        df["INST_TEMP_MIDDLE"] = df["INST_TTA525"]
        
    if "INST_FLOORTEMP" in df.columns:
        df["INST_TEMP_LOWER"] = df["INST_FLOORTEMP"]

    # 2. Calculate Deltas
    if "INST_TEMP_UPPER" in df.columns and "INST_TEMP_MIDDLE" in df.columns:
        df["INST_DELTA_T1"] = df["INST_TEMP_UPPER"] - df["INST_TEMP_MIDDLE"]
        
    if "INST_TEMP_MIDDLE" in df.columns and "INST_TEMP_LOWER" in df.columns:
        df["INST_DELTA_T2"] = df["INST_TEMP_MIDDLE"] - df["INST_TEMP_LOWER"]
        
    if "INST_DELTA_T1" in df.columns and "INST_DELTA_T2" in df.columns:
        df["INST_DELTA_T_DIFF"] = df["INST_DELTA_T1"] - df["INST_DELTA_T2"]

    # 3. Calculate time-based features (Rolling / Rate)
    # Assuming 30T timeframe. 12 records = 6 hours.
    try:
        # Gas flow variance over the last 6 hours
        if "INST_FTB061" in df.columns: 
            df["INST_FLOW_VAR_6H"] = df["INST_FTB061"].rolling(window=12, min_periods=1).var().fillna(0.0)
            
        # Bustle Pressure slope
        if "INST_PTB081" in df.columns: 
            df["INST_BPR_SLOPE"] = df["INST_PTB081"].diff().fillna(0.0)
            
        # Bustle Temp rate of change
        if "INST_TTA801" in df.columns: 
            df["INST_BUSTLE_TEMP_RATE"] = df["INST_TTA801"].diff().fillna(0.0)
    except Exception as e:
        logger.warning("Feature Engineering Error (Rolling features): %s", e)
    # =========================================================================

    # --- Exact extraction of Covariates (first from model, then from fallback list) ---
    expected_covs = None
    active_model = model_md if model_md is not None else model_c

    if active_model is not None and hasattr(active_model, 'past_covariate_components'):
        expected_covs = list(active_model.past_covariate_components)

    if not expected_covs and 'PAST_COVARIATES_COLS' in globals():
        expected_covs = PAST_COVARIATES_COLS

    if not expected_covs:
        expected_covs = df.select_dtypes(include=['number']).columns.tolist()
        if TARGET_MD_COL in expected_covs: expected_covs.remove(TARGET_MD_COL)
        if TARGET_C_COL in expected_covs: expected_covs.remove(TARGET_C_COL)

    # Pad missing columns with 0.0
    for col in expected_covs:
        if col not in df.columns:
            df[col] = 0.0

    cov_df = df[list(expected_covs)].fillna(0.0)

    def get_ts(data_df):
        try:
            return TimeSeries.from_dataframe(data_df, freq=safe_freq)
        except Exception:
            return TimeSeries.from_dataframe(data_df)

    # ---- DEBUG STEP 2: Inspect exact model input ----
    global _MODEL_INPUT_EXPORTED
    try:
        zero_cols = [c for c in cov_df.columns if (cov_df[c] == 0.0).all()]
        if zero_cols:
            logger.debug(
                "DEBUG_02 | %d columns are ENTIRELY zeros (suspicious): %s",
                len(zero_cols), zero_cols,
            )
        else:
            logger.debug("DEBUG_02 | No entirely-zero columns detected in cov_df.")

        critical_features = ["PELLET_CCS", "INST_TEMP_UPPER", "INST_WTH15"]
        for feat in critical_features:
            if feat in cov_df.columns:
                logger.debug(
                    "DEBUG_02 | %s  ->  min=%.4f  max=%.4f  mean=%.4f",
                    feat, cov_df[feat].min(), cov_df[feat].max(), cov_df[feat].mean(),
                )
            else:
                logger.debug("DEBUG_02 | %s  ->  NOT PRESENT in cov_df", feat)

        if not _MODEL_INPUT_EXPORTED:
            debug_input_path = UPLOADS_DIR / "DEBUG_02_model_input_features.csv"
            cov_df.to_csv(debug_input_path)
            logger.debug("DEBUG_02 | Exported cov_df (%s) -> %s", cov_df.shape, debug_input_path)
            _MODEL_INPUT_EXPORTED = True
    except Exception as exc:
        logger.warning("DEBUG_02 | Logging failed (non-fatal): %s", exc)

    covariate_series = get_ts(cov_df)

    def get_target_series(target_name):
        if target_name in df.columns:
            ts_df = df[[target_name]].fillna(0.0)
        else:
            ts_df = pd.DataFrame({target_name: [0.0] * len(df)}, index=df.index)
        return get_ts(ts_df)

    # --- MD Inference ---
    if model_md is not None:
        try:
            target_series_md = get_target_series(TARGET_MD_COL)
            pred = model_md.predict(n=1, series=target_series_md, past_covariates=covariate_series)
            val = float(pred.values().flatten()[0])
            # ---- DEBUG STEP 3: Raw MD output ----
            logger.debug("DEBUG_03 | MD raw prediction value: %.6f", val)
            out["MD"] = round(max(0.0, min(val, 100.0)), 2)
            logger.debug("DEBUG_03 | MD after clamping: %s", out["MD"])
        except Exception as e:
            logger.error(f"MD prediction failed: {e}")
            out["MD"] = None

    # --- C Inference ---
    if model_c is not None:
        try:
            target_series_c = get_target_series(TARGET_C_COL)
            pred = model_c.predict(n=1, series=target_series_c, past_covariates=covariate_series)
            val = float(pred.values().flatten()[0])
            # ---- DEBUG STEP 3: Raw C output ----
            logger.debug("DEBUG_03 | C raw prediction value: %.6f", val)
            out["C"] = round(max(0.0, val), 2)
            logger.debug("DEBUG_03 | C after clamping: %s", out["C"])
        except Exception as e:
            logger.error(f"C prediction failed: {e}")
            out["C"] = None

    return out
