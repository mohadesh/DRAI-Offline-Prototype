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

# Limit lines printed from subprocess to avoid IDE buffer OOM
MAX_LINES_LOGGED = 5000
LINES_AFTER_CAP = 500  # After cap, log every N lines as progress
LOG_STDERR_LINES = 200  # Max stderr lines to include in exception


def run_script(script_name, args):
    """
    Run a Python script as a subprocess. Streams output with a line cap to avoid OOM.
    Closes pipes properly; on failure raises a clear exception without crashing the parent.
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

        lines_logged = [0]  # use list so inner function can mutate
        lines_skipped = [0]
        prefix = f"[{script_name}] "

        def read_stdout():
            try:
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    line = line.rstrip()
                    if lines_logged[0] < MAX_LINES_LOGGED:
                        try:
                            print(prefix + line, flush=True)
                        except (OSError, UnicodeEncodeError):
                            pass
                        lines_logged[0] += 1
                    else:
                        lines_skipped[0] += 1
                        if lines_skipped[0] % LINES_AFTER_CAP == 0:
                            try:
                                print(prefix + f"... ({lines_skipped[0]} more lines suppressed)", flush=True)
                            except (OSError, UnicodeEncodeError):
                                pass
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                try:
                    process.stdout.close()
                except Exception:
                    pass

        def read_stderr():
            try:
                for line in iter(process.stderr.readline, ""):
                    if line:
                        stderr_lines.append(line.rstrip())
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                try:
                    process.stderr.close()
                except Exception:
                    pass

        out_thread = threading.Thread(target=read_stdout, daemon=True)
        err_thread = threading.Thread(target=read_stderr, daemon=True)
        out_thread.start()
        err_thread.start()
        returncode = process.wait()
        out_thread.join(timeout=5.0)
        err_thread.join(timeout=2.0)

        if returncode != 0:
            tail = "\n".join(stderr_lines[-LOG_STDERR_LINES:]) if stderr_lines else "(no stderr)"
            raise RuntimeError(
                f"{script_name} failed with exit code {returncode}. Stderr:\n{tail}"
            )

        if lines_skipped[0] > 0:
            logger.info("%s produced %d extra lines (not printed).", script_name, lines_skipped[0])
        logger.info("%s completed successfully.", script_name)

    except FileNotFoundError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Error running %s", script_name)
        raise RuntimeError(f"Subprocess error running {script_name}: {e}") from e
    finally:
        if process is not None:
            try:
                process.stdout.close()
            except Exception:
                pass
            try:
                process.stderr.close()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


def save_uploaded_files(files, folder):
    """Save uploaded file objects to folder; return list of saved paths."""
    saved = []
    for f in files:
        if f and getattr(f, "filename", None):
            path = folder / f.filename
            f.save(str(path))
            saved.append(str(path))
    return saved


def process_data(process_files, pellet_files, md_files, model_md_path=None, model_c_path=None):
    """
    Main entry called by app.py. Runs ProcessTags -> Pellet -> MDnC -> merging.
    Returns dict with success, merged_df, stats. Raises on failure.
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    logger.info("Starting session: %s", session_id)

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
            "--resample-rate", "30min",
        ])

        run_script("Pellet.py", ["--input", pellet_paths[0], "--output", str(output_pellet)])

        run_script("MDnC.py", ["--input", md_paths[0], "--output", str(output_md)])

        run_script("merging.py", [
            "--process", str(output_process),
            "--pellet", str(output_pellet),
            "--md", str(output_md),
            "--output", str(output_merged),
            "--rate", "5T",
        ])

        if not output_merged.exists():
            raise FileNotFoundError("Merged file was not created. Check script outputs above.")

        final_df = pd.read_csv(output_merged)
        if "georgian_datetime" in final_df.columns:
            final_df["georgian_datetime"] = pd.to_datetime(final_df["georgian_datetime"], errors="coerce")
        logger.info("Session %s done. Rows: %d", session_id, len(final_df))

        return {
            "success": True,
            "merged_df": final_df,
            "stats": {"rows": len(final_df), "columns": list(final_df.columns)},
        }

    except Exception as e:
        logger.exception("Pipeline failed for session %s", session_id)
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass
        raise e


def run_inference_for_md_c(model_md, model_c, df_window):
    """Run Darts inference for MD and C. Returns dict with MD and C values or None on error."""
    try:
        from darts import TimeSeries
    except ImportError:
        logger.warning("darts not installed; inference skipped.")
        return None

    if df_window is None or (hasattr(df_window, "empty") and df_window.empty):
        return None

    df = df_window.copy()
    if "georgian_datetime" in df.columns:
        df["georgian_datetime"] = pd.to_datetime(df["georgian_datetime"], errors="coerce")
        df = df.set_index("georgian_datetime")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    series = TimeSeries.from_dataframe(df[numeric_cols].fillna(0))

    out = {}
    if model_md is not None:
        try:
            pred = model_md.predict(n=1, series=series)
            out["MD"] = float(pred.values().flatten()[0])
        except Exception as e:
            logger.debug("MD prediction failed: %s", e)
            out["MD"] = None
    else:
        out["MD"] = None
    if model_c is not None:
        try:
            pred = model_c.predict(n=1, series=series)
            out["C"] = float(pred.values().flatten()[0])
        except Exception as e:
            logger.debug("C prediction failed: %s", e)
            out["C"] = None
    else:
        out["C"] = None
    return out
