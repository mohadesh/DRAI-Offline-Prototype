# app.py
import os
import uuid
import logging
import socket
import pickle
import random
import warnings
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, send_from_directory
import pandas as pd
import jdatetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- تمیز کردن لاگ‌های ترمینال ---
# نادیده گرفتن هشدارهای مربوط به نام فیچرهای LightGBM
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
# نادیده گرفتن هشدارهای مربوط به T و min در Pandas/Darts
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'T' is deprecated.*")

# ماژول‌های داخلی
import core_logic
import simulation
import model_loader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["SECRET_KEY"] = "secret-key-offline"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
DRAI_MODELING_DIR = BASE_DIR.parent / "DRAI-Modeling" / "data" / "analysis"

# Cache for inline SVG content (loaded once from disk)
_svg_content_cache = None  # type: str | None

def get_svg_content():
    """Read and cache the dashboard SVG for inline injection."""
    global _svg_content_cache
    if _svg_content_cache is None:
        svg_path = BASE_DIR / "dashboard" / "GolgoharDashboard.svg"
        try:
            raw = svg_path.read_text(encoding="utf-8")
            # Strip XML declaration so it can be embedded inline in HTML
            if raw.lstrip().startswith("<?xml"):
                raw = raw.split("?>", 1)[1].lstrip()
            _svg_content_cache = raw
        except Exception as e:
            logger.error(f"Failed to read SVG file: {e}")
            _svg_content_cache = ""
    return _svg_content_cache

# ── Simulated-prediction mode ─────────────────────────────────────────────────
# Set env var  DRAI_SIMULATE_PREDICTIONS=1  (or "true"/"yes") to bypass the
# real ML models and instead derive predicted MD/C from the actual MDNC_M_D /
# MDNC_C tag values with ±5–15 % random noise (useful for demos / dev).
_env_sim = os.environ.get("DRAI_SIMULATE_PREDICTIONS", "").strip().lower()
SIMULATE_PREDICTIONS: bool = _env_sim in ("1", "true", "yes", "on")

_MD_MAX = 97.0   # clamp ceiling for Metallization Degree (%)
_C_MAX  = 10.0   # clamp ceiling for Carbon content (%)
_C_MIN  = 0.0

def _simulated_predictions(current_data):
    """Return {"MD": float|None, "C": float|None} with ±5–15 % noise on real tag values."""
    def _get_tag(data, tag_name):
        for key, v in data.items():
            key_str = str(key).strip()
            if key_str == tag_name or key_str.endswith(tag_name):
                try:
                    f = float(v)
                    return f if not (f != f) else None  # NaN guard
                except (TypeError, ValueError):
                    pass
        return None

    def _noisy(value, lo, max_val, pct_lo, pct_hi):
        if value is None:
            return None
        sign   = random.choice((-1, 1))
        pct    = random.uniform(pct_lo, pct_hi)
        result = value * (1 + sign * pct)
        return round(min(max(result, lo), max_val), 2)

    md_raw = _get_tag(current_data, "MDNC_M_D")
    c_raw  = _get_tag(current_data, "MDNC_C")

    return {
        "MD": _noisy(md_raw, 0.0, _MD_MAX, 0.01, 0.02),   # MD: ±1–3 %
        "C":  _noisy(c_raw,  _C_MIN, _C_MAX, 0.05, 0.15), # C:  ±5–15 %
    }

# تنظیمات فرکانس
FREQUENCY_OPTIONS = ("1h", "15T", "30T")
DEFAULT_MODEL_FREQUENCY = "30T"
FREQUENCY_PATTERNS = {
    "1h": "darts_pipeline_freq_1h_*",
    "15T": "darts_pipeline_freq_15T_*",
    "30T": "darts_pipeline_freq_30T_*",
}

# کش مدل‌ها در رم
MODEL_CACHE = {}
_simulation_runners = {}

def get_model_paths_for_frequency(frequency):
    pattern = FREQUENCY_PATTERNS.get(frequency)
    if not pattern: return None, None
    for base in (ANALYSIS_DIR, DRAI_MODELING_DIR):
        if not base.exists(): continue
        matches = list(base.glob(pattern))
        if matches:
            matches.sort()
            latest_folder = matches[-1]
            md_path = latest_folder / "MDNC_M_D" / "model_MDNC_M_D.pkl"
            c_path = latest_folder / "MDNC_C" / "model_MDNC_C.pkl"
            return (md_path, c_path)
    if frequency == "30T":
        md = MODELS_DIR / "model_MDNC_M_D.pkl"
        c = MODELS_DIR / "model_MDNC_C.pkl"
        return (md if md.exists() else None, c if c.exists() else None)
    return (None, None)

def get_cached_model(path):
    if not path: return None
    path_str = str(path)
    if path_str in MODEL_CACHE:
        return MODEL_CACHE[path_str]

    try:
        # 1. تلاش برای لود کردن مدل از طریق core_logic (اگر تابعی وجود داشته باشد)
        model = core_logic.load_model(path_str) if hasattr(core_logic, 'load_model') else None

        # 2. در غیر این صورت، استفاده از کتابخانه استاندارد pickle برای فایل‌های .pkl
        if model is None:
            with open(path_str, 'rb') as f:
                model = pickle.load(f)

        if model:
            MODEL_CACHE[path_str] = model
            logger.info(f"✅ Model cached successfully: {path.name}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model {path.name}: {e}")
        return None

@app.route("/")
def index():
    if 'session_id' not in session: session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    p_files = request.files.getlist("process_files")
    pl_files = request.files.getlist("pellet_files")
    m_files = request.files.getlist("md_files")

    valid = lambda f: f and f.filename.endswith(('.csv', '.xlsx'))
    p_files = [f for f in p_files if valid(f)]
    pl_files = [f for f in pl_files if valid(f)]
    m_files = [f for f in m_files if valid(f)]

    if not (p_files and pl_files and m_files):
        return jsonify({"success": False, "error": "Missing files"}), 400

    if 'session_id' not in session: session['session_id'] = str(uuid.uuid4())
    sid = session['session_id']

    model_freq = request.form.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
    if model_freq not in FREQUENCY_OPTIONS: model_freq = DEFAULT_MODEL_FREQUENCY
    session["model_frequency"] = model_freq

    try:
        logger.info(f"Starting session: {sid} with frequency: {model_freq}")
        result = core_logic.process_data(
            process_files=p_files,
            pellet_files=pl_files,
            md_files=m_files,
            resample_rate=model_freq
        )

        runner = simulation.SimulationRunner(result['merged_df'])
        _simulation_runners[sid] = runner

        md_path, c_path = get_model_paths_for_frequency(model_freq)
        if md_path: get_cached_model(md_path)
        if c_path: get_cached_model(c_path)

        return jsonify({"success": True, "message": "Processed successfully", "stats": result['stats']})
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/simulation/start", methods=["POST"])
def sim_start():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.start(); return jsonify({"success": True})
    return jsonify({"success": False, "error": "No data"}), 400

@app.route("/simulation/pause", methods=["POST"])
def sim_pause():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.pause(); return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/simulation/resume", methods=["POST"])
def sim_resume():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.resume(); return jsonify({"success": True})
    return jsonify({"success": False}), 400

@app.route("/simulation/reset", methods=["POST"])
def sim_reset():
    runner = _simulation_runners.get(session.get('session_id'))
    if runner: runner.reset(); return jsonify({"success": True})
    return jsonify({"success": False}), 400

# Allowed process tags for the graphical dashboard (exact or suffix match)
ALLOWED_PROCESS_TAGS = [
    "AITC10", "AITC09", "FTA22A36", "TTA351", "AITA321", "AITA311", "FTA33", "FTA82",
    "PDIA48", "AITA331", "TTA341", "PTA45", "WTH15", "AITA18", "TTA19", "FTA19",
    "MDNC_M_D", "MDNC_C", "Pellet_FeO", "Pellet_CCS",
    "TTA521", "TTA522", "TTA523", "TTA524", "TTA525", "TTA526", "TTA527", "TTA528", "TTA529", "TTA5210",
]

def _build_dashboard_tag_values(current_data):
    """Build a dict of tag name -> formatted value for SVG val__ elements (exact or suffix match)."""
    out = {}
    if not current_data:
        return {tag: "—" for tag in ALLOWED_PROCESS_TAGS}
    for tag in ALLOWED_PROCESS_TAGS:
        value = None
        for key, v in current_data.items():
            key_str = str(key).strip()
            if key_str == tag or key_str.endswith(tag):
                if v is not None and v != "":
                    try:
                        f = float(v)
                        value = f
                        break
                    except (TypeError, ValueError):
                        pass
        if value is not None:
            out[tag] = str(int(value)) if value == int(value) else "%.2f" % value
        else:
            out[tag] = "—"
    return out

@app.route("/dashboard/<path:filename>")
def serve_dashboard(filename):
    """Serve dashboard assets (e.g. GolgoharDashboard.svg)."""
    return send_from_directory("dashboard", filename)

@app.route("/update-dashboard", methods=["GET"])
def update_dashboard():
    """
    Endpoint for HTMX polling to get current simulation state.
    Sends a HISTORY WINDOW to the inference engine.
    """
    tag_values_default = {t: "—" for t in ALLOWED_PROCESS_TAGS}
    tag_values_default["predicted-md"] = "—"
    tag_values_default["predicted-c"] = "—"
    tag_values_default["time"] = "—"
    if 'session_id' not in session:
        return render_template(
            "partials/dashboard_full_oob.html",
            error="Session not found",
            current_data=None,
            predictions=None,
            progress=None,
            tag_values=tag_values_default,
            svg_content=get_svg_content(),
        )

    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)

    if runner is None:
        return render_template(
            "partials/dashboard_full_oob.html",
            error="No data loaded",
            current_data=None,
            predictions=None,
            progress=None,
            tag_values=tag_values_default,
            svg_content=get_svg_content(),
        )

    # Get current step data (for display)
    current_data = runner.get_current_step()
    progress = runner.get_progress()

    predictions = None
    if current_data:
        # 1. FIX JALALI DATE EXPLICITLY IN BACKEND
        if 'georgian_datetime' in current_data and pd.notnull(current_data['georgian_datetime']):
            try:
                g_date = pd.to_datetime(current_data['georgian_datetime'])
                j_date = jdatetime.datetime.fromgregorian(datetime=g_date)
                current_data['jalali_datetime_str'] = j_date.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error(f"Jalali conversion error: {e}")

        # 2. PREDICTION — only while simulation is actively stepping forward
        if SIMULATE_PREDICTIONS and runner.is_running and not runner.is_paused:
            predictions = _simulated_predictions(current_data)
            logger.debug("SIMULATE_PREDICTIONS active: MD=%s  C=%s",
                         predictions.get("MD"), predictions.get("C"))
        elif runner.is_running and not runner.is_paused:
            WINDOW_SIZE = 120
            start_idx = max(0, runner.current_index - WINDOW_SIZE)
            end_idx = runner.current_index + 1
            df_window = runner.df.iloc[start_idx:end_idx].copy()

            freq = session.get("model_frequency", DEFAULT_MODEL_FREQUENCY)
            md_path, c_path = get_model_paths_for_frequency(freq)

            model_md = get_cached_model(md_path)
            model_c = get_cached_model(c_path)

            if model_md or model_c:
                predictions = core_logic.run_inference_for_md_c(
                    model_md,
                    model_c,
                    df_window,
                    frequency=freq
                )

    # Advance to next step
    if runner.is_running and not runner.is_paused:
        runner.get_next_step()

    tag_values = _build_dashboard_tag_values(current_data)
    # Add predicted MD and C to tag_values for SVG val__predicted-md and val__predicted-c
    if predictions is not None and predictions.get("MD") is not None:
        tag_values["predicted-md"] = "%.2f" % predictions["MD"]
    else:
        tag_values["predicted-md"] = "—"
    if predictions is not None and predictions.get("C") is not None:
        tag_values["predicted-c"] = "%.2f" % predictions["C"]
    else:
        tag_values["predicted-c"] = "—"

    # Add Jalali date/time for SVG val__time (multiple possible column names from merge)
    if current_data:
        tag_values["time"] = (
            current_data.get("enter_jalali_datetime_str")
            or current_data.get("jalali_datetime_str")
            or current_data.get("jalali_datetime")
            or str(current_data.get("georgian_datetime", "—"))
        )
    else:
        tag_values["time"] = "—"

    partial_rest = request.args.get("partial") == "rest"

    if partial_rest:
        return render_template(
            "partials/dashboard_rest.html",
            current_data=current_data,
            predictions=predictions,
            progress=progress,
            tag_values=tag_values,
            error=None,
        )
    return render_template(
        "partials/dashboard_full_oob.html",
        current_data=current_data,
        predictions=predictions,
        progress=progress,
        tag_values=tag_values,
        svg_content=get_svg_content(),
        error=None,
    )

def find_free_port(start_port=8000, max_tries=100):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise IOError("No free port found")

if __name__ == "__main__":
    try:
        # خاموش کردن لاگ‌های اضافی Werkzeug برای خوانایی بهتر ترمینال
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        port = find_free_port()
        print(f"\n========================================")
        print(f" DRAI OFFLINE SERVER STARTED ")
        print(f" Server URL: http://127.0.0.1:{port}")
        print(f"========================================\n")
        app.run(debug=True, port=port, use_reloader=False)
    except IOError as e:
        print(f"Error: {e}")
