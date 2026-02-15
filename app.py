"""
سرور Flask — فقط مسئول روتینگ و وب‌هوک.
منطق پردازش در core_logic قرار دارد.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
import uuid

import core_logic
import simulation

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["SECRET_KEY"] = "drai-offline-secret-key-change-in-production"  # For sessions

# مسیرهای پروژه
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
# Try to find models in local models/ directory first, then check DRAI-Modeling
DRAI_MODELING_DIR = BASE_DIR.parent / "DRAI-Modeling" / "data" / "analysis"
# Look for a model directory (using 30T frequency as default)
_model_subdirs = list(DRAI_MODELING_DIR.glob("darts_pipeline_freq_30T_*")) if DRAI_MODELING_DIR.exists() else []
DEFAULT_MODEL_MD = None
DEFAULT_MODEL_C = None
if _model_subdirs:
    # Use the first matching directory
    model_base = _model_subdirs[0]
    md_path = model_base / "MDNC_M_D" / "model_MDNC_M_D.pkl"
    c_path = model_base / "MDNC_C" / "model_MDNC_C.pkl"
    if md_path.exists():
        DEFAULT_MODEL_MD = md_path
    if c_path.exists():
        DEFAULT_MODEL_C = c_path
# Fallback to local models directory
if DEFAULT_MODEL_MD is None:
    DEFAULT_MODEL_MD = MODELS_DIR / "model_MDNC_M_D.pkl"
if DEFAULT_MODEL_C is None:
    DEFAULT_MODEL_C = MODELS_DIR / "model_MDNC_C.pkl"

UPLOADS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# In-memory storage for simulation runners (in production, use Redis or database)
# Key: session_id, Value: SimulationRunner instance
_simulation_runners = {}


@app.route("/")
def index():
    """صفحه اصلی داشبورد."""
    # Initialize session if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    دریافت سه فایل CSV (Process Tags, Pellet, MD/Quality) و پردازش.
    """
    # Get files
    process_file = request.files.get("process_tags")
    pellet_file = request.files.get("pellet_data")
    mdnc_file = request.files.get("mdnc_data")
    
    if not all([process_file, pellet_file, mdnc_file]):
        return jsonify({
            "success": False,
            "error": "لطفاً هر سه فایل را آپلود کنید: Process Tags, Pellet, MD/Quality"
        }), 400
    
    # Ensure session exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    
    # Save files temporarily
    saved_paths = {}
    try:
        process_path = UPLOADS_DIR / f"{session_id}_process_tags.csv"
        pellet_path = UPLOADS_DIR / f"{session_id}_pellet.csv"
        mdnc_path = UPLOADS_DIR / f"{session_id}_mdnc.csv"
        
        process_file.save(str(process_path))
        pellet_file.save(str(pellet_path))
        mdnc_file.save(str(mdnc_path))
        
        saved_paths = {
            'process': str(process_path),
            'pellet': str(pellet_path),
            'mdnc': str(mdnc_path)
        }
        
        # Process and merge files
        print(f"\n[Upload] Processing files for session {session_id}...")
        result = core_logic.process_data(
            process_tags_path=saved_paths['process'],
            pellet_path=saved_paths['pellet'],
            mdnc_path=saved_paths['mdnc'],
            model_md_path=str(DEFAULT_MODEL_MD) if DEFAULT_MODEL_MD.exists() else None,
            model_c_path=str(DEFAULT_MODEL_C) if DEFAULT_MODEL_C.exists() else None
        )
        
        # Create simulation runner
        runner = simulation.SimulationRunner(result['merged_df'])
        _simulation_runners[session_id] = runner
        
        return jsonify({
            "success": True,
            "message": f"فایل‌ها با موفقیت پردازش شدند. {result['stats']['rows']} ردیف داده آماده است.",
            "stats": result['stats']
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"خطا در پردازش: {str(e)}"
        }), 500
    
    finally:
        # Clean up uploaded files after processing
        for path in saved_paths.values():
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


@app.route("/simulation/start", methods=["POST"])
def simulation_start():
    """شروع شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No data loaded. Please upload files first."}), 400
    
    runner.start()
    return jsonify({"success": True, "message": "Simulation started"})


@app.route("/simulation/pause", methods=["POST"])
def simulation_pause():
    """توقف موقت شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No simulation running"}), 400
    
    runner.pause()
    return jsonify({"success": True, "message": "Simulation paused"})


@app.route("/simulation/resume", methods=["POST"])
def simulation_resume():
    """ادامه شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No simulation running"}), 400
    
    runner.resume()
    return jsonify({"success": True, "message": "Simulation resumed"})


@app.route("/simulation/reset", methods=["POST"])
def simulation_reset():
    """بازنشانی شبیه‌سازی."""
    if 'session_id' not in session:
        return jsonify({"success": False, "error": "Session not found"}), 400
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return jsonify({"success": False, "error": "No simulation running"}), 400
    
    runner.reset()
    return jsonify({"success": True, "message": "Simulation reset"})


@app.route("/update-dashboard", methods=["GET"])
def update_dashboard():
    """
    Endpoint for HTMX polling to get current simulation state.
    Returns HTML fragment to update dashboard.
    """
    if 'session_id' not in session:
        return render_template("partials/dashboard.html", error="Session not found")
    
    session_id = session['session_id']
    runner = _simulation_runners.get(session_id)
    
    if runner is None:
        return render_template("partials/dashboard.html", error="No data loaded")
    
    # Get current step
    current_data = runner.get_current_step()
    progress = runner.get_progress()
    
    # Run inference if models are available
    predictions = None
    if current_data:
        # Create a single-row dataframe for inference
        import pandas as pd
        df_row = pd.DataFrame([current_data])
        
        model_md = core_logic.load_model(str(DEFAULT_MODEL_MD)) if DEFAULT_MODEL_MD.exists() else None
        model_c = core_logic.load_model(str(DEFAULT_MODEL_C)) if DEFAULT_MODEL_C.exists() else None
        
        if model_md or model_c:
            predictions = core_logic.run_inference_for_md_c(model_md, model_c, df_row)
    
    # Advance to next step (for next poll)
    if runner.is_running and not runner.is_paused:
        runner.get_next_step()
    
    return render_template(
        "partials/dashboard.html",
        current_data=current_data,
        predictions=predictions,
        progress=progress,
        error=None
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Legacy endpoint for backward compatibility.
    Redirects to new upload endpoint.
    """
    return upload()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
