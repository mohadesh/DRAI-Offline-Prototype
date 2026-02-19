# core_logic.py
import os
import sys
import uuid
import shutil
import logging
import pandas as pd
import subprocess
from pathlib import Path

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"  # پوشه برای خروجی‌های میانی

# اطمینان از وجود پوشه‌ها
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def run_script(script_name, args):
    """
    اجرای یک اسکریپت پایتون دیگر به عنوان یک فرآیند جداگانه.
    خروجی را به صورت زنده (Real-time) در ترمینال چاپ می‌کند.
    """
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path)] + args
    
    logger.info(f"🚀 Running {script_name} with args: {args}")
    
    try:
        # استفاده از Popen برای خواندن خروجی به صورت زنده
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8', 
            errors='replace' # جلوگیری از کرش کردن روی کاراکترهای خاص فارسی
        )

        # خواندن و چاپ خط به خط خروجی استاندارد
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{script_name}] {output.strip()}")

        # بررسی خطاهای احتمالی
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"[{script_name} ERROR] {stderr_output.strip()}")

        if process.returncode != 0:
            raise RuntimeError(f"{script_name} failed with return code {process.returncode}")

        logger.info(f"✅ {script_name} completed successfully.")

    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        raise

def save_uploaded_files(files, folder):
    """ذخیره لیست فایل‌های آپلود شده در یک پوشه خاص"""
    saved_paths = []
    for file in files:
        if file:
            file_path = folder / file.filename
            file.save(file_path)
            saved_paths.append(str(file_path))
    return saved_paths

def load_model(path):
    """لود کردن مدل (به صورت Placeholder برای جلوگیری از خطای ایمپورت اگر فایل نبود)"""
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        return None

def run_inference_for_md_c(model_md, model_c, df_window):
    """
    اجرای اینفرنس روی مدل‌های Darts.
    در اینجا فرض بر این است که df_window دیتافریم آماده است.
    """
    try:
        # تبدیل دیتافریم به TimeSeries (مختص Darts)
        from darts import TimeSeries
        
        # نکته: نام ستون زمان باید دقیق باشد، فرض بر 'date' یا ایندکس است
        # اگر ایندکس datetime است:
        series = TimeSeries.from_dataframe(df_window)

        pred_md = model_md.predict(n=1, series=series)
        pred_c = model_c.predict(n=1, series=series)

        return {
            "MD": pred_md.values()[0][0],
            "C": pred_c.values()[0][0]
        }
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return None

def process_data(process_files, pellet_files, md_files, model_md_path=None, model_c_path=None):
    """
    تابع اصلی که توسط app.py صدا زده می‌شود.
    تمام مراحل را مدیریت و اسکریپت‌ها را زنجیره‌ای اجرا می‌کند.
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    logger.info(f"🏁 Starting processing for Session: {session_id}")

    try:
        # 1. ذخیره فایل‌های ورودی در پوشه موقت
        process_paths = save_uploaded_files(process_files, session_dir)
        pellet_paths = save_uploaded_files(pellet_files, session_dir)
        md_paths = save_uploaded_files(md_files, session_dir)

        # تعریف مسیرهای خروجی میانی
        output_process = session_dir / "Process_Cleaned.csv"
        output_pellet = session_dir / "Pellet_Cleaned.csv"
        output_md = session_dir / "MD_Cleaned.csv"
        output_merged = session_dir / "Merged_Final.csv"

        # ---------------------------------------------------------
        # STEP 1: ProcessTags.py
        # ---------------------------------------------------------
        # نکته: فرض می‌کنیم فایل اول پروسس فایل اصلی است
        run_script("ProcessTags.py", [
            "--input", process_paths[0],
            "--output", str(output_process),
            "--resample-rate", "30T"
        ])

        # ---------------------------------------------------------
        # STEP 2: Pellet.py
        # ---------------------------------------------------------
        # فرض: پلت می‌تواند چند فایل باشد یا یکی. فعلا اولی را پاس می‌دهیم
        run_script("Pellet.py", [
            "--input", pellet_paths[0],
            "--output", str(output_pellet)
        ])

        # ---------------------------------------------------------
        # STEP 3: MDnC.py
        # ---------------------------------------------------------
        run_script("MDnC.py", [
            "--input", md_paths[0],
            "--output", str(output_md)
        ])

        # ---------------------------------------------------------
        # STEP 4: Merging (merging.py)
        # ---------------------------------------------------------
        run_script("merging.py", [
            "--process", str(output_process),
            "--pellet", str(output_pellet),
            "--md", str(output_md),
            "--output", str(output_merged)
        ])

        # ---------------------------------------------------------
        # STEP 5: Load Result & Return
        # ---------------------------------------------------------
        if output_merged.exists():
            final_df = pd.read_csv(output_merged)
            
            # Ensure georgian_datetime is proper datetime
            if 'georgian_datetime' in final_df.columns:
                final_df['georgian_datetime'] = pd.to_datetime(final_df['georgian_datetime'])
            elif 'date' in final_df.columns:
                final_df = final_df.rename(columns={'date': 'georgian_datetime'})
                final_df['georgian_datetime'] = pd.to_datetime(final_df['georgian_datetime'])
            elif 'Date' in final_df.columns:
                final_df = final_df.rename(columns={'Date': 'georgian_datetime'})
                final_df['georgian_datetime'] = pd.to_datetime(final_df['georgian_datetime'])
            
            # Drop rows where all data columns are NaN (empty time slots)
            data_cols = [c for c in final_df.columns if c.startswith(('INST_', 'PELLET_', 'MDNC_'))]
            if data_cols:
                final_df = final_df.dropna(subset=data_cols, how='all')
            
            logger.info(f"🎉 All steps completed. Final shape: {final_df.shape}")
            
            return {
                "success": True,
                "merged_df": final_df,
                "stats": {
                    "rows": len(final_df),
                    "columns": list(final_df.columns)
                }
            }
        else:
            raise FileNotFoundError("Merged file was not created. Check script outputs above.")

    except Exception as e:
        logger.error(f"💥 Critical Error in pipeline: {e}")
        raise e
    
    finally:
        # پاکسازی فایل‌های موقت (اختیاری - برای دیباگ فعلا کامنت شده)
        # shutil.rmtree(session_dir, ignore_errors=True)
        pass
