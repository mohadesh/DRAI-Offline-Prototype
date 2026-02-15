"""
اسکریپت آماده‌سازی محیط آفلاین.
۱) فایل‌های HTMX، Alpine.js و Tailwind را دانلود و در static/vendor ذخیره می‌کند.
۲) در صورت نبود مدل، یک مدل Dummy برای تست می‌سازد و در models/ ذخیره می‌کند.
این اسکریپت باید یک‌بار (با اتصال به اینترنت برای دانلود) اجرا شود.
"""

import os
import urllib.request
from pathlib import Path

# مسیرهای پروژه
BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "static" / "vendor"
MODELS_DIR = BASE_DIR / "models"
DUMMY_MODEL_PATH = MODELS_DIR / "dummy_model.pkl"

# URLهای معتبر برای دانلود (فقط برای اجرای این اسکریپت؛ در HTML از مسیر لوکال استفاده می‌شود)
ASSETS = [
    {
        "url": "https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js",
        "filename": "htmx.min.js",
    },
    {
        "url": "https://unpkg.com/alpinejs@3.14.3/dist/cdn.min.js",
        "filename": "alpine.min.js",
    },
    {
        "url": "https://cdn.tailwindcss.com",
        "filename": "tailwind.js",
    },
]


def download_file(url: str, dest_path: Path) -> None:
    """دانلود یک فایل از URL و ذخیره در مسیر مشخص."""
    print(f"  دانلود: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    dest_path.write_bytes(content)
    print(f"  ذخیره شد: {dest_path}")


def create_dummy_model() -> None:
    """ساخت یک مدل Dummy با sklearn و ذخیره با joblib برای تست."""
    if DUMMY_MODEL_PATH.exists():
        print("مدل dummy از قبل وجود دارد؛ نادیده گرفته شد.")
        return
    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        import numpy as np
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # دادهٔ مصنوعی برای fit
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)
        model = DummyClassifier(strategy="stratified")
        model.fit(X, y)
        joblib.dump(model, DUMMY_MODEL_PATH)
        print(f"مدل dummy ساخته و ذخیره شد: {DUMMY_MODEL_PATH}")
    except ImportError as e:
        print(f"برای ساخت مدل dummy به scikit-learn و joblib نیاز است: {e}")


def main() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    print("شروع دانلود فایل‌های فرانت‌اند برای محیط آفلاین...\n")
    for asset in ASSETS:
        dest = VENDOR_DIR / asset["filename"]
        try:
            download_file(asset["url"], dest)
        except Exception as e:
            print(f"  خطا در دانلود {asset['filename']}: {e}")
    print("\nبررسی مدل تست (dummy)...")
    create_dummy_model()
    print("\nپایان. فایل‌ها در پوشه static/vendor و در صورت نیاز مدل در models/ قرار گرفتند.")


if __name__ == "__main__":
    main()
