# model_loader.py
import pickle
import logging
import sys
from pathlib import Path

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(path):
    """
    Load a pickled model from the specified path.
    Handles errors gracefully to prevent app crashes.
    """
    if not path:
        return None
        
    path_obj = Path(path)
    if not path_obj.exists():
        logger.error(f"❌ Model file not found: {path}")
        return None

    try:
        logger.info(f"⏳ Loading model from: {path_obj.name}")
        with open(path_obj, "rb") as f:
            model = pickle.load(f)
            logger.info(f"✅ Model loaded successfully: {path_obj.name}")
            return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from {path}: {e}")
        return None
