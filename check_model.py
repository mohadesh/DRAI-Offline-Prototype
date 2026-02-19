import pickle
import sys

# مسیر مدل خود را اینجا بگذارید
MODEL_PATH = "/media/mohadeseh/E/projects/DRAI-OFFLINE/my_inference_app/analysis/darts_pipeline_freq_30T_horizon_8_lr_0.03_depth_18/MDNC_M_D/model_MDNC_M_D.pkl"

def deep_inspect(path):
    print(f"Deep inspecting: {path}")
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        
        print(f"Model Class: {type(model)}")
        
        # Try to access inner LGBM booster
        if hasattr(model, "model"):
            lgbm_model = model.model
            print(f"Inner Model Class: {type(lgbm_model)}")
            
            if hasattr(lgbm_model, "booster_"):
                print("\n✅ Features used during training:")
                print(lgbm_model.booster_.feature_name())
            elif hasattr(lgbm_model, "feature_name_"):
                print("\n✅ Features used during training:")
                print(lgbm_model.feature_name_)
            else:
                print("Could not find feature names in inner model.")
        else:
            print("No .model attribute found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    deep_inspect(MODEL_PATH)
