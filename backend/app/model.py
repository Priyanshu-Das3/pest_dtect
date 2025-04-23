import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pest_detector_model.pkl")

_feature_ranges = {
    "Moisture_Sensor": (0, 100),
    "Humidity": (20, 100),
    "Temperature": (10, 50),
    "Infrared_Sensor": (0.1, 1.0),
    "Motion_Sensor": (0, 1),
    "Vibration_Sensor": (0, 1),
    "Gas_Sensor": (0, 1023)
}

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def get_feature_ranges():
    return _feature_ranges
