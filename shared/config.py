from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "mimic-iv-ed-2.2" / "ed"
VOCABULARY_PATH = BASE_DIR / "shared" / "vocabulary.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "triage_model.pkl"
