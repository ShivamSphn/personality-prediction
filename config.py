import os
from pathlib import Path

class Config:
    # Model settings
    DEFAULT_EMBED = "bert-base"
    DEFAULT_TOKEN_LENGTH = 512
    DEFAULT_DATASET = "kaggle"
    DEFAULT_FINETUNE_MODEL = "mlp_lm"
    
    # Paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    OP_DIR = BASE_DIR / "pkl_data/"
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Minimum required interactions for accurate prediction
    MIN_INTERACTIONS = int(os.getenv("MIN_INTERACTIONS", 3))
