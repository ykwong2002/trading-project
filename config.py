"""
Configuration for RLVR: assets, paths, FinBERT model, and bandit parameters.
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Assets: list of (asset_key, yfinance_ticker) for download and training
ASSETS = [
    ("gold", "GLD"),      # Gold ETF; use "GC=F" for futures if preferred
    ("nvidia", "NVDA"),
]

# FinBERT model name (Hugging Face)
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# News and bandit
NEWS_WINDOW_DAYS = 30
ACTION_SPACE_SIZE = 3  # long / short / no_trade

# Training
DEFAULT_CASH = 100_000  # USD for backtest
