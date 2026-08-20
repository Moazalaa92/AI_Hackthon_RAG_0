"""Configuration for the LLM"""

import os

from dotenv import load_dotenv

load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
SAFETY_INTENT_LLM_ENABLED = os.getenv(
    "SAFETY_INTENT_LLM_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
RATE_LIMIT_PER_IP_HOUR = int(os.getenv("RATE_LIMIT_PER_IP_HOUR", "10"))
RATE_LIMIT_GLOBAL_DAILY = int(os.getenv("RATE_LIMIT_GLOBAL_DAILY", "1000"))
QUESTION_CACHE_SIZE = int(os.getenv("QUESTION_CACHE_SIZE", "256"))

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(DATA_DIR, "chroma_db"))
PDF_DIR = os.getenv("PDF_DIR", os.path.join(DATA_DIR, "pdfs"))

TOP_K: int = int(os.getenv("TOP_K", "5"))

EVAL_TOP_K: int = int(os.getenv("EVAL_TOP_K", "10"))
EVALUATION_DIR = os.getenv("EVALUATION_DIR", "evaluation")
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(EVALUATION_DIR, "results"))
METRICS_DIR = os.getenv("METRICS_DIR", os.path.join(EVALUATION_DIR, "metrics"))
EXPERIMENTS_DIR = os.getenv(
    "EXPERIMENTS_DIR", os.path.join(CHROMA_DB_DIR, "experiments")
)
