"""Configuration for the LLM"""

import os

from dotenv import load_dotenv

load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

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
