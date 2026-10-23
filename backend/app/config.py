from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# backend/app/config.py -> app -> backend -> root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = ROOT_DIR / ".env"

class Settings(BaseSettings):
    # if these variables are not found in the environment variables or .env file, Pydantic will throw an exception immediately when this file is imported, refusing to start.
    OPENROUTER_API_KEY: str
    DATABASE_URL: str
    REDIS_URL: str
    
    # Optional parameters with default fallbacks
    LLM_MODEL: str = "openai/gpt-4o-mini"
    FALLBACK_LLM_MODEL: str = "anthropic/claude-3-haiku"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Retrieval (see app/services/retrieval.py and tests/retrieval_eval/)
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # empty string disables reranking
    USE_BM25: bool = True           # add keyword (BM25) candidates to the dense ones
    RETRIEVAL_CANDIDATES: int = 30  # candidates taken from each of dense and BM25 before reranking
    RETRIEVAL_TOP_K: int = 8        # most chunks passed to the LLM (a fixed count when ADAPTIVE_TOP_K is off)

    # Adaptive cutoff on reranker scores: pass only chunks close to the best one, none if the best is
    # irrelevant. The scores are on the reranker's own scale: these values were calibrated for
    # ms-marco-MiniLM-L-6-v2 with tests/retrieval_eval/run_cutoff_eval.py and must be re-run for another model.
    ADAPTIVE_TOP_K: bool = True
    RERANK_MIN_SCORE: float = -6.0  # best chunk below this -> "not found" (about 2 points below the lowest answerable question)
    RERANK_MARGIN: float = 6.0      # keep chunks scoring within this of the best one
    RETRIEVAL_MIN_K: int = 2        # keep at least this many once the best chunk passes RERANK_MIN_SCORE
    
    # lock absolute path
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()  # singleton
