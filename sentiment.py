"""
Shared FinBERT backbone: map a list of text chunks (e.g. 30-day headlines) to a
fixed-size sentiment feature vector. Used by bandit env and training.
"""
import numpy as np
from typing import List

# Lazy load to avoid importing torch/transformers at module level if not used
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        from config import FINBERT_MODEL_NAME
        _pipeline = pipeline("sentiment-analysis", model=FINBERT_MODEL_NAME, top_k=None)
    return _pipeline


def texts_to_sentiment_vector(texts: List[str], max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    """
    Run FinBERT on each text and aggregate to a fixed-size vector.
    Returns shape (3,) or (3 * 2,) for [mean_pos, mean_neg, mean_neutral] or with std.
    Labels from FinBERT are typically 'positive', 'negative', 'neutral' with scores.
    """
    if not texts:
        return np.zeros(3, dtype=np.float32)  # no text -> neutral

    pipe = _get_pipeline()
    # Truncate long texts; run in batches
    scores_list = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:max_length] if isinstance(t, str) else str(t)[:max_length] for t in texts[i : i + batch_size]]
        try:
            out = pipe(batch)
            # out is list of list of dicts: [{"label": "positive", "score": 0.8}, ...]
            for item in out:
                if isinstance(item, list):
                    d = {x["label"].lower(): x["score"] for x in item}
                else:
                    d = {item["label"].lower(): item["score"]}
                pos = d.get("positive", 1.0 / 3)
                neg = d.get("negative", 1.0 / 3)
                neu = d.get("neutral", 1.0 / 3)
                scores_list.append([pos, neg, neu])
        except Exception:
            scores_list.append([1.0 / 3, 1.0 / 3, 1.0 / 3])

    arr = np.array(scores_list, dtype=np.float32)
    mean = np.mean(arr, axis=0)
    # Return 3-dim (mean) so context size is predictable; optionally append std for 6-dim
    return mean
