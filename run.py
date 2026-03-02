import logging
import os

# Keep startup clean and force PyTorch-only path for transformers/sentence-transformers.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.web_api.web_app import app, recommender, search_pipeline
from src.core.text_embeddings import embedding_handler


def warmup() -> None:
    """
    Execute warmup.
    
    This function implements the warmup step for this module.
    It is used to keep the broader workflow readable and easier to maintain.
    """
    print("[startup] Initializing search pipeline...", flush=True)
    _ = search_pipeline.instance
    print("[startup] Search pipeline loaded.", flush=True)

    print("[startup] Loading embedding model...", flush=True)
    _ = embedding_handler.load_model()
    _ = embedding_handler.encode(["eraex warmup"], batch_size=1, show_progress_bar=False)
    print("[startup] Embedding model ready.", flush=True)

    print("[startup] Initializing recommender...", flush=True)
    _ = recommender.instance
    print("[startup] Recommender loaded.", flush=True)

    print("[startup] EraEx ready to use at http://127.0.0.1:5000", flush=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
    )
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    warmup()
    app.run(debug=False, use_reloader=False, port=5000, host="0.0.0.0")
