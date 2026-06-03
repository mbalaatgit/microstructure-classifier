"""
Central configuration for the microstructure classifier.
Adjust paths and model settings here, or override them with environment variables.
"""
import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return a Path from an environment variable, falling back to a default."""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = _env_path("MICROSTRUCTURE_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = _env_path("MICROSTRUCTURE_RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = _env_path("MICROSTRUCTURE_PROCESSED_DIR", DATA_DIR / "processed")
EMBEDDINGS_DIR = _env_path("MICROSTRUCTURE_EMBEDDINGS_DIR", DATA_DIR / "embeddings")

# Dataset-specific paths
UHCS_DIR = _env_path("MICROSTRUCTURE_UHCS_DIR", RAW_DIR / "uhcs")
UHCS_MICROGRAPHS = _env_path("MICROSTRUCTURE_UHCS_MICROGRAPHS", UHCS_DIR / "micrographs")
UHCS_SQLITE = _env_path("MICROSTRUCTURE_UHCS_SQLITE", UHCS_DIR / "microstructures.sqlite")

AACHEN_DIR = _env_path("MICROSTRUCTURE_AACHEN_DIR", RAW_DIR / "aachen")

# Index files
FAISS_INDEX_PATH = _env_path(
    "MICROSTRUCTURE_FAISS_INDEX_PATH", EMBEDDINGS_DIR / "microstructure.index"
)
METADATA_PATH = _env_path("MICROSTRUCTURE_METADATA_PATH", EMBEDDINGS_DIR / "metadata.pkl")
INDEX_MANIFEST_PATH = _env_path(
    "MICROSTRUCTURE_INDEX_MANIFEST_PATH", EMBEDDINGS_DIR / "index_manifest.json"
)

# ── Model Settings ───────────────────────────────────────────────────────────
# Options: "resnet50", "clip"
EMBEDDING_MODEL = os.getenv("MICROSTRUCTURE_EMBEDDING_MODEL", "clip")

# ResNet settings
RESNET_WEIGHTS = os.getenv("MICROSTRUCTURE_RESNET_WEIGHTS", "IMAGENET1K_V2")
EMBEDDING_DIM = int(os.getenv("MICROSTRUCTURE_EMBEDDING_DIM", "2048"))

# CLIP settings
CLIP_MODEL_NAME = os.getenv("MICROSTRUCTURE_CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("MICROSTRUCTURE_CLIP_PRETRAINED", "openai")

# ── Image Preprocessing ─────────────────────────────────────────────────────
IMAGE_SIZE = int(os.getenv("MICROSTRUCTURE_IMAGE_SIZE", "224"))
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ── Retrieval Settings ───────────────────────────────────────────────────────
DEFAULT_TOP_K = int(os.getenv("MICROSTRUCTURE_DEFAULT_TOP_K", "5"))
SIMILARITY_METRIC = os.getenv("MICROSTRUCTURE_SIMILARITY_METRIC", "cosine")  # "L2" or "cosine"

# ── Phase Classification (Phase 2) ──────────────────────────────────────────
PHASE_CLASSES = [
    "ferrite",
    "pearlite",
    "martensite",
    "bainite",
    "austenite",
    "cementite",
]
NUM_CLASSES = len(PHASE_CLASSES)

# ── Ensure directories exist ─────────────────────────────────────────────────
for d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, UHCS_DIR, AACHEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)
