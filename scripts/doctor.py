#!/usr/bin/env python3
"""Preflight checks for local deployment and index readiness."""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.utils import collect_image_paths


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _status(flag: bool) -> str:
    return "OK" if flag else "MISSING"


def main() -> int:
    print("μStruct deployment doctor")
    print("=" * 72)

    print("\nConfiguration")
    print(f"  project root:       {config.PROJECT_ROOT}")
    print(f"  raw data dir:       {config.RAW_DIR}")
    print(f"  embeddings dir:     {config.EMBEDDINGS_DIR}")
    print(f"  index path:         {config.FAISS_INDEX_PATH}")
    print(f"  metadata path:      {config.METADATA_PATH}")
    print(f"  manifest path:      {config.INDEX_MANIFEST_PATH}")
    print(f"  server model:       {config.EMBEDDING_MODEL}")
    print(f"  similarity metric:  {config.SIMILARITY_METRIC}")

    print("\nPython dependencies")
    dependency_names = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("open_clip", "open-clip-torch"),
        ("faiss", "faiss-cpu"),
        ("fastapi", "fastapi"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
    ]
    dependency_failures = 0
    for import_name, package_name in dependency_names:
        ok, detail = _check_import(import_name)
        dependency_failures += 0 if ok else 1
        print(f"  {_status(ok):8} {package_name:18} {detail}")

    print("\nData and index")
    raw_images = collect_image_paths(config.RAW_DIR) if config.RAW_DIR.exists() else []
    has_index = config.FAISS_INDEX_PATH.exists()
    has_metadata = config.METADATA_PATH.exists()
    has_manifest = config.INDEX_MANIFEST_PATH.exists()
    print(f"  {_status(config.RAW_DIR.exists()):8} raw data directory ({len(raw_images)} images found)")
    print(f"  {_status(has_index):8} FAISS index")
    print(f"  {_status(has_metadata):8} metadata pickle")
    print(f"  {_status(has_manifest):8} index manifest")

    manifest_model = None
    if has_manifest:
        try:
            manifest = json.loads(config.INDEX_MANIFEST_PATH.read_text())
            manifest_model = manifest.get("model")
            print("\nIndex manifest")
            print(f"  created at:         {manifest.get('created_at', 'unknown')}")
            print(f"  indexed model:      {manifest_model}")
            print(f"  vectors:            {manifest.get('count', 'unknown')}")
            print(f"  embedding dim:      {manifest.get('embedding_dim', 'unknown')}")
            print(f"  sources:            {manifest.get('sources', {})}")
        except Exception as exc:
            print(f"  Manifest could not be read: {type(exc).__name__}: {exc}")

    if manifest_model and manifest_model != config.EMBEDDING_MODEL:
        print(
            "\nWARNING: index was built with model "
            f"'{manifest_model}', but the server is configured for "
            f"'{config.EMBEDDING_MODEL}'. Set MICROSTRUCTURE_EMBEDDING_MODEL={manifest_model} "
            "or rebuild the index."
        )

    print("\nNext commands")
    if not raw_images:
        print("  1. Put micrograph images under data/raw/<dataset-or-class-name>/")
        print("     or use: python scripts/build_index.py --data-dir /path/to/images --dry-run")
    if not (has_index and has_metadata):
        print("  2. Build embeddings/index:")
        print("     python scripts/build_index.py --model clip --include-excluded")
    print("  3. Start app:")
    print("     uvicorn app.server:app --host 0.0.0.0 --port 8000")
    print("  4. Check health:")
    print("     curl http://localhost:8000/api/health")

    if dependency_failures:
        return 2
    if not (has_index and has_metadata):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
