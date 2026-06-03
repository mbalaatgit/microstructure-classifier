#!/usr/bin/env python3
"""
Build the FAISS similarity index from all available microstructure datasets.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --model clip
    python scripts/build_index.py --data-dir /path/to/custom/images
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingest import load_all_datasets, load_generic_directory
from src.embed import get_embedder
from src.index import MicrostructureIndex


def main():
    parser = argparse.ArgumentParser(description="Build microstructure search index")
    parser.add_argument(
        "--model",
        choices=["resnet50", "clip"],
        default=config.EMBEDDING_MODEL,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Custom directory of images (overrides default dataset loading)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include excluded/low-quality images (more data for similarity search)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize records without embedding images or writing an index",
    )
    args = parser.parse_args()
    
    # ── Load data ────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading datasets")
    print("=" * 60)
    
    if args.data_dir:
        records = load_generic_directory(args.data_dir)
    else:
        records = load_all_datasets(include_excluded=args.include_excluded)
    
    if not records:
        print("\nERROR: No images found!")
        print("Please download datasets first. See README.md for instructions.")
        print(f"Expected data in: {config.RAW_DIR}")
        print("Tip: for a quick proof-of-life build, pass --data-dir /path/to/folder/of/images")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete: records loaded successfully; no embeddings/index written.")
        return
    
    # ── Generate embeddings ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Generating embeddings ({args.model})")
    print("=" * 60)
    
    embedder = get_embedder(args.model)
    image_paths = [r["path"] for r in records]
    
    embeddings = embedder.embed_batch(image_paths, batch_size=args.batch_size)
    print(f"Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    
    # ── Build index ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Building FAISS index")
    print("=" * 60)
    
    index = MicrostructureIndex(embedding_dim=embeddings.shape[1])
    
    # Prepare metadata (strip non-serializable items)
    metadata_list = []
    for r in records:
        meta = {
            "path": str(r["path"]),
            "filename": r["filename"],
            "source": r.get("source", "unknown"),
            "label": r.get("label", ""),
        }
        # Include any extra metadata from the dataset
        if "metadata" in r and isinstance(r["metadata"], dict):
            for k, v in r["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        metadata_list.append(meta)
    
    index.add(embeddings, metadata_list)
    
    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: Saving index")
    print("=" * 60)
    
    index.save()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "metric": config.SIMILARITY_METRIC,
        "count": int(index.size),
        "index_path": str(config.FAISS_INDEX_PATH),
        "metadata_path": str(config.METADATA_PATH),
        "data_dir": str(args.data_dir) if args.data_dir else str(config.RAW_DIR),
        "include_excluded": bool(args.include_excluded),
        "sources": {},
        "labels": {},
    }
    for meta in metadata_list:
        source = meta.get("source", "unknown")
        label = meta.get("label", "unknown")
        manifest["sources"][source] = manifest["sources"].get(source, 0) + 1
        manifest["labels"][label] = manifest["labels"].get(label, 0) + 1

    config.INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INDEX_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved index manifest to {config.INDEX_MANIFEST_PATH}")
    
    print(f"\nDone! Index contains {index.size} microstructures.")
    print(f"Query with: python scripts/query.py --image <path>")


if __name__ == "__main__":
    main()
