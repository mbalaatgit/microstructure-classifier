# Reproducible local setup, indexing, and deployment

This project does **not** commit datasets, embeddings, FAISS indexes, or model weights.
A working deployment therefore has two phases:

1. install the runtime and place image data under `data/raw/`, and
2. create the embedding index in `data/embeddings/` before starting the API.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the preflight checker at any point:

```bash
python scripts/doctor.py
```

The doctor reports dependency status, configured paths, raw image counts, whether
`microstructure.index` and `metadata.pkl` exist, and the exact next commands to run.

## 2. Add images

For the built-in loaders, place data under:

```text
data/raw/
├── aachen/      # Aachen-Heerlen images, if available
├── kaggle/      # class/folder-style steel microstructure images
├── uhcs/        # UHCS micrographs + optional SQLite metadata
└── my_dataset/  # any additional folder of images
```

Any extra subdirectory under `data/raw/` is ingested with the generic loader.
For a welding-electrodes QA library, a simple layout like this is enough to start:

```text
data/raw/welding_electrodes/
├── ferrite_pearlite/
├── martensite/
├── bainite/
└── unknown_or_review/
```

The generic loader uses the parent folder name as the raw label, so meaningful
folder names improve the similarity-voting part of `/api/identify`.

To validate the loader without spending time embedding images:

```bash
python scripts/build_index.py --data-dir data/raw/welding_electrodes --dry-run
```

## 3. Build embeddings and the FAISS index

For the default CLIP/OpenCLIP workflow:

```bash
python scripts/build_index.py --model clip --include-excluded
```

For one custom folder:

```bash
python scripts/build_index.py --model clip --data-dir data/raw/welding_electrodes
```

This writes:

```text
data/embeddings/microstructure.index   # FAISS vectors
data/embeddings/metadata.pkl           # image metadata parallel to FAISS ids
data/embeddings/index_manifest.json    # reproducibility metadata
```

The server model must match the model used to build the index. If you built with
CLIP, start with the default `MICROSTRUCTURE_EMBEDDING_MODEL=clip`. If you build
with ResNet50, set `MICROSTRUCTURE_EMBEDDING_MODEL=resnet50` when running the
server.

## 4. Start and verify the app

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Then check:

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/stats
```

The app is ready for uploads when `/api/health` returns `index_loaded: true` and
`index_size` is greater than zero.

## 5. Docker workflow

Build the image:

```bash
docker build -t microstructure-classifier .
```

Run with local data mounted into the container:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -e MICROSTRUCTURE_EMBEDDING_MODEL=clip \
  microstructure-classifier
```

If `data/embeddings/` does not already contain an index, build it first on the
host with `scripts/build_index.py` or run the build command inside a container
with the same volume mounted.

## Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MICROSTRUCTURE_DATA_DIR` | `<repo>/data` | Root for raw data and generated embeddings |
| `MICROSTRUCTURE_RAW_DIR` | `$MICROSTRUCTURE_DATA_DIR/raw` | Raw image datasets |
| `MICROSTRUCTURE_EMBEDDINGS_DIR` | `$MICROSTRUCTURE_DATA_DIR/embeddings` | Generated FAISS files |
| `MICROSTRUCTURE_FAISS_INDEX_PATH` | `$MICROSTRUCTURE_EMBEDDINGS_DIR/microstructure.index` | FAISS index path |
| `MICROSTRUCTURE_METADATA_PATH` | `$MICROSTRUCTURE_EMBEDDINGS_DIR/metadata.pkl` | Metadata pickle path |
| `MICROSTRUCTURE_EMBEDDING_MODEL` | `clip` | Runtime embedder: `clip` or `resnet50` |
| `MICROSTRUCTURE_CLIP_MODEL_NAME` | `ViT-B-32` | OpenCLIP architecture |
| `MICROSTRUCTURE_CLIP_PRETRAINED` | `openai` | OpenCLIP pretrained weights tag |
