# μStruct — Microstructure Identifier

An open-source tool for **microstructure image retrieval and phase identification** in metals. Upload a micrograph and get AI-powered phase identification, a comprehensive educational report, and visually similar matches from a multi-dataset library.

**[→ Try the live demo on Hugging Face Spaces](https://kevwill-microstructure-classifier.hf.space)**

---

## What It Does

**Identify** — Upload a micrograph and get a diagnostic report: the identified phase, confidence level, key visual features, optical/SEM appearance descriptions, formation conditions, engineering significance, and reference images from the library. Two AI methods (CLIP zero-shot classification and similarity-weighted voting) are combined in an ensemble for robust identification.

**Phase Guide** — A built-in encyclopedia covering 16 steel microstructural phases. Each entry includes descriptions, visual features for optical and SEM microscopy, typical hardness ranges, formation conditions, composition ranges, commonly confused phases, and engineering significance. Example images from the database are linked directly.

**Search Library** — Search through the indexed micrograph library by image similarity or text description. Filter by imaging modality (SEM/optical) and dataset source. Powered by CLIP embeddings and FAISS vector search.

## Supported Phases

| Phase | Hardness (HV) | Category |
|-------|---------------|----------|
| Ferrite (α-Iron) | 80–150 | Equilibrium Phase |
| Pearlite | 200–350 | Eutectoid Constituent |
| Ferrite + Pearlite | 120–250 | Common Two-Phase |
| Martensite | 400–700+ | Diffusionless Transformation |
| Tempered Martensite | 300–600 | Heat-Treated Phase |
| Bainite | 300–450 | Intermediate Transformation |
| Bainite with M/A Islands | 280–400 | Multi-Phase Constituent |
| Austenite (Retained) | 150–300 | High-Temp / Stabilized |
| Cementite (Fe₃C) | 800–1000 | Carbide Phase |
| Network Cementite | 800–1000 | Detrimental Carbide Morphology |
| Spheroidite | 150–250 | Annealed Microstructure |
| Widmanstätten Ferrite | 150–250 | Transformation Product |
| Acicular Ferrite | 200–280 | Transformation Product |
| Duplex / Dual-Phase | Varies | Engineered Multi-Phase |
| Pearlite + Widmanstätten | 180–300 | Mixed Microstructure |
| Pearlite + Spheroidite | 180–280 | Transitional Microstructure |

## How It Works

```
Upload Micrograph
        │
        ▼
┌─────────────────────────────────┐
│  CLIP ViT-B-32 Embedding        │
│  (512-dim feature vector)       │
└──────────┬──────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐ ┌──────────────────┐
│ Zero-Shot │ │ FAISS Similarity  │
│ Classify  │ │ Search + Voting   │
│ (40%)     │ │ (60%)             │
└─────┬────┘ └────────┬─────────┘
      │               │
      └───────┬───────┘
              ▼
     ┌────────────────┐
     │   Ensemble      │
     │   Prediction    │
     └───────┬────────┘
             ▼
     ┌────────────────┐
     │ Diagnostic      │
     │ Report +        │
     │ Phase Guide     │
     └────────────────┘
```

**CLIP Zero-Shot** — The uploaded image is compared against text descriptions of each phase (4 prompts per phase, averaged into prototypes). No labeled training data required — CLIP matches visual appearance to natural language descriptions.

**Similarity Voting** — The image embedding is searched against the FAISS index. The top-K nearest neighbors vote for their phase label, weighted by cosine similarity. A label normalization layer maps raw dataset labels to canonical phase keys at query time.

**Ensemble** — Both methods are combined (40% zero-shot, 60% voting) and the result is mapped to the Phase Knowledge Base for a complete educational report.

## Datasets & Attribution

This project is built on openly available research datasets. Proper credit to the researchers and institutions who created and shared this data:

### Aachen-Heerlen Annotated Steel Microstructure Dataset

- **Images:** ~1,700 SEM micrographs
- **Content:** Bainitic steel with martensite-austenite (M/A) island annotations
- **Source:** [Figshare Collection](https://figshare.com/collections/Aachen-Heerlen_Annotated_Steel_Microstructure_Dataset/5185004)
- **Citation:** Iren, D., Reichert, T., Bäcke, M., & Pernack, E. (2021). Aachen-Heerlen Annotated Steel Microstructure Dataset. *Scientific Data*, 8, 213. [doi:10.1038/s41597-021-00926-7](https://doi.org/10.1038/s41597-021-00926-7)

### UHCS — Ultrahigh Carbon Steel Micrograph Database

- **Images:** ~960 optical micrographs
- **Content:** Ultrahigh carbon steel microstructures with phase labels (pearlite, spheroidite, network cementite, martensite, Widmanstätten) and processing metadata via SQLite database
- **Source:** [NIST Materials Data Repository](https://materialsdata.nist.gov/handle/11256/940)
- **Citation:** DeCost, B. L., Lei, B., Francis, T., & Holm, E. A. (2017). UHCSDB: UltraHigh Carbon Steel Micrograph DataBase. *Integrating Materials and Manufacturing Innovation*, 6, 197–205. [doi:10.1007/s40192-017-0098-z](https://doi.org/10.1007/s40192-017-0098-z)

### Kaggle Steel Microstructure Collections

- **Images:** ~840 optical micrographs
- **Content:** Steel alloy micrographs organized by alloy type (CPJ alloys, HR alloys, P92 alloys)
- **Source:** [Kaggle](https://www.kaggle.com)

### DoITPoMS Micrograph Library

- **Images:** ~900 micrographs
- **Content:** Diverse metals, ceramics, and composites with detailed metadata including composition, processing history, imaging technique, and expert descriptions
- **Source:** [DoITPoMS, University of Cambridge](https://www.doitpoms.ac.uk/miclib/)
- **License:** CC BY-NC-SA 4.0 International. Copyright remains with individual contributors; per-image attribution is preserved in metadata.

### Ti-6Al-4V Optical Microstructure Dataset

- **Images:** 1,225 optical micrographs
- **Content:** Titanium alloy (Ti-6Al-4V) microstructures under varied heat treatments, labeled as lamellar, bi-modal/duplex, or martensitic
- **Source:** [GitHub (RPI)](https://github.com/ArunBaskaran/Image-Driven-Machine-Learning-Approach-for-Microstructure-Classification-and-Segmentation-Ti-6Al-4V)
- **License:** MIT
- **Citation:** Baskaran, A., Kane, G., & Biber, K. (2021). Image driven machine learning methods for microstructure recognition. *Computational Materials Science*, 190, 110281.

### MLography Dataset

- **Images:** 42 full-resolution optical micrographs + segmentation data
- **Content:** Uranium-chromium alloy (U-0.1wt%Cr) metallographic scans with expert-labeled inclusions and grain boundaries
- **Source:** [GitHub (NRCN Scientific Computing Lab)](https://github.com/Scientific-Computing-Lab/MLography)
- **License:** CC BY 4.0
- **Citation:** Moshkovitz, Y., et al. (2022). MLography: An Automated Metallography Dataset for Machine Learning. *Scientific Reports*.

### Metallographic Collection Dataset

- **Images:** ~480 real grain micrographs + ~320 texture boundary images + additional subsets
- **Content:** Multi-dataset bundle including real grain images, texture boundary micrographs, and spheroidite images from multiple research groups
- **Source:** [GitHub](https://github.com/inbalc2/Metallographic-Collection-Dataset)
- **License:** Various (open/research use)

### NASA Pretrained Microscopy Models — Benchmark Data

- **Images:** ~100 benchmark micrographs (Ni-superalloy + environmental barrier coatings)
- **Content:** SEM micrographs with segmentation labels
- **Source:** [GitHub (NASA)](https://github.com/nasa/pretrained-microscopy-models)
- **License:** MIT

## Project Structure

```
microstructure-classifier/
├── app/
│   ├── server.py              # FastAPI backend (identification engine, API, phase knowledge base)
│   └── index.html             # Web UI (Identify, Phase Guide, Search Library)
├── src/
│   ├── ingest.py              # Dataset loaders (Aachen, Kaggle, UHCS, generic)
│   ├── embed.py               # CLIP and ResNet50 embedding extractors
│   ├── index.py               # FAISS vector index wrapper with metadata
│   ├── classify.py            # Phase classification model (transfer learning)
│   └── utils.py               # Image preprocessing and visualization
├── scripts/
│   ├── build_index.py         # Embed all images and build FAISS index
│   ├── doctor.py              # Preflight check for dependencies, data, and index readiness
│   └── query.py               # CLI similarity search
├── docs/
│   └── reproducible_setup.md  # Repeatable setup, indexing, and deployment workflow
├── config.py                  # Paths, model settings, hyperparameters
├── Dockerfile                 # Container config for deployment
├── requirements.txt
└── tests/
```

## Quick Start

### Local Development

```bash
# 1. Clone and set up environment
git clone https://github.com/kevinjohnwilliams/microstructure-classifier.git
cd microstructure-classifier
python -m venv venv && source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate dependency compatibility before indexing
python scripts/doctor.py

# 4. Add datasets to data/raw/ (see Datasets & Attribution above)
mkdir -p data/raw/welding_electrodes
# copy your micrograph folders/images into data/raw/welding_electrodes/

# 5. Optional: verify ingestion without generating embeddings
python scripts/build_index.py --data-dir data/raw/welding_electrodes --dry-run

# 6. Build the embedding index
python scripts/build_index.py --model clip --include-excluded

# 7. Confirm the index is ready
python scripts/doctor.py

# 8. Launch the web server
uvicorn app.server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000). See [`docs/reproducible_setup.md`](docs/reproducible_setup.md) for the full repeatable indexing and Docker workflow.

### CLI Search

```bash
python scripts/query.py --image path/to/micrograph.png --top-k 5 --visualize
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/identify` | Upload image → full diagnostic report with phase ID |
| `POST` | `/api/search` | Upload image → similar microstructures from library |
| `POST` | `/api/search-text` | Text query → matching microstructures (CLIP) |
| `GET` | `/api/phases` | Phase knowledge base (all 16 phases) |
| `GET` | `/api/phase-guide` | Phase encyclopedia with example counts and images |
| `GET` | `/api/image/{idx}` | Thumbnail by index |
| `GET` | `/api/image/{idx}/full` | Full-resolution image |
| `GET` | `/api/stats` | Index statistics, label distribution, modalities |
| `GET` | `/api/health` | Health check |

## Tech Stack

- **Embeddings:** [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32) for image and text embeddings
- **Vector Search:** [FAISS](https://github.com/facebookresearch/faiss) for sub-millisecond nearest-neighbor retrieval
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) with async endpoints
- **Frontend:** Vanilla HTML/CSS/JS (single-page app, no build step)
- **Deployment:** Docker on [Hugging Face Spaces](https://huggingface.co/spaces)

## References

- Azimi, S. M., Britz, D., Schwarz, M., Steiner, M., Senk, D., & Mücklich, F. (2018). [Advanced Steel Microstructural Classification by Deep Learning Methods](https://doi.org/10.1038/s41598-018-20037-5). *Scientific Reports*, 8, 2128.
- DeCost, B. L., Lei, B., Francis, T., & Holm, E. A. (2017). [UHCSDB: UltraHigh Carbon Steel Micrograph DataBase](https://doi.org/10.1007/s40192-017-0098-z). *Integrating Materials and Manufacturing Innovation*, 6, 197–205.
- Iren, D., Reichert, T., Bäcke, M., & Pernack, E. (2021). [Aachen-Heerlen Annotated Steel Microstructure Dataset](https://doi.org/10.1038/s41597-021-00926-7). *Scientific Data*, 8, 213.
