# 🛍️ Visual Product Search Engine

A **multimodal visual product retrieval system** built on the **full DeepFashion dataset (289,222 images)** using **CLIP embeddings**, **FAISS similarity search**, **FastAPI**, and **Streamlit**.

This project supports:

- **Image-to-Image Search** → upload a product image and retrieve visually similar items
- **Text-to-Image Search** → describe a product in natural language and retrieve matching items
- **Category-Aware Retrieval** → optional coarse garment-class filtering
- **Production-Style Serving** → FastAPI backend + Streamlit frontend
- **Large-Scale Indexing** → built over the full DeepFashion Category & Attribute benchmark

---

## 🚀 Features

### Image Search
Upload a fashion image and retrieve the **top-k visually similar products**.

### Text Search
Use text queries like:
- `black tank top`
- `white blouse`
- `floral dress`
- `blue jeans`

The system embeds both **images and text into a shared CLIP embedding space**, enabling multimodal retrieval.

---

## 🧠 Tech Stack

- **Model:** OpenCLIP (ViT-B/32)
- **Vector Search:** FAISS (`IndexFlatIP`)
- **Similarity Metric:** Cosine similarity via **L2-normalized embeddings + inner product**
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Dataset:** DeepFashion (Category & Attribute Prediction Benchmark)
- **Language:** Python

---

## 📦 Project Architecture

```text
visual-product-search/
├── app/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── clip_model.py
│   │   ├── config.py
│   │   ├── faiss_index.py
│   │   ├── metrics.py
│   │   └── utils.py
│   └── schemas/
│       └── response.py
├── data/
│   └── README.md
├── embeddings/                # generated locally (gitignored)
├── indexes/                   # generated locally (gitignored)
├── scripts/
│   ├── generate_metadata.py
│   ├── validate_dataset.py
│   ├── build_index_subset.py
│   ├── build_index_full.py
│   ├── evaluate.py
│   └── benchmark.py
├── assets/
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── requirements.txt
└── README.md


---
📊 Scale & Performance
Full Local Build
Dataset size: 289,222 images
Embedding dimension: 512
Index type: FAISS IndexFlatIP
Indexing mode: full global index + coarse-label category indexes
Retrieval Quality (offline evaluation)
Queries evaluated: 200
Precision@5: 0.6240
Precision@10: 0.5985
Avg retrieval latency: 49.357 ms
P95 retrieval latency: 66.299 ms
End-to-End CPU Benchmark
Queries benchmarked: 100
Avg embedding latency: 125.628 ms
P95 embedding latency: 155.352 ms
Avg search latency: 48.939 ms
P95 search latency: 59.585 ms
Avg total latency: 174.566 ms
P95 total latency: 207.424 ms

Note: max latency outliers can occur due to CPU warmup, file I/O, and runtime initialization overhead.

🔍 API Endpoints
Health
GET /health
Image Search
POST /search-image

Form fields:

file → uploaded image
top_k → number of results
category (optional) → category filter
Text Search
POST /search-text

Form fields:

text → text query
top_k → number of results
category (optional) → category filter
Metrics
GET /metrics
🖼️ Example Screenshots

Add your screenshots to assets/ and keep these names:

api-docs.png
image-search-results.png
text-search-results.png
API Docs

Image Search Results

Text Search Results

🗂️ Dataset

This project uses the DeepFashion Category & Attribute Prediction Benchmark.

Recommended downloads:

Clothes Images (img.zip)
Category Annotations

Optional future extensions:

attributes
landmarks
bounding boxes

Large raw datasets and generated indexes are not included in this repository.

⚙️ Local Setup
1) Clone repo
git clone <your-repo-url>
cd visual-product-search
2) Create virtual environment
python -m venv venv
Windows (PowerShell)
venv\Scripts\activate
3) Install dependencies
pip install -r requirements.txt
4) Configure environment

Create a .env file based on .env.example.

Example:

APP_NAME=Visual Product Search Engine
DEVICE=cpu

RAW_DATASET_ROOT=F:/datasets/DeepFashion/CategoryAttribute

METADATA_PATH=data/metadata.csv
GLOBAL_INDEX_PATH=indexes/global_full.index
CATEGORY_INDEX_DIR=indexes/coarse_indexes_full

DEFAULT_TOP_K=5
MAX_SEARCH_K=10

CLIP_MODEL_NAME=ViT-B-32
CLIP_PRETRAINED=laion2b_s34b_b79k
🏗️ Build the Full Local Index
1) Generate metadata
$env:PYTHONPATH="."
python scripts/generate_metadata.py
2) Validate dataset
$env:PYTHONPATH="."
python scripts/validate_dataset.py
3) Build full embeddings + indexes
$env:PYTHONPATH="."
python scripts/build_index_full.py
📈 Evaluate Retrieval Quality
$env:PYTHONPATH="."
python scripts/evaluate.py
⚡ Benchmark End-to-End Performance
$env:PYTHONPATH="."
python scripts/benchmark.py
▶️ Run the App
Terminal 1 — FastAPI backend
$env:PYTHONPATH="."
uvicorn app.api.main:app --port 8002
Terminal 2 — Streamlit frontend
$env:PYTHONPATH="."
streamlit run streamlit_app.py
🐳 Docker
Build API image
docker build -t visual-product-search .
Run with Docker Compose
docker compose up --build

Note: the repository excludes large datasets, embeddings, and FAISS indexes.
Docker support is intended for reproducibility and lightweight subset deployments, not for shipping the full 289K local artifact bundle inside the container.

🧩 Design Decisions
Why CLIP?

CLIP places images and text in a shared embedding space, enabling:

image-to-image retrieval
text-to-image retrieval
multimodal product discovery
Why FAISS IndexFlatIP?

Embeddings are L2-normalized, so:

inner product == cosine similarity

This makes IndexFlatIP a correct and simple baseline for cosine-style nearest neighbor retrieval.

Why FastAPI + Streamlit?

This separates:

backend inference + retrieval
frontend interaction + visualization

That’s a stronger engineering pattern than a notebook-only demo.

📌 Future Improvements
Approximate nearest neighbor indexes (IVF, HNSW, PQ) for faster/larger deployment
Metadata-aware re-ranking
Attribute filtering (color / fabric / sleeve type)
Lightweight 25K / 50K hosted demo
Cloud deployment with persisted subset artifacts