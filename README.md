# Book Recommender System

End-to-end recommendation system that combines data engineering, collaborative filtering, graph-based recommendation (LightGCN), API serving, and a frontend UI.

## Overview

This project is built as a full ML pipeline:

1. Data ingestion
2. Data validation
3. Data transformation
4. Model training
5. Deployment via FastAPI and Docker

It uses two main catalog sources:

- Amazon Books dataset from Kaggle (`mohamedbakhet/amazon-books-reviews`)
- OpenLibrary scraped metadata (current dataset: `8516` books, `8516` unique ISBNs)

These are merged into a unified dataset before training.

## Pipeline

### 1) Data Ingestion

Code:

- `src/steps/stage_00_data_ingestion/ingest_amazonbooks.py`
- `src/steps/stage_00_data_ingestion/ingest_openlibrary.py`
- `src/steps/stage_00_data_ingestion/merge_sources.py`
- `src/steps/stage_00_data_ingestion/Ingest_step.py`

What it does:

- Downloads Amazon books + reviews via `kagglehub`
- Reads OpenLibrary metadata CSV (or scrapes via ingestion module)
- Merges books and ratings, plus support ratings
- Writes:
  - `artifacts/dataset/ingested_data/current_books.csv`
  - `artifacts/dataset/ingested_data/current_reviews.csv`

### 2) Data Validation

Code:

- `src/steps/stage_01_data_validation/validate_step.py`

What it does:

- Cleans nulls and malformed fields
- Deduplicates books by ISBN
- Normalizes publication year
- Filters weak descriptions / unwanted content patterns
- Writes cleaned data:
  - `artifacts/dataset/clean_data/current_books_cleaned.csv`
  - `artifacts/dataset/clean_data/current_reviews_cleaned.csv`

### 3) Data Transformation

Code:

- `src/steps/stage_02_data_transformation/transform_step.py`

What it does:

- Builds popularity list
- Builds user-item pivot table
- Creates train/test interaction files
- Builds vector DB (Chroma + sentence embeddings)
- Writes artifacts such as:
  - `artifacts/dataset/transformed_data/most_popular_books.pkl`
  - `artifacts/dataset/transformed_data/piovt_table_data.pkl`
  - `artifacts/dataset/transformed_data/train_data.csv`
  - `artifacts/dataset/transformed_data/test_data.csv`
  - `artifacts/serialized_objects/item_similarity.pkl`
  - `artifacts/serialized_objects/piovt_table_data.pkl`
  - `artifacts/vectorstores/chroma_db`

### 4) Model Training

Code:

- `src/steps/stage_03_model_trainer/train_step.py`

Models:

- Popularity baseline
- LightGCN (Microsoft Recommenders implementation)

Tracking:

- MLflow logs params and metrics

### 5) Deployment

Code:

- API: `api/main.py`
- Frontend: `frontend/app.html`, `frontend/app.js`, `frontend/app.css`
- Container: `Dockerfile`

Deployment stack:

- FastAPI + Uvicorn
- Dockerized runtime

## Reported Metrics

Top-K = 10

Final filtered test set: `333,298` user-item pairs

### Popularity Baseline

- Precision@10: `0.1893`
- Recall@10: `0.2961`
- nDCG@10: `0.2505`
- MAP@10: `0.1261`

### LightGCN

- Precision@10: `0.4276`
- Recall@10: `0.6261`
- nDCG@10: `0.6769`
- MAP@10: `0.5493`

### Data Scale Snapshot

- Train interactions: `1,230,060`
- Test interactions: `345,228`
- Train unique users/items: `51,366 / 3,954`
- Test unique users/items: `63,261 / 2,939`
- Train sparsity: `99.39%`
- Avg interactions per user: `23.95`
- Avg interactions per item: `311.09`

## Project Structure

```text
.
|- api/
|  |- main.py
|  |- database.py
|  |- models.py
|  `- vectordb.py
|- config/
|  `- config.yaml
|- frontend/
|  |- app.html
|  |- app.css
|  `- app.js
|- src/
|  |- config/
|  |- pipeline/
|  |- steps/
|  |  |- stage_00_data_ingestion/
|  |  |- stage_01_data_validation/
|  |  |- stage_02_data_transformation/
|  |  `- stage_03_model_trainer/
|  `- logger/
|- tests/
|- Dockerfile
|- main.py
`- requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+ (recommended to match CI and Docker)
- pip
- (Optional) Docker
- Kaggle access/token for Amazon dataset download

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) (If needed) Install Microsoft Recommenders package

If your local repo does not already contain/install it:

```bash
git clone https://github.com/microsoft/recommenders.git src/steps/stage_03_model_trainer/recommenders_microsoft
pip install -e src/steps/stage_03_model_trainer/recommenders_microsoft
```

### 3) Configure project settings

Edit:

- `config/config.yaml`

## Run

### Run full training pipeline

```bash
python main.py
```

This executes:

- Data ingestion -> validation -> transformation -> model training

### Analyze sparsity

```bash
python analyze_data_sparsity.py
```

### Start App (API + Frontend UI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

This command starts the full application (backend API and frontend UI).

Then open in your browser:

- `http://127.0.0.1:8000/`

## Docker

Build:

```bash
docker build -t book-recommender-system .
```

Run:

```bash
docker run -p 8000:8000 book-recommender-system
```

## API Endpoints

- `GET /` -> serves frontend
- `GET /api/search?q=<query>` -> semantic/vector search
- `GET /api/books/popular` -> top popular books
- `GET /api/user/recommendations` -> LightGCN user recommendations
- `GET /api/books/{book_id}` -> book detail
- `GET /api/books/{book_id}/related` -> item-item related books

## Testing and CI/CD

- Tests: `tests/test_app.py`
- CI workflow: `.github/workflows/CICD.yml`
- CI runs:
  - dependency install
  - pipeline execution
  - pytest
  - Docker build and push

## Notes

- Artifacts and logs are written under `artifacts/`, `logs/`, and `mlruns/`.
- Frontend static assets are versioned through query strings to reduce stale-cache UI issues.

## License

Add your preferred license in this repository (e.g., MIT) if you plan to open-source it publicly.
