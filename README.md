# Legal Semantic Search (Python API + Next.js Frontend)

This repository processes legal PDFs into structured records, generates embeddings, and provides semantic search through a Python API.

The frontend now lives in [frontend](frontend) (Next.js). The Python API no longer serves an HTML page.

## Project Layout

- [src/app.py](src/app.py): extract legal articles from PDFs into JSON/CSV
- [src/add_ids.py](src/add_ids.py): add stable IDs
- [src/semantic_chunk.py](src/semantic_chunk.py): semantic chunking pipeline
- [src/embed.py](src/embed.py): generate embeddings
- [src/embed_missing.py](src/embed_missing.py): backfill missing embeddings only
- [src/recherche.py](src/recherche.py): CLI semantic search
- [src/api.py](src/api.py): backend search API
- [frontend](frontend): Next.js frontend
- [output](output): generated artifacts
- [pdfs](pdfs): source PDF files

## Requirements

- Python 3.10+
- Node.js 20+ (for frontend)

Install Python dependencies from project root:

```bash
pip install pandas pymupdf requests google-genai numpy
```

## Environment Variables

Create [\.env](.env) in the repository root:

```env
GOOGLE_API_KEY=your_key_here
```

Accepted API key names:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

Frontend API base URL template:

- [frontend/.env.local.example](frontend/.env.local.example)

## Data Pipeline

1. Extract from all PDFs:

```bash
python src/app.py
```

2. Add IDs:

```bash
python src/add_ids.py output/extracted --output-dir output/with_ids
```

3. (Optional) semantic chunking:

```bash
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json
```

4. Generate embeddings:

```bash
python src/embed.py output/extracted/codedecommerce_articles.json
```

5. Fill missing embeddings only:

```bash
python src/embed_missing.py output/embeddings/codedecommerce_articles_embedded.json
```

## Run Backend API

From repository root:

```bash
python src/api.py
```

API endpoints:

- `GET /` returns JSON service info (no built-in UI)
- `GET /health` service + embeddings status
- `GET /search?...` query via URL
- `POST /search` query via JSON body

### Example Search Request

```json
{
  "query": "obligation d'ouvrir un compte",
  "top_k": 5,
  "threshold": 0.2,
  "close_filter": "balanced",
  "verify_results": false
}
```

`close_filter` values:

- `off`
- `loose`
- `balanced`
- `strict`

Effective filter rule:

$$
effective\_threshold = \max(threshold, close\_filter\_threshold)
$$

Notes:

- Search scans local embedded vectors; this is CPU work, not one API call per article.
- Query embedding uses one Gemini embedding request per search.
- Optional AI verification (`verify_results=true`) uses Gemini `generate_content` and may hit quota.
- If verification fails (quota/rate limit), API still returns similarity-ranked results.

## Run Frontend

1. In one terminal (root):

```bash
python src/api.py
```

2. In another terminal:

```bash
cd frontend
npm install
npm run dev
```

3. Open:

```text
http://localhost:3000
```

## CLI Search

Search all embeddings in folder:

```bash
python src/recherche.py output/embeddings -q "obligation d'ouvrir un compte"
```

## Troubleshooting

- `Google API key required`: verify [\.env](.env) has a valid key.
- `Embeddings source not found`: check [output/embeddings](output/embeddings) or pass `--embeddings-dir`.
- `429 RESOURCE_EXHAUSTED`: disable `verify_results` or retry later.
- PowerShell `curl` alias issues: use `curl.exe`.
- API returns relative source file paths by design.
