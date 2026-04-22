# Legal PDF Processing and Semantic Search

This project extracts legal articles from PDF files, enriches them, creates embeddings, and exposes both CLI and web search.

## Project Structure

- `src/app.py`: PDF extraction to structured article JSON/CSV
- `src/add_ids.py`: add stable sequential IDs
- `src/semantic_chunk.py`: semantic chunking pipeline (supports resume/checkpoints)
- `src/embed.py`: embed full datasets
- `src/embed_missing.py`: only embed records missing vectors
- `src/recherche.py`: CLI semantic search
- `src/api.py`: web API + browser frontend
- `output/`: generated artifacts (`extracted`, `with_ids`, `chunks`, `embeddings`, `questions`)
- `pdfs/`: input PDF files

## Setup

Run from project root.

```bash
pip install pandas pymupdf requests google-genai numpy
```

### Environment Variables

Gemini-backed scripts require one of:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

Ollama question generation can use:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:latest`)

Recommended `.env` example:

```env
GOOGLE_API_KEY=your_key_here
```

## Typical Workflow

### 1) Extract Articles

All PDFs in `pdfs/`:

```bash
python src/app.py
```

Single PDF:

```bash
python src/app.py --pdf "pdfs/codedecommerce.pdf"
```

Outputs go to `output/extracted/` by default.

### 2) Add IDs

Directory mode:

```bash
python src/add_ids.py output/extracted --output-dir output/with_ids
```

Single file mode:

```bash
python src/add_ids.py output/extracted/codedecommerce_articles.json --output-dir output/with_ids
```

### 3) Semantic Chunking (Optional)

```bash
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json
```

Resume from checkpoint:

```bash
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json --resume
```

### 4) Generate Embeddings

Full embedding run:

```bash
python src/embed.py output/extracted/codedecommerce_articles.json
```

Only missing embeddings:

```bash
python src/embed_missing.py output/embeddings/codedecommerce_articles_embedded.json
```

### 5) Search from CLI

Single embedded file:

```bash
python src/recherche.py output/embeddings/codedecommerce_articles_embedded.json -q "obligation d'ouvrir un compte"
```

All embedded files in folder:

```bash
python src/recherche.py output/embeddings -q "obligation d'ouvrir un compte"
```

## Web API and Frontend

Run server:

```bash
python src/api.py
```

Open browser UI:

```text
http://localhost:8000/
```

The frontend uses the API key loaded on the server from `.env`.

### API Endpoints

- `GET /`: browser search page
- `GET /health`: service and embeddings status
- `GET /search?query=...&top_k=5&threshold=0.2`: quick search
- `POST /search`: JSON search body

Sample POST:

```json
{
  "query": "obligation d'ouvrir un compte",
  "top_k": 5,
  "threshold": 0.2
}
```

### API CLI Options

```bash
python src/api.py --help
```

Key options:

- `--host` (default from env or `0.0.0.0`)
- `--port` (default from env or `8000`)
- `--embeddings-dir` (default from env or `output/embeddings`)

## Script Help

Use built-in help for exact options/defaults:

```bash
python src/app.py --help
python src/add_ids.py --help
python src/semantic_chunk.py --help
python src/embed.py --help
python src/embed_missing.py --help
python src/recherche.py --help
python src/api.py --help
```

## Troubleshooting

- `Google API key required`: verify `.env` is in project root and has `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
- `Embeddings source not found`: ensure `output/embeddings` exists or set `--embeddings-dir`.
- Rate-limit/quota errors (`429 RESOURCE_EXHAUSTED`): reduce load, then retry.
- PowerShell `curl` warnings: use `curl.exe` instead of `curl` alias.
