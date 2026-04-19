# PDF Legal Pipeline

This repository extracts legal articles from PDFs and supports downstream processing:

1. Extraction (`src/app.py`)
2. ID assignment (`src/add_ids.py`)
3. Question generation (`src/generate_questions.py`)
4. Semantic chunking (`src/semantic_chunk.py`)
5. Embedding (`src/embed.py`, `src/embed_missing.py`)
6. Semantic search (`src/recherche.py`)

All commands below assume you run them from the project root.

## Python and Dependencies

Current project runtime in this workspace: Python 3.14.0.

Install dependencies:

```bash
pip install pandas pymupdf requests google-genai numpy
```

## Environment Variables

Gemini-based scripts require one of:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

Ollama-based question generation can use:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:latest`)

Where variables are loaded from:

- `src/semantic_chunk.py`, `src/embed.py`, `src/embed_missing.py`, `src/recherche.py` load `.env` from current working directory.
- `src/generate_questions.py` loads `src/.env` (and also reads already-exported environment variables).

## Defaults (No Ambiguity)

### `src/app.py`

- Default `--pdf-dir`: `pdfs`
- Default `--output-dir`: `output/extracted`
- Default `--max-pages`: `0` (all pages)

### `src/semantic_chunk.py`

- Input can be extracted JSON or PDF.
- If input is PDF, extraction is run first.
- Default `--max-pages`: `0` (all pages)
- Default checkpoint frequency: every `50` articles

## Typical Workflow

### 1) Extract articles from PDFs

All PDFs in `pdfs/`:

```bash
python src/app.py
```

Single PDF:

```bash
python src/app.py --pdf "pdfs/codedecommerce.pdf"
```

First N pages only:

```bash
python src/app.py --pdf "pdfs/codedecommerce.pdf" --max-pages 50
```

Outputs per PDF:

- `output/extracted/<pdf_stem>_articles.json`
- `output/extracted/<pdf_stem>_articles.csv`

Extraction columns:

1. `document_name`
2. `article_number`
3. `livre`
4. `titre`
5. `chapitre`
6. `section`
7. `sous_section`
8. `pages`
9. `content`
10. `source_relative_path`

### 2) Add IDs

Directory input (`*_articles.json` files):

```bash
python src/add_ids.py output/extracted --output-dir output/with_ids
```

Single file:

```bash
python src/add_ids.py output/extracted/codedecommerce_articles.json --output-dir output/with_ids
```

Output names:

- `*_with_ids.json`
- `*_with_ids.csv`

### 3) Semantic chunking

From extracted JSON:

```bash
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json
```

Directly from PDF (extract + chunk in one command):

```bash
python src/semantic_chunk.py "pdfs/codedecommerce.pdf"
```

Force first 50 pages only:

```bash
python src/semantic_chunk.py "pdfs/codedecommerce.pdf" --max-pages 50
```

Resume from latest checkpoint:

```bash
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json --resume
```

Clean existing semantic chunks JSON and rewrite JSON+CSV:

```bash
python src/semantic_chunk.py output/chunks/codedecommerce_articles_semantic_chunks.json --clean
```

Default chunk outputs:

- `output/chunks/<input_stem>_semantic_chunks.json`
- `output/chunks/<input_stem>_semantic_chunks.csv`
- `output/chunks/checkpoints/<input_stem>/...`

Chunk columns:

- `chunk_id`, `chunk_index`, `chunk_count`
- `article_number`, `document_name`
- `livre`, `titre`, `chapitre`, `section`, `sous_section`
- `pages`, `source_relative_path`, `content`

### 4) Generate questions (Ollama)

`generate_questions.py` expects one JSON file per run.

```bash
python src/generate_questions.py output/with_ids/codedecommerce_articles_with_ids.json --output-dir output/questions
```

Outputs:

- `output/questions/<input_stem>_questions.json`
- `output/questions/<input_stem>_questions.csv`
- batch files in `output/questions/<input_stem>_batches/`

### 5) Build embeddings

Embed all records in a JSON file:

```bash
python src/embed.py output/extracted/codedecommerce_articles.json
```

Enable periodic checkpoints (every N processed records):

```bash
python src/embed.py output/extracted/codedecommerce_articles.json --checkpoint-every 50
```

By default, checkpoint snapshots are written to:

- `output/embeddings/checkpoints/<input_stem>/`

You can override this with `--checkpoint-dir`.

Default output (if `-o` not set):

- `output/embeddings/<input_stem>_embedded.json`

Fill only missing embeddings:

```bash
python src/embed_missing.py output/embeddings/codedecommerce_articles_embedded.json
```

`embed_missing.py` also writes periodic checkpoint snapshots (default every 25 missing records):

- Default checkpoint path: `<output_parent>/checkpoints/<input_stem>/`
- Override with `--checkpoint-dir`

`embed_missing.py` overwrites input by default when `-o` is not provided.

### 6) Semantic search

```bash
python src/recherche.py output/embeddings/codedecommerce_articles_embedded.json -q "obligation d'ouvrir un compte"
```

## Troubleshooting

### `semantic_chunk.py` exits with code 1 after starting

Most common cause is Gemini quota/rate limit (for example `429 RESOURCE_EXHAUSTED`).

What to do:

1. Check API quota/billing for the key.
2. Retry with reduced load (smaller file or lower page count).
3. Keep checkpoints enabled and resume with `--resume`.

### Missing API key errors

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY`, or pass `--api-key`.

### Extraction finds no PDFs

Verify `pdfs/` exists or pass explicit `--pdf` / `--pdf-dir`.
