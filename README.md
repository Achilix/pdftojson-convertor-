# PDF Articles Pipeline

This project processes legal PDF corpora end-to-end:

1. Extract articles and hierarchy metadata from PDF files.
2. Add stable IDs.
3. Generate study questions.
4. Build semantic chunks.
5. Build embeddings and run semantic search.

## Project Structure

```text
project/
  src/
    app.py
    add_ids.py
    generate_questions.py
    semantic_chunk.py
    embed.py
    embed_missing.py
    recherche.py
    pipeline_extract_semantic.py
  pdfs/
  output/
    extracted/
    with_ids/
    questions/
    chunks/
    embeddings/
```

## Requirements

- Python 3.10+
- Packages:
  - pandas
  - pymupdf
  - requests
  - google-genai
  - numpy

Install:

```bash
pip install pandas pymupdf requests google-genai numpy
```

## Environment Variables

For Gemini-based scripts (`semantic_chunk.py`, `embed.py`, `embed_missing.py`, `recherche.py`):

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

For question generation (`generate_questions.py` + Ollama):

- `OLLAMA_BASE_URL` (optional, default: `http://localhost:11434`)
- `OLLAMA_MODEL` (optional, default: `qwen2.5:latest`)

Notes:

- Most scripts load `.env` from project root when run from root.
- `generate_questions.py` also loads `src/.env`.

## Step-by-Step Workflow

### 1) Extract Articles From PDFs

Process all PDFs in `pdfs/`:

```bash
python src/app.py
```

Process one PDF:

```bash
python src/app.py --pdf "pdfs/codedecommerce.pdf"
```

Optional page limit:

```bash
python src/app.py --pdf "pdfs/codedecommerce.pdf" --max-pages 50
```

Important default:

- `--max-pages 0` means all pages (no page cap).

Default extraction outputs:

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

Process all `*_articles.json` files in a folder:

```bash
python src/add_ids.py output/extracted --output-dir output/with_ids
```

Process one file:

```bash
python src/add_ids.py output/extracted/codedecommerce_articles.json --output-dir output/with_ids
```

Outputs:

- `*_with_ids.json`
- `*_with_ids.csv`

### 3) Generate Questions

`generate_questions.py` accepts a single JSON file per run.

```bash
python src/generate_questions.py output/with_ids/codedecommerce_articles_with_ids.json --output-dir output/questions
```

Outputs:

- `output/questions/<input_stem>_questions.json`
- `output/questions/<input_stem>_questions.csv`
- intermediate batch files in `output/questions/<input_stem>_batches/`

### 4) Semantic Chunking

Chunk from extracted JSON:

```bash
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json
```

Chunk directly from PDF (extract + chunk in one command):

```bash
python src/semantic_chunk.py "pdfs/codedecommerce.pdf"
```

Important default:

- `semantic_chunk.py --max-pages` now defaults to `0` (all pages).
- There is no default 50-page cap.

If you want a limit, pass it explicitly:

```bash
python src/semantic_chunk.py "pdfs/codedecommerce.pdf" --max-pages 50
```

Useful options:

```bash
# Tune chunk behavior
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json --target-chars 800 --max-chars 1300 --similarity-threshold 0.70

# Save checkpoints every N articles (default is 50)
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json --checkpoint-every 50

# Resume from latest checkpoint
python src/semantic_chunk.py output/extracted/codedecommerce_articles.json --resume

# Clean an existing semantic chunks JSON and regenerate CSV
python src/semantic_chunk.py output/chunks/codedecommerce_articles_semantic_chunks.json --clean
```

Default chunk outputs:

- `output/chunks/<input_stem>_semantic_chunks.json`
- `output/chunks/<input_stem>_semantic_chunks.csv`
- checkpoints: `output/chunks/checkpoints/<input_stem>/`

Chunk rows include:

- `chunk_id`, `chunk_index`, `chunk_count`
- `document_name`, `article_number`
- `livre`, `titre`, `chapitre`, `section`, `sous_section`
- `pages`, `content`, `source_relative_path`

### 5) Build Embeddings

```bash
python src/embed.py output/extracted/codedecommerce_articles.json
```

Custom output:

```bash
python src/embed.py output/extracted/codedecommerce_articles.json -o output/embeddings/codedecommerce_embedded.json
```

Resume only missing embeddings:

```bash
python src/embed_missing.py output/embeddings/codedecommerce_embedded.json
```

### 6) Semantic Search

```bash
python src/recherche.py output/embeddings/codedecommerce_embedded.json -q "obligation d'ouvrir un compte"
```

Optional filters:

```bash
python src/recherche.py output/embeddings/codedecommerce_embedded.json -q "compte bancaire" -k 10 -t 0.5
```

## Optional One-Command Extract + Chunk Pipeline

If you prefer a dedicated wrapper:

```bash
python src/pipeline_extract_semantic.py "pdfs/codedecommerce.pdf"
```

## Troubleshooting

- If imports fail, reinstall dependencies:

```bash
pip install --upgrade pandas pymupdf requests google-genai numpy
```

- If Gemini scripts fail, verify `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set.
- If question generation fails, verify Ollama is running and reachable at `OLLAMA_BASE_URL`.
