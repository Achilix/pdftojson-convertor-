# PDF Articles Pipeline

This project extracts legal articles from PDF files, assigns stable IDs, and generates study questions.

It is designed for documents structured with headings such as:

- Livre
- Titre
- Chapitre
- Section
- Sous-section
- Article

## Features

- Extracts each article with:
  - article number
  - content
  - page span
  - hierarchy fields (`livre`, `titre`, `chapitre`, `section`, `sous_section`)
- Exports both JSON and CSV
- Includes source metadata:
  - `document_name` (first column)
  - `source_relative_path` (last column)
- Uses PyMuPDF line parsing for cleaner heading detection
- Handles OCR variant `HAPITRE` for chapter detection
- Normalizes OCR-merged article numbers (example: `1012` -> `10.12`)
- Converts repeated sub-articles to hyphen form when body markers exist (example: `392`, `-1`, `-2` -> `392-1`, `392-2`)

## Project Structure

```
project/
├── src/                    # Source code
│   ├── app.py             # Main extraction script
│   ├── add_ids.py         # Adds `id` values to extracted article files
│   ├── generate_questions.py  # Generates questions from a single `*_with_ids.json` file
│   ├── embed.py           # Generates embeddings for articles using Gemini API
│   └── recherche.py       # Semantic search using cosine similarity on embeddings
├── pdfs/                  # Input PDF files
├── output/                # Generated JSON/CSV files
├── README.md              # This file
└── .gitignore
```

### Running Scripts

From the project root directory:

```bash
python src/app.py
python src/add_ids.py
python src/generate_questions.py
python src/embed.py <input_json_file>
```

#### Embedding Articles

The `embed.py` script generates vector embeddings for articles using Google Generative AI (Gemini):

```bash
# Basic usage (embeds 'content' field)
python src/embed.py output/extracted/codedecommerce_articles.json

# With custom output file
python src/embed.py output/extracted/codedecommerce_articles.json -o output/embeddings/codedecommerce_embedded.json

# Specify custom text field to embed
python src/embed.py output/extracted/codedecommerce_articles.json -f "article"

# Use custom API key
python src/embed.py output/extracted/codedecommerce_articles.json -k YOUR_API_KEY

# Resume an interrupted embedding run (embed only missing records)
python src/embed_missing.py output/embeddings/codedecommerce_embedded.json
```

**Requirements for embedding:**

- Google API key configured via `.env` (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- Install google-generativeai: `pip install google-generativeai`

Create a `.env` file at project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

The script generates a new JSON file with an `embedding` field added to each article.

#### Semantic Search Using Cosine Similarity

The `recherche.py` script performs semantic search on embedded articles using cosine similarity of embedding vectors:

```bash
# Basic search (returns top 5 results)
python src/recherche.py output/embeddings/codedecommerce_embedded.json -q "obligation d'ouvrir un compte"

# Custom number of results
python src/recherche.py output/embeddings/codedecommerce_embedded.json -q "compte bancaire" -k 10

# Set minimum similarity threshold (0.0-1.0)
python src/recherche.py output/embeddings/codedecommerce_embedded.json -q "commerce électronique" -t 0.5

# Interactive mode (prompts for query)
python src/recherche.py output/embeddings/codedecommerce_embedded.json
```

**Features:**

- **Multilingual support**: Works with French, English, and other languages via Gemini embeddings
- Pure cosine similarity (no AI re-inference needed) - query embedding created once, compared against 841 pre-computed article embeddings
- Configurable number of results (default: 5)
- Optional similarity threshold filtering
- Shows similarity percentage and article content preview
- 3072-dimensional vector embeddings from `gemini-embedding-001` model

**How it works:**

1. Query is embedded once using Gemini API (3072 dims)
2. Pre-computed article embeddings loaded from JSON (841 articles × 3072 dims)
3. Cosine similarity calculated between query and each article embedding
4. Results sorted by similarity score and returned

**Example output (French query):**

```
#1 - Similarity: 0.8094 (80.94%)
Article: 18
Page(s): ?-?
Content: Tout commerçant, pour les besoins de son commerce, a l'obligation d'ouvrir un compte...

#2 - Similarity: 0.7162 (71.62%)
Article: 488
Page(s): ?-?
Content: L'établissement bancaire doit, préalablement à l'ouverture d'un compte, vérifier...
```

## Requirements

- Python 3.10+
- Packages:
  - `pandas`
  - `pymupdf`
  - `requests`
  - `google-genai` (for embedding articles with Gemini API)
  - `numpy` (for cosine similarity calculations in semantic search)

Install dependencies:

```bash
pip install pandas pymupdf requests google-genai numpy
```

## Usage

### 1) Process all PDFs in `pdfs/`

```bash
python app.py
```

### 2) Process a single PDF

```bash
python app.py --pdf pdfs/codedecommerce.pdf
```

### 3) Custom input/output folders

```bash
python app.py --pdf-dir pdfs --output-dir output
```

## Output Files

For each PDF, two files are generated:

- `output/<pdf_stem>_articles.json`
- `output/<pdf_stem>_articles.csv`

When processing all PDFs from `pdfs/`, two combined files are also generated:

pip install pandas pymupdf requests

- `output/all_pdfs_articles.csv`

### Output Columns (order)

### Step 1) Extract articles from PDFs

#### 1) Process all PDFs in `pdfs/`

```bash
python app.py
```

#### 2) Process a single PDF

```bash
python app.py --pdf pdfs/codedecommerce.pdf
```

#### 3) Custom input/output folders

```bash
python app.py --pdf-dir pdfs --output-dir output
```

### Step 2) Add article IDs

Run the ID assignment script on files produced by extraction.

Example:

```bash
python add_ids.py output/extracted --output-dir output/with_ids
```

### Step 3) Generate questions (single file per run)

`generate_questions.py` now accepts only one JSON file at a time.

```bash
python generate_questions.py output/with_ids/codedecommerce_articles_with_ids.json
```

Optional output directory:

```bash
python generate_questions.py output/with_ids/codedecommerce_articles_with_ids.json --output-dir output/questions
```

Passing a directory is not supported and will fail by design.

## Extraction Output Files

For each PDF, two files are generated:

- `output/<pdf_stem>_articles.json`
- `output/<pdf_stem>_articles.csv`

When processing all PDFs from `pdfs/`, two combined files are also generated:

- `output/all_pdfs_articles.json`
- `output/all_pdfs_articles.csv`

### Extraction Output Columns (order)

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

## Question Generation Output

For each processed input file, one JSON and one CSV are produced:

- `output/questions/<input_stem>_questions.json`
- `output/questions/<input_stem>_questions.csv`

The JSON includes metadata (`source_file`, `model`, `questions_per_article`, `batch_index`) and article question arrays.
The CSV includes `article_id` and `question_1` ... `question_10` columns.

## Notes

- `generate_questions.py` reads `OLLAMA_BASE_URL` (and optional `OLLAMA_MODEL`) from environment or `.env`.
- Use `--chunk-size`, `--pause-seconds`, and `--max-retries` to control API pressure and reliability.
- `--chunk-size` only affects internal processing batches; output stays a single JSON + CSV pair.
- Heading detection is start-of-line based to avoid false positives from normal paragraph text.
- `chapitre` sits between `titre` and `section` in hierarchy logic.
- `chapitre` headings are matched with heading shape like `CHAPITRE II:` (or OCR `HAPITRE II:`) to avoid citation false positives.
- New hierarchy resets lower levels:
  - new `livre` resets `titre`, `chapitre`, `section`, `sous_section`
  - new `titre` resets `chapitre`, `section`, `sous_section`
  - new `chapitre` resets `section`, `sous_section`
  - new `section` resets `sous_section`

### Article Number Normalization

- If OCR merges a superscript/reference into the article number, the parser can split it using sequence context:
  - example: `1012` -> `10.12`
- If repeated base articles carry explicit sub-markers at the start of content (`-1`, `.2`, etc.), the parser emits hyphenated sub-article IDs:
  - example: `392-1`, `392-2`, `430-1`

## Troubleshooting

- If you get import errors, install dependencies again:

```bash
pip install --upgrade pandas pymupdf requests
```

- If output seems stale, rerun extraction:

```bash
python app.py
```

- If question generation fails, verify:
  - `OLLAMA_BASE_URL` is set
  - you passed a file path (not a directory)
  - input JSON is a list of objects with `id`
