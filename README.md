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

- `app.py`: main extraction script
- `add_ids.py`: adds `id` values to extracted article files
- `generate_questions.py`: generates questions from a single `*_with_ids.json` file
- `pdfs/`: input PDFs
- `output/`: generated JSON/CSV

## Requirements

- Python 3.10+
- Packages:
  - `pandas`
  - `pymupdf`
  - `requests`

Install dependencies:

```bash
pip install pandas pymupdf requests
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
