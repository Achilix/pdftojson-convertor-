# PDF Articles Extractor

This project extracts legal articles from PDF files and exports them to JSON and CSV.

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

## Project Structure

- `app.py`: main extraction script
- `pdfs/`: input PDFs
- `output/`: generated JSON/CSV

## Requirements

- Python 3.10+
- Packages:
  - `pandas`
  - `pymupdf`

Install dependencies:

```bash
pip install pandas pymupdf
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

### Output Columns (order)

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

## Notes

- Heading detection is start-of-line based to avoid false positives from normal paragraph text.
- `chapitre` sits between `titre` and `section` in hierarchy logic.
- New hierarchy resets lower levels:
  - new `livre` resets `titre`, `chapitre`, `section`, `sous_section`
  - new `titre` resets `chapitre`, `section`, `sous_section`
  - new `chapitre` resets `section`, `sous_section`
  - new `section` resets `sous_section`

## Troubleshooting

- If you get import errors, install dependencies again:

```bash
pip install --upgrade pandas pymupdf
```

- If output seems stale, rerun:

```bash
python app.py
```
