import argparse
import json
import re
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


ARTICLE_PATTERN = re.compile(r"(?im)^\s*Article\s+(\d+)\b")
LIVRE_PATTERN = re.compile(r"(?im)^\s*(Livre\b[^\n\r]*)")
TITRE_PATTERN = re.compile(r"(?im)^\s*(Titre\b[^\n\r]*)")
CHAPITRE_PATTERN = re.compile(r"(?im)^\s*((?:C|H)HAPITRE\b[^\n\r]*)")
SECTION_PATTERN = re.compile(r"(?im)^\s*(Section\b[^\n\r]*)")
SOUS_SECTION_PATTERN = re.compile(r"(?im)^\s*(Sous(?:\s|-)?section\b[^\n\r]*)")
FOOTNOTE_LINE_RE = re.compile(r"(?im)^\s*\d+\s*[-–—]\s+.*$")
INLINE_FOOTNOTE_SPLIT_RE = re.compile(
	r"(?im)\n\s*\d+\s*[-–—]\s*(?:Les\s+dispositions|Dahir|Bulletin\s+Officiel|Voir|Tel\s+qu['’]il)\b"
)
SKIP_ARTICLE_PREV_LINE_RE = re.compile(
	r"(?im)(?:\bVoir\s+articles?\b|pr[ée]cit[ée]e?\.\s*$)"
)


def _load_pymupdf_document(pdf_path: Path):
	"""Open a PDF with PyMuPDF."""
	try:
		import fitz  # PyMuPDF
	except ImportError as exc:
		raise RuntimeError(
			"Missing PyMuPDF. Install it with: `pip install pymupdf`."
		) from exc

	return fitz.open(str(pdf_path))


def _extract_page_structure(pdf_path: Path) -> Tuple[str, List[Tuple[int, int, int]], List[Tuple[int, str, str]]]:
	"""Extract raw text, page ranges, and hierarchy events from a PDF using PyMuPDF."""
	doc = _load_pymupdf_document(pdf_path)
	pages_data: List[Dict[str, object]] = []
	line_sizes: List[float] = []

	for page_num, page in enumerate(doc, start=1):
		page_dict = page.get_text("dict")
		page_lines: List[Dict[str, object]] = []

		for block in page_dict.get("blocks", []):
			if block.get("type") != 0:
				continue

			for line in block.get("lines", []):
				spans = line.get("spans", [])
				text = _normalize_heading("".join(span.get("text", "") for span in spans))
				if not text:
					continue

				span_sizes = [float(span.get("size", 0.0)) for span in spans if span.get("text")]
				line_size = max(span_sizes) if span_sizes else 0.0
				if line_size:
					line_sizes.append(line_size)

				page_lines.append(
					{
						"text": text,
						"spans": spans,
						"size": line_size,
					}
				)

		pages_data.append({"page_num": page_num, "lines": page_lines})

	body_size = statistics.median(line_sizes) if line_sizes else 12.0
	section_size_threshold = body_size * 1.12
	titre_size_threshold = body_size * 1.22
	livre_size_threshold = body_size * 1.35

	raw_parts: List[str] = []
	page_ranges: List[Tuple[int, int, int]] = []
	hierarchy_events: List[Tuple[int, str, str]] = []
	cursor = 0

	for page_data in pages_data:
		page_num = int(page_data["page_num"])
		lines = page_data["lines"]  # type: ignore[assignment]
		page_text_parts: List[str] = []
		page_start = cursor
		line_cursor = cursor

		for line in lines:
			text = str(line["text"])
			spans = line["spans"]  # type: ignore[assignment]
			line_size = float(line.get("size", 0.0))
			page_text_parts.append(text)
			added_sous_at_line_start = False

			if _is_livre_heading(text, spans, line_size, body_size, livre_size_threshold):
				hierarchy_events.append((line_cursor, "livre", _normalize_heading(text)))
			elif _is_titre_heading(text, spans, line_size, body_size, titre_size_threshold):
				hierarchy_events.append((line_cursor, "titre", _normalize_heading(text)))
			elif _is_chapitre_heading(text, spans, line_size, body_size, section_size_threshold):
				hierarchy_events.append((line_cursor, "chapitre", _normalize_heading(text)))
			elif _is_sous_section_heading(text, spans, line_size, body_size, section_size_threshold):
				hierarchy_events.append((line_cursor, "sous_section", _normalize_heading(text)))
				added_sous_at_line_start = True
			elif _is_section_heading(text, spans, line_size, body_size, section_size_threshold):
				hierarchy_events.append((line_cursor, "section", _normalize_heading(text)))

			sous_section_inline = _extract_sous_section_from_line(text)
			if sous_section_inline is not None:
				rel_pos, heading_value = sous_section_inline
				if not (added_sous_at_line_start and rel_pos == 0):
					hierarchy_events.append((line_cursor + rel_pos, "sous_section", heading_value))

			line_cursor += len(text) + 1

		page_text = "\n".join(page_text_parts)
		raw_parts.append(page_text)
		page_end = cursor + len(page_text)
		page_ranges.append((page_start, page_end, page_num))
		cursor = page_end + 1

	raw_text = "\n".join(raw_parts)
	hierarchy_events.sort(key=lambda item: item[0])
	return raw_text, page_ranges, hierarchy_events


def _position_to_page(position: int, page_ranges: List[Tuple[int, int, int]]) -> int:
	"""Map a character offset in the merged text to a page number."""
	if not page_ranges:
		return 1

	if position <= page_ranges[0][1]:
		return page_ranges[0][2]

	for start, end, page_num in page_ranges:
		if start <= position <= end:
			return page_num

	if position < page_ranges[0][0]:
		return page_ranges[0][2]

	return page_ranges[-1][2]


def _format_page_span(start_page: int, end_page: int) -> str:
	if start_page == end_page:
		return str(start_page)
	return f"{start_page}-{end_page}"


def _normalize_heading(value: str) -> str:
	"""Normalize heading whitespace into a single readable line."""
	cleaned = re.sub(r"\s+", " ", value.strip())
	return cleaned


def _line_starts_with(text: str, keyword: str) -> bool:
	return re.match(rf"(?i)^\s*{re.escape(keyword)}\b", text) is not None


def _extract_sous_section_from_line(text: str) -> Tuple[int, str] | None:
	"""Extract sous-section heading using the same regex idea as the provided script."""
	normalized = _normalize_heading(text)
	match = re.search(r"(?i)\bSOUS[ -]?SECTION\b\s*[:\-–—]?\s*", normalized)
	if not match:
		return None

	remainder = normalized[match.end():].strip()
	if remainder:
		next_heading = re.search(r"\b(LIVRE|TITRE|CHAPITRE|SECTION|SOUS[ -]?SECTION|ARTICLE)\b", remainder, re.I)
		if next_heading:
			remainder = remainder[:next_heading.start()].strip()

	heading_value = "Sous-section"
	if remainder:
		heading_value = _normalize_heading(f"Sous-section {remainder}")

	return match.start(), heading_value


def _should_skip_article_from_previous_context(raw_text: str, article_start: int) -> bool:
	"""Skip quoted external-law articles when preceding line indicates a footnote/citation."""
	window_start = max(0, article_start - 1000)
	context = raw_text[window_start:article_start]
	for line in reversed(context.splitlines()):
		candidate = _normalize_heading(line)
		if not candidate:
			continue
		return SKIP_ARTICLE_PREV_LINE_RE.search(candidate) is not None
	return False


def _strip_footnotes_from_content(content: str) -> str:
	"""Remove footnote-only lines and split mid-paragraph legal footnote tails."""
	filtered_lines: List[str] = []
	for line in content.splitlines():
		if FOOTNOTE_LINE_RE.match(line):
			continue
		filtered_lines.append(line)

	cleaned = "\n".join(filtered_lines)
	inline_match = INLINE_FOOTNOTE_SPLIT_RE.search(cleaned)
	if inline_match:
		cleaned = cleaned[:inline_match.start()]

	return cleaned.strip()


def _line_metrics(spans: List[Dict[str, object]]) -> Tuple[float, bool, str]:
	"""Return (max font size, bold flag, concatenated font names) for a line."""
	span_sizes = [float(span.get("size", 0.0)) for span in spans if span.get("text")]
	max_size = max(span_sizes) if span_sizes else 0.0
	fonts = [str(span.get("font", "")).lower() for span in spans]
	is_bold = any(
		any(marker in font for marker in ("bold", "black", "heavy", "semibold", "demibold"))
		for font in fonts
	)
	font_blob = " ".join(fonts)
	return max_size, is_bold, font_blob


def _text_has_heading_keyword(text: str, keyword: str) -> bool:
	return re.search(rf"(?i)\b{re.escape(keyword)}\b", text) is not None


def _is_livre_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
) -> bool:
	# Heading must start the line to avoid matching words inside paragraph content.
	return _line_starts_with(text, "livre")


def _is_titre_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
) -> bool:
	# Heading must start the line to avoid matching words inside paragraph content.
	return _line_starts_with(text, "titre")


def _is_section_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
) -> bool:
	if _is_sous_section_heading(text, spans, line_size, body_size, threshold):
		return False

	# Heading must start the line to avoid matching words inside paragraph content.
	return _line_starts_with(text, "section")


def _is_chapitre_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
) -> bool:
	# Heading must start the line and follow chapter-heading shape.
	# Supports OCR drop of leading "C" ("HAPITRE") but avoids in-paragraph citations.
	return re.match(
		r"(?i)^\s*(?:chapitre|hapitre)\s+(?:premier|[ivxlcdm]+|\d+)\s*[:：]",
		text,
	) is not None


def _is_sous_section_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
) -> bool:
	# Regex-only rule: if a line starts with "Sous-section" variant, treat it as sous_section.
	# Supports "Sous-section", "Sous section", and common dash characters.
	return re.match(r"(?i)^\s*sous(?:\s*[-–—‑]\s*|\s+)section\b", text) is not None


def _extract_articles(
	raw_text: str,
	page_ranges: List[Tuple[int, int, int]],
	hierarchy_events: List[Tuple[int, str, str]],
	document_name: str,
	source_relative_path: str,
) -> List[Dict[str, str]]:
	"""Find article sections and include hierarchy state at each article location."""
	matches = list(ARTICLE_PATTERN.finditer(raw_text))
	articles: List[Dict[str, str]] = []
	h_idx = 0

	current_livre = ""
	current_titre = ""
	current_chapitre = ""
	current_section = ""
	current_sous_section = ""

	for i, match in enumerate(matches):
		while h_idx < len(hierarchy_events) and hierarchy_events[h_idx][0] <= match.start():
			_, level, value = hierarchy_events[h_idx]

			if level == "livre":
				current_livre = value
				current_titre = ""
				current_chapitre = ""
				current_section = ""
				current_sous_section = ""
			elif level == "titre":
				current_titre = value
				current_chapitre = ""
				current_section = ""
				current_sous_section = ""
			elif level == "chapitre":
				current_chapitre = value
				current_section = ""
				current_sous_section = ""
			elif level == "section":
				current_section = value
				current_sous_section = ""
			elif level == "sous_section":
				current_sous_section = value

			h_idx += 1

		if _should_skip_article_from_previous_context(raw_text, match.start()):
			continue

		article_number = match.group(1)
		start_idx = match.end()
		next_match_start = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

		# If a heading starts between this article and the next one, exclude it from article content.
		content_end = next_match_start
		for event_pos, _, _ in hierarchy_events[h_idx:]:
			if event_pos <= start_idx:
				continue
			if event_pos >= next_match_start:
				break
			content_end = event_pos
			break

		end_idx = max(content_end - 1, match.start())

		start_page = _position_to_page(match.start(), page_ranges)
		end_page = _position_to_page(end_idx, page_ranges)

		content = _strip_footnotes_from_content(raw_text[start_idx:content_end])
		# Flatten whitespace so content is easier to read in JSON/CSV and Excel.
		content = re.sub(r"\r?\n+", " ", content)
		content = re.sub(r"[ \t]+", " ", content)
		content = content.strip()

		articles.append(
			{
				"document_name": document_name,
				"article_number": article_number,
				"livre": current_livre,
				"titre": current_titre,
				"chapitre": current_chapitre,
				"section": current_section,
				"sous_section": current_sous_section,
				"pages": _format_page_span(start_page, end_page),
				"content": content,
				"source_relative_path": source_relative_path,
			}
		)

	return articles


def process_pdf_to_outputs(pdf_path: Path, output_dir: Path) -> Tuple[Path, Path]:
	"""Parse one PDF and write extracted articles as JSON and CSV in output directory."""
	raw_text, page_ranges, hierarchy_events = _extract_page_structure(pdf_path)
	try:
		source_relative_path = pdf_path.relative_to(Path.cwd()).as_posix()
	except ValueError:
		source_relative_path = pdf_path.as_posix()

	articles = _extract_articles(
		raw_text,
		page_ranges,
		hierarchy_events,
		document_name=pdf_path.stem,
		source_relative_path=source_relative_path,
	)

	output_dir.mkdir(parents=True, exist_ok=True)
	json_file = output_dir / f"{pdf_path.stem}_articles.json"
	csv_file = output_dir / f"{pdf_path.stem}_articles.csv"

	with json_file.open("w", encoding="utf-8") as f:
		json.dump(articles, f, ensure_ascii=False, indent=2)

	try:
		df = pd.DataFrame(
			articles,
			columns=[
				"document_name",
				"article_number",
				"livre",
				"titre",
				"chapitre",
				"section",
				"sous_section",
				"pages",
				"content",
				"source_relative_path",
			],
		)
	except Exception as exc:
		raise RuntimeError(
			"Unable to build CSV with pandas. Ensure pandas is installed: `pip install pandas`."
		) from exc

	df.to_csv(csv_file, index=False, encoding="utf-8-sig")

	return json_file, csv_file


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Extract 'Article <number>' sections from PDF(s) and save to JSON."
	)
	parser.add_argument(
		"--pdf",
		type=str,
		default=None,
		help="Optional path to a single PDF file. If omitted, all PDFs in ./pdfs are processed.",
	)
	parser.add_argument(
		"--pdf-dir",
		type=str,
		default="pdfs",
		help="Directory containing PDF files (default: pdfs).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="output",
		help="Directory to write JSON output files (default: output).",
	)
	args = parser.parse_args()

	output_dir = Path(args.output_dir)

	if args.pdf:
		pdf_paths = [Path(args.pdf)]
	else:
		pdf_paths = sorted(Path(args.pdf_dir).glob("*.pdf"))

	if not pdf_paths:
		raise FileNotFoundError("No PDF files found to process.")

	for pdf_path in pdf_paths:
		if not pdf_path.exists():
			print(f"Skipping missing file: {pdf_path}")
			continue

		json_file, csv_file = process_pdf_to_outputs(pdf_path, output_dir)
		print(f"Processed {pdf_path} -> {json_file} and {csv_file}")


if __name__ == "__main__":
	main()
