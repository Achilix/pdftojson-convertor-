import argparse
import json
import re
import statistics
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


ARTICLE_PATTERN = re.compile(r"(?im)^\s*Article\s+(premi(?:er|ère)|1er|1re|\d+)\b")
LIVRE_PATTERN = re.compile(r"(?im)^\s*(Livre\b[^\n\r]*)")
TITRE_PATTERN = re.compile(r"(?im)^\s*(Titre\b[^\n\r]*)")
CHAPITRE_PATTERN = re.compile(r"(?im)^\s*((?:C|H)HAPITRE\b[^\n\r]*)")
SECTION_PATTERN = re.compile(r"(?im)^\s*(Section\b[^\n\r]*)")
SOUS_SECTION_PATTERN = re.compile(r"(?im)^\s*(Sous(?:\s|-)?section\b[^\n\r]*)")
NUMBERED_SOUS_SECTION_PATTERN = re.compile(r"(?i)^\s*(\d{1,2})\s*[-–—]\s+(.+?)\s*$")
FOOTNOTE_LINE_RE = re.compile(r"(?im)^\s*\d+\s*[-–—]\s*.*$")
PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
LEADING_SUBARTICLE_TOKEN_RE = re.compile(
	r"(?is)^\s*(?:[\.\-–—]\s*\d{1,6}(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?(?:\s*\d{2,3})?|(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\s*\.?\s*\d{2,3}|(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)(?:\s*[:：\-–—]\s*|\s+\(|\s+))"
)
FOOTNOTE_BODY_KEYWORD_RE = re.compile(
	r"(?i)^\s*(?:voir\b|article\b|les\s+dispositions\b|dahir\b|bulletin\s+officiel\b|comparer\b|m[êe]me\s+remarque\b|ibid\b|d[ée]cret\b|loi\b|a\s+comparer\b)"
)
INLINE_FOOTNOTE_SPLIT_RE = re.compile(
	r"(?im)\n\s*\d+\s*[-–—]\s*(?:Les\s+dispositions|Dahir|Bulletin\s+Officiel|Voir|Tel\s+qu['’]il)\b"
)
INLINE_PAREN_FOOTNOTE_SPLIT_RE = re.compile(
	r"(?i)\s+\(\d{1,2}\)\s*(?:l['’]emploi|le\s+ministre\s+d['’]etat|sont\s+abrog[ée]es|dans\s+les\s+cas\s+o[uù]|modifi[ée]|ajout[ée]|article\s+\d+\s*[—-])"
)
HEADER_FOOTER_LINE_PATTERNS = (
	re.compile(r"(?i)^\s*public\s*$"),
	re.compile(r"(?i)^\s*recueil\s+des\s+textes\s+l[ée]gislatifs\b.*$"),
	re.compile(r"(?i)^\s*mis\s+[àa]\s+jour\b.*$"),
	re.compile(r"(?i)^\s*\d+\s*\.\s*statut\s+de\s+bank\s+al[\s-]?maghrib\b.*$"),
	re.compile(r"(?i)^\s*publi[ée]\s+au\s+bulletin\s+officiel\b.*$"),
)
SKIP_ARTICLE_PREV_LINE_RE = re.compile(
	r"(?im)(?:\bVoir\s+articles?\b|pr[ée]cit[ée]e?\.\s*$)"
)
LEADING_ANNOTATION_LABEL_RE = re.compile(
	r"(?is)^\s*\(\s*((?:modifi[ée]|ajout[ée]|avant\s+dernier|dernier|paragraphes?|alin[ée]as?|le\s+n[°º]?\d+|les\s+num[ée]ros?)\s+[^)]{0,220})\)"
)
LEADING_ANNOTATION_LABEL_NO_OPEN_PAREN_RE = re.compile(
	r"(?is)^\s*((?:modifi[ée]|ajout[ée])\s+[^)]{0,220})\)"
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


def _extract_page_structure(
	pdf_path: Path,
	allow_unlabeled_sous_section: bool = False,
	max_pages: int = 0,
) -> Tuple[str, List[Tuple[int, int, int]], List[Tuple[int, str, str]]]:
	"""Extract raw text, page ranges, and hierarchy events from a PDF using PyMuPDF."""
	doc = _load_pymupdf_document(pdf_path)
	pages_data: List[Dict[str, object]] = []
	line_sizes: List[float] = []

	if max_pages > 0:
		page_iterable = enumerate(doc[:max_pages], start=1)
	else:
		page_iterable = enumerate(doc, start=1)

	for page_num, page in page_iterable:
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

		idx = 0
		while idx < len(lines):
			line = lines[idx]
			text = str(line["text"])
			spans = line["spans"]  # type: ignore[assignment]
			line_size = float(line.get("size", 0.0))
			page_text_parts.append(text)
			added_sous_at_line_start = False
			consumed_next_heading_line = False

			next_text = ""
			if idx + 1 < len(lines):
				next_text = str(lines[idx + 1]["text"])

			if _is_livre_heading(text, spans, line_size, body_size, livre_size_threshold):
				heading_value = _normalize_hierarchy_heading("livre", text)
				continuation = _extract_heading_continuation(next_text, "livre")
				if continuation is not None:
					heading_value = _normalize_hierarchy_heading("livre", f"{heading_value} {continuation}")
					consumed_next_heading_line = True
				hierarchy_events.append((line_cursor, "livre", heading_value))
			elif _is_titre_heading(text, spans, line_size, body_size, titre_size_threshold):
				heading_value = _normalize_hierarchy_heading("titre", text)
				continuation = _extract_heading_continuation(next_text, "titre")
				if continuation is not None:
					heading_value = _normalize_hierarchy_heading("titre", f"{heading_value} {continuation}")
					consumed_next_heading_line = True
				hierarchy_events.append((line_cursor, "titre", heading_value))
			elif _is_chapitre_heading(text, spans, line_size, body_size, section_size_threshold):
				heading_value = _normalize_hierarchy_heading("chapitre", text)
				continuation = _extract_heading_continuation(next_text, "chapitre")
				if continuation is not None:
					heading_value = _normalize_hierarchy_heading("chapitre", f"{heading_value} {continuation}")
					consumed_next_heading_line = True
				hierarchy_events.append((line_cursor, "chapitre", heading_value))
			elif _is_sous_section_heading(
				text,
				spans,
				line_size,
				body_size,
				section_size_threshold,
				allow_unlabeled_sous_section=allow_unlabeled_sous_section,
			):
				hierarchy_events.append((line_cursor, "sous_section", _normalize_heading(text)))
				added_sous_at_line_start = True
			elif _is_section_heading(
				text,
				spans,
				line_size,
				body_size,
				section_size_threshold,
				allow_unlabeled_sous_section=allow_unlabeled_sous_section,
			):
				heading_value = _normalize_hierarchy_heading("section", text)
				continuation = _extract_heading_continuation(next_text, "section")
				if continuation is not None:
					heading_value = _normalize_hierarchy_heading("section", f"{heading_value} {continuation}")
					consumed_next_heading_line = True
				hierarchy_events.append((line_cursor, "section", heading_value))

			sous_section_inline = _extract_sous_section_from_line(text)
			if sous_section_inline is not None:
				rel_pos, heading_value = sous_section_inline
				if not (added_sous_at_line_start and rel_pos == 0):
					hierarchy_events.append((line_cursor + rel_pos, "sous_section", heading_value))

			line_cursor += len(text) + 1

			if consumed_next_heading_line and idx + 1 < len(lines):
				second_line_text = str(lines[idx + 1]["text"])
				page_text_parts.append(second_line_text)
				line_cursor += len(second_line_text) + 1
				idx += 2
				continue

			idx += 1

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
	# OCR often glues superscript references to the last word (e.g., "CHÉQUE65").
	cleaned = re.sub(r"(?i)([A-Za-zÀ-ÖØ-öø-ÿ])(\d{2,3})$", r"\1.\2", cleaned)
	return cleaned


def _normalize_article_number(raw_number: str, previous_base: int | None) -> Tuple[str, int | None]:
	"""Normalize OCR-merged superscript references in article numbers.

	Example: if previous article is 9, raw "1012" is normalized to "10.12".
	"""
	raw_number = raw_number.strip()
	if re.match(r"(?i)^premi(?:er|ère)$", raw_number) or re.match(r"(?i)^1(?:er|re)$", raw_number):
		return "1", 1

	if not raw_number.isdigit():
		return raw_number, previous_base

	current_base = int(raw_number)
	if previous_base is None:
		return raw_number, current_base

	expected_base = previous_base + 1
	expected_str = str(expected_base)

	# If OCR merged a superscript/reference number into the article number,
	# split it as <expected_article>.<suffix>.
	if raw_number.startswith(expected_str) and len(raw_number) > len(expected_str):
		suffix = raw_number[len(expected_str):]
		if suffix.isdigit() and len(suffix) <= 3:
			return f"{expected_str}.{suffix}", expected_base

	return raw_number, current_base


def _extract_subarticle_suffix(raw_article_body: str) -> str | None:
	"""Extract a leading sub-article marker from article body text.

	Examples:
	- "-1 ..." -> "1"
	- ".2 ..." -> "2"
	- "-3 bis116 ..." -> "3 bis.116"
	- "-4117 ..." -> "4.117"
	- "bis252 ..." -> "bis.252"
	"""
	qualifier_only_match = re.match(
		r"(?is)^\s*(?:[\.\-–—]\s*)?(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\s*\.?\s*(\d{2,3})\b",
		raw_article_body,
	)
	if qualifier_only_match:
		qualifier = qualifier_only_match.group(1).lower().strip()
		superscript = qualifier_only_match.group(2).strip()
		return f"{qualifier}.{superscript}"

	qualifier_marker_only_match = re.match(
		r"(?is)^\s*(?:[\.\-–—]\s*)?(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)(?:\s*[:：\-–—]\s+|\s+\()",
		raw_article_body,
	)
	if qualifier_marker_only_match:
		return qualifier_marker_only_match.group(1).lower().strip()

	qualifier_plain_match = re.match(
		r"(?is)^\s*(?:[\.\-–—]\s*)?(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\s+(?=[A-Za-zÀ-ÖØ-öø-ÿ])",
		raw_article_body,
	)
	if qualifier_plain_match:
		return qualifier_plain_match.group(1).lower().strip()

	match = re.match(
		r"(?is)^\s*[\.\-–—]\s*(\d{1,6})(?:\s*(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?(?:\s*(\d{2,3}))?\b",
		raw_article_body,
	)
	if not match:
		return None

	raw_number = match.group(1)
	qualifier = (match.group(2) or "").lower().strip()
	explicit_superscript = (match.group(3) or "").strip()

	base_number = raw_number
	compact_superscript = ""

	# OCR sometimes merges sub-article and superscript note into one token (e.g., 4117, 10123).
	if len(raw_number) >= 3:
		for split_at in (2, 1, 3):
			if split_at >= len(raw_number):
				continue
			left = raw_number[:split_at]
			right = raw_number[split_at:]
			if not left.isdigit() or not right.isdigit():
				continue
			left_num = int(left)
			if 1 <= left_num <= 30 and 2 <= len(right) <= 3:
				base_number = str(left_num)
				compact_superscript = right
				break

	if not base_number.isdigit():
		return None

	base_num_value = int(base_number)
	if not (1 <= base_num_value <= 30):
		return None

	normalized = str(base_num_value)
	if qualifier:
		normalized += f" {qualifier}"

	superscript = explicit_superscript or compact_superscript
	if superscript:
		normalized += f".{superscript}"

	return normalized


def _line_starts_with(text: str, keyword: str) -> bool:
	return re.match(rf"(?i)^\s*{re.escape(keyword)}\b", text) is not None


def _is_probable_header_footer_line(text: str) -> bool:
	normalized = _normalize_heading(text)
	if not normalized:
		return False

	if PAGE_NUMBER_LINE_RE.match(normalized):
		return True

	for pattern in HEADER_FOOTER_LINE_PATTERNS:
		if pattern.match(normalized):
			return True

	if re.match(r"(?i)^\s*\d+\s*\.\s+[A-ZÀ-ÖØ-Þ\s'’\-]{8,}\s*$", normalized):
		return True

	return False


def _is_uppercase_heading_fragment(text: str) -> bool:
	letters = [ch for ch in text if ch.isalpha()]
	if len(letters) < 8:
		return False
	upper_count = sum(1 for ch in letters if ch.isupper())
	return (upper_count / len(letters)) >= 0.75


def _extract_heading_continuation(next_text: str, level: str) -> str | None:
	"""Return a heading continuation line (e.g., "De ...") if it looks like a title fragment."""
	normalized = _normalize_heading(next_text)
	if not normalized:
		return None

	if _is_probable_header_footer_line(normalized):
		return None

	if FOOTNOTE_LINE_RE.match(normalized):
		return None

	if re.match(r"(?i)^\s*(?:livre|titre|chapitre|section|sous(?:\s|-)?section|article)\b", normalized):
		return None

	if NUMBERED_SOUS_SECTION_PATTERN.match(normalized):
		return None

	if re.match(r"(?i)^(de(?:\s+l['’]?)?|des|du|d['’])\b", normalized) is None:
		if level not in {"livre", "titre", "chapitre", "section"}:
			return None
		if not _is_uppercase_heading_fragment(normalized):
			return None

	if len(normalized) > 160:
		return None

	return normalized


def _normalize_hierarchy_heading(level: str, heading_value: str) -> str:
	"""Normalize structural heading text for hierarchy fields."""
	normalized = _normalize_heading(heading_value)

	if level == "chapitre":
		# OCR often glues note markers to Roman numerals: "CHAPITRE IX9" -> "CHAPITRE IX".
		normalized = re.sub(
			r"(?i)^\s*((?:chapitre|hapitre)\s+[ivxlcdm]+)\d{1,2}(\b.*)$",
			r"\1\2",
			normalized,
		)

	return normalized


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


def _should_skip_footnote_article_reset(
	raw_text: str,
	article_start: int,
	raw_article_number: str,
	previous_article_base: int | None,
) -> bool:
	"""Skip fake article headings emitted inside footnote-note blocks (e.g., note text with 'Article 1 —')."""
	if previous_article_base is None:
		return False

	if not raw_article_number.isdigit():
		return False

	current_number = int(raw_article_number)
	# Footnote blocks often restart at tiny article numbers while the code is already far ahead.
	if not (current_number <= 5 and previous_article_base >= 10):
		return False

	window_start = max(0, article_start - 1800)
	context = raw_text[window_start:article_start]
	recent_lines = [_normalize_heading(line) for line in context.splitlines() if _normalize_heading(line)]
	if not recent_lines:
		return False

	last_chunk = "\n".join(recent_lines[-10:])
	if re.search(r"(?i)\bdispose\s+en\s+outre\s+que\b", last_chunk):
		return True

	if re.search(
		r"(?is)\(\d{1,2}\)\s*(?:l['’]emploi|le\s+ministre|sont\s+abrog[ée]es|dans\s+les\s+cas\s+o[uù]|modifi[ée]|ajout[ée]|article\s+\d+\s*[—-])",
		last_chunk,
	):
		return True

	return False


def _strip_footnotes_from_content(content: str) -> str:
	"""Remove footnote-only lines and split mid-paragraph legal footnote tails."""
	filtered_lines: List[str] = []
	for line in content.splitlines():
		normalized = _normalize_heading(line)
		if not normalized:
			continue

		if PAGE_NUMBER_LINE_RE.match(normalized):
			continue

		if _is_probable_header_footer_line(normalized):
			continue

		if _is_probable_footnote_line(normalized):
			continue

		filtered_lines.append(normalized)

	cleaned = "\n".join(filtered_lines)
	cleaned = LEADING_SUBARTICLE_TOKEN_RE.sub("", cleaned, count=1)
	cut_pos: int | None = None
	inline_match = INLINE_FOOTNOTE_SPLIT_RE.search(cleaned)
	if inline_match:
		cut_pos = inline_match.start()

	paren_match = INLINE_PAREN_FOOTNOTE_SPLIT_RE.search(cleaned)
	if paren_match and (cut_pos is None or paren_match.start() < cut_pos):
		cut_pos = paren_match.start()

	if cut_pos is not None:
		cleaned = cleaned[:cut_pos]

	return cleaned.strip()


def _extract_leading_annotation_label(content: str) -> str | None:
	"""Return a leading amendment-note label to disambiguate duplicate article numbers."""
	normalized = _normalize_heading(content)
	if not normalized:
		return None

	match = LEADING_ANNOTATION_LABEL_RE.match(normalized)
	if not match:
		match = LEADING_ANNOTATION_LABEL_NO_OPEN_PAREN_RE.match(normalized)
		if not match:
			return None

	label_core = _normalize_heading(match.group(1))
	if not label_core:
		return None

	return f"({label_core})"


def _is_probable_footnote_line(line: str) -> bool:
	"""Classify numbered dash lines as footnotes using legal-citation heuristics."""
	match = FOOTNOTE_LINE_RE.match(line)
	if not match:
		return False

	marker_match = re.match(r"^\s*(\d+)\s*[-–—]", line)
	marker_number = int(marker_match.group(1)) if marker_match else 0
	if marker_number >= 30:
		return True

	body = re.sub(r"^\s*\d+\s*[-–—]\s*", "", line).strip()
	if not body:
		return False

	# Avoid stripping ordinary enumerations used in legal prose.
	if re.match(r"(?i)^(?:l['’]|le\b|la\b|les\b|du\b|de\b|des\b)", body):
		return False

	if FOOTNOTE_BODY_KEYWORD_RE.match(body):
		return True

	if re.search(r"(?i)\b(?:article|dahir|bulletin\s+officiel|d[ée]cret|loi|code)\b", body):
		return True

	# Dense citation-like lines are usually notes, not article prose.
	return len(body) > 120


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
	# Require a real title-heading shape (e.g., "Titre I", "Titre premier").
	# This avoids false positives like "titre particulier ..." inside article text.
	return re.match(
		r"(?i)^\s*titre\s+(?:premier|[ivxlcdm]+|\d+)\b",
		text,
	) is not None


def _is_section_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
	allow_unlabeled_sous_section: bool = True,
) -> bool:
	if _is_sous_section_heading(
		text,
		spans,
		line_size,
		body_size,
		threshold,
		allow_unlabeled_sous_section=allow_unlabeled_sous_section,
	):
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
	# Some corpora render "CHAPITRE PREMIER" without trailing colon.
	# OCR may glue footnote markers to Roman numerals (e.g., "CHAPITRE IX9").
	return re.match(
		r"(?i)^\s*(?:chapitre|hapitre)\s+(?:premier|[ivxlcdm]+(?:\d{1,2})?|\d+)\b(?:\s*[:：].*)?$",
		text,
	) is not None


def _is_sous_section_heading(
	text: str,
	spans: List[Dict[str, object]],
	line_size: float,
	body_size: float,
	threshold: float,
	allow_unlabeled_sous_section: bool = True,
) -> bool:
	# Regex-only rule: if a line starts with "Sous-section" variant, treat it as sous_section.
	# Supports "Sous-section", "Sous section", and common dash characters.
	if re.match(r"(?i)^\s*sous(?:\s*[-–—‑]\s*|\s+)section\b", text) is not None:
		return True

	normalized = _normalize_heading(text)
	numbered_match = NUMBERED_SOUS_SECTION_PATTERN.match(normalized)
	if not numbered_match:
		if not allow_unlabeled_sous_section:
			return False

		# Dahir corpus uses unlabeled headings like "Du commandement".
		if not _is_probable_unlabeled_sous_section_title(normalized):
			return False

		max_size, is_bold, _ = _line_metrics(spans)
		effective_size = line_size or max_size
		return is_bold or effective_size >= (body_size * 1.08)

	number = int(numbered_match.group(1))
	remainder = numbered_match.group(2).strip()
	if not (1 <= number <= 30 and _is_probable_numbered_sous_section_title(remainder)):
		return False

	# TOC and note lines often include dot leaders; headings in body text do not.
	if re.search(r"\.{5,}", remainder):
		return False

	max_size, is_bold, _ = _line_metrics(spans)
	effective_size = line_size or max_size
	return is_bold or effective_size >= (body_size * 1.06)


def _is_probable_numbered_sous_section_title(text: str) -> bool:
	"""Heuristic filter for numbered sous-section headings (e.g., "1- De ...")."""
	normalized = _normalize_heading(text)
	if not normalized:
		return False

	if len(normalized) > 90:
		return False

	if re.search(r"(?i)\b(article|dahir|bulletin|officiel|voir|d[ée]cret|loi|pr[ée]cit[ée]e?)\b", normalized):
		return False

	# In this corpus these headings are phrased as "De/Des/Du ...".
	if re.match(r"(?i)^(de|des|du)\b", normalized) is None:
		return False

	return True


def _is_probable_unlabeled_sous_section_title(text: str) -> bool:
	"""Heuristic for Dahir-style sous-section headings (e.g., "Du commandement")."""
	normalized = _normalize_heading(text)
	if not normalized:
		return False

	if len(normalized) > 90:
		return False

	if re.search(r"\.{5,}", normalized):
		return False

	if re.search(r"(?i)\b(article|dahir|bulletin|officiel|voir|d[ée]cret|loi|code|pr[ée]cit[ée]e?)\b", normalized):
		return False

	if re.match(r"(?i)^(de(?:\s+l['’]?)?|du|des)\b", normalized) is None:
		return False

	first_alpha_match = re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", normalized)
	if first_alpha_match and normalized[first_alpha_match.start()].islower():
		return False

	if normalized.endswith((".", ";", ":", ",")):
		return False

	word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’-]+", normalized))
	if word_count < 2 or word_count > 10:
		return False

	return True


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
	previous_article_base: int | None = None

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

		raw_article_number = match.group(1)
		if _should_skip_footnote_article_reset(
			raw_text,
			match.start(),
			raw_article_number,
			previous_article_base,
		):
			continue

		article_number, current_article_base = _normalize_article_number(
			raw_article_number,
			previous_article_base,
		)
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

		raw_article_body = raw_text[start_idx:content_end]
		subarticle_suffix = _extract_subarticle_suffix(raw_article_body)
		if (
			subarticle_suffix is not None
			and previous_article_base is not None
			and current_article_base is not None
			and previous_article_base == current_article_base
			and re.fullmatch(r"\d+", article_number) is not None
		):
			if re.match(r"(?i)^(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\b", subarticle_suffix):
				article_number = f"{current_article_base}{subarticle_suffix}"
			else:
				article_number = f"{current_article_base}-{subarticle_suffix}"

		content = _strip_footnotes_from_content(raw_article_body)
		annotation_label = _extract_leading_annotation_label(content)
		if (
			annotation_label is not None
			and re.fullmatch(r"\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)?", article_number) is not None
		):
			article_number = f"{article_number} {annotation_label}"

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

		if current_article_base is not None:
			previous_article_base = current_article_base

	return articles


def process_pdf_to_outputs(pdf_path: Path, output_dir: Path, max_pages: int = 0) -> Tuple[Path, Path]:
	"""Parse one PDF and write extracted articles as JSON and CSV in output directory."""
	allow_unlabeled_sous_section = re.search(r"(?i)dahir", pdf_path.stem) is not None
	raw_text, page_ranges, hierarchy_events = _extract_page_structure(
		pdf_path,
		allow_unlabeled_sous_section=allow_unlabeled_sous_section,
		max_pages=max(0, max_pages),
	)
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
	if max_pages > 0:
		suffix = f"_first_{max_pages}p_articles"
	else:
		suffix = "_articles"

	json_file = output_dir / f"{pdf_path.stem}{suffix}.json"
	csv_file = output_dir / f"{pdf_path.stem}{suffix}.csv"

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
		default="output/extracted",
		help="Directory to write JSON output files (default: output/extracted).",
	)
	parser.add_argument(
		"--max-pages",
		type=int,
		default=0,
		help="Only process the first N pages (default: 0, process all pages).",
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

		json_file, csv_file = process_pdf_to_outputs(pdf_path, output_dir, max_pages=max(0, args.max_pages))
		print(f"Processed {pdf_path} -> {json_file} and {csv_file}")


if __name__ == "__main__":
	main()
