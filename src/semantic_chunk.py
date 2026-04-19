import argparse
from collections import defaultdict
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
	import google.genai as genai
except ImportError as exc:
	raise RuntimeError("Missing google-genai. Install it with: pip install google-genai") from exc


DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_CHECKPOINT_EVERY = 50
RETRYABLE_HINTS = (
	"429",
	"500",
	"503",
	"504",
	"resource_exhausted",
	"unavailable",
	"internal",
	"deadline",
	"timeout",
)
FOOTER_ONLY_RE = re.compile(r"^\s*\d+\s*[\u00ad\-–—]*\s*$")


def _load_env_file(env_path: Path) -> None:
	"""Load simple KEY=VALUE pairs from a local .env file into os.environ."""
	if not env_path.exists():
		return

	with env_path.open("r", encoding="utf-8") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue

			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key and key not in os.environ:
				os.environ[key] = value


def _resolve_api_key(cli_api_key: Optional[str]) -> str:
	if cli_api_key:
		return cli_api_key
	return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def _is_retryable_error(exc: Exception) -> bool:
	message = str(exc).lower()
	return any(hint in message for hint in RETRYABLE_HINTS)


def _normalize_model_name(model: str) -> str:
	return f"models/{model}" if not model.startswith("models/") else model


def _extract_embedding_values(embedding: Any) -> List[float]:
	if hasattr(embedding, "values"):
		return list(embedding.values)
	if isinstance(embedding, (list, tuple)):
		return list(embedding)
	raise RuntimeError(f"Unexpected embedding type: {type(embedding)}")


def _embed_texts_with_retry(
	client: Any,
	texts: List[str],
	model: str,
	max_retries: int,
) -> List[List[float]]:
	if not texts:
		return []

	attempt = 0
	model_name = _normalize_model_name(model)
	while True:
		try:
			response = client.models.embed_content(
				model=model_name,
				contents=texts,
			)
			if not response.embeddings:
				raise RuntimeError("No embedding returned")

			if len(response.embeddings) != len(texts):
				raise RuntimeError(
					f"Embedding count mismatch: expected {len(texts)}, got {len(response.embeddings)}"
				)

			return [_extract_embedding_values(embedding) for embedding in response.embeddings]
		except Exception as exc:
			attempt += 1
			if attempt > max_retries or not _is_retryable_error(exc):
				raise
			base_delay = min(30.0, 2 ** (attempt - 1))
			jitter = random.uniform(0.0, 0.35)
			time.sleep(base_delay + jitter)


def _embed_text_with_retry(
	client: Any,
	text: str,
	model: str,
	max_retries: int,
) -> List[float]:
	return _embed_texts_with_retry(
		client=client,
		texts=[text],
		model=model,
		max_retries=max_retries,
	)[0]


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
	arr1 = np.array(vec1, dtype=np.float32)
	arr2 = np.array(vec2, dtype=np.float32)
	norm1 = np.linalg.norm(arr1)
	norm2 = np.linalg.norm(arr2)
	if norm1 == 0 or norm2 == 0:
		return 0.0
	return float(np.dot(arr1, arr2) / (norm1 * norm2))


def _normalize_spaces(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def _clean_chunk_content(text: str) -> str:
	if not text:
		return ""

	cleaned = text.replace("\u0007", " ")

	for marker in (
		"2.CADRE RELATIF AUX ETABLISSEMENTS DE CREDIT",
		"LOUANGE A DIEU SEUL",
		"A décidé ce qui suit",
	):
		idx = cleaned.find(marker)
		if idx != -1:
			cleaned = cleaned[:idx]

	cleaned = re.sub(
		r"\s*\d+\s+Publié au Bulletin officiel.*?(?=(?:\s+-\s)|$)",
		" ",
		cleaned,
		flags=re.IGNORECASE,
	)
	cleaned = re.sub(
		r"\s*\d+\s+Les dispositions de l[’']article.*?(?=(?:\s+-\s)|$)",
		" ",
		cleaned,
		flags=re.IGNORECASE,
	)
	cleaned = re.sub(
		r"\s*Ledit Dahir.*?(?=(?:\s+-\s)|$)",
		" ",
		cleaned,
		flags=re.IGNORECASE,
	)
	cleaned = re.sub(
		r"\s*\d+\s+L[’']article.*$",
		" ",
		cleaned,
		flags=re.IGNORECASE,
	)
	cleaned = re.sub(r"\s+\d+\s*[\u00ad\-–—]?\s*$", "", cleaned)

	return _normalize_spaces(cleaned)


def normalize_chunks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	kept: List[Dict[str, Any]] = []
	for row in rows:
		row.pop("article_id", None)
		content = _clean_chunk_content(str(row.get("content", "")))
		if not content:
			continue
		if FOOTER_ONLY_RE.match(content):
			continue
		row["content"] = content
		kept.append(row)

	group_key = lambda r: (
		r.get("document_name", ""),
		r.get("livre", ""),
		r.get("titre", ""),
		r.get("chapitre", ""),
		r.get("section", ""),
		r.get("sous_section", ""),
		r.get("article_number", ""),
	)

	grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
	for row in kept:
		grouped[group_key(row)].append(row)

	group_id_by_key: Dict[tuple, int] = {}
	next_group_id = 1
	for row in kept:
		key = group_key(row)
		if key not in group_id_by_key:
			group_id_by_key[key] = next_group_id
			next_group_id += 1

	for key, rows_in_group in grouped.items():
		total = len(rows_in_group)
		group_id = group_id_by_key[key]
		article_number = str(rows_in_group[0].get("article_number", "")).strip() or "unknown"
		for idx, row in enumerate(rows_in_group, start=1):
			row["chunk_index"] = idx
			row["chunk_count"] = total
			row["chunk_id"] = f"{article_number}_{idx:03d}_g{group_id:03d}"

	return kept


def clean_semantic_chunks_file(input_path: Path, output_path: Path) -> None:
	with input_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError(f"Expected a JSON array in {input_path}")

	rows = [item for item in payload if isinstance(item, dict)]
	cleaned_rows = normalize_chunks(rows)

	_write_final_outputs(output_path, cleaned_rows)
	print(f"Cleaned {len(cleaned_rows)} semantic chunk(s) -> {output_path}")
	print(f"Cleaned CSV -> {output_path.with_suffix('.csv')}")


def _split_long_sentence(text: str, max_chars: int) -> List[str]:
	clean = _normalize_spaces(text)
	if len(clean) <= max_chars:
		return [clean]

	parts: List[str] = []
	remaining = clean
	while len(remaining) > max_chars:
		split_at = remaining.rfind(" ", 0, max_chars + 1)
		if split_at < max_chars // 2:
			split_at = max_chars
		parts.append(remaining[:split_at].strip())
		remaining = remaining[split_at:].strip()
	if remaining:
		parts.append(remaining)
	return parts


def _split_sentences(text: str, max_chars: int) -> List[str]:
	raw_units = re.split(r"(?<=[\.!?;:])\s+", text.replace("\n", " "))
	sentences: List[str] = []
	for unit in raw_units:
		clean = _normalize_spaces(unit)
		if not clean:
			continue
		sentences.extend(_split_long_sentence(clean, max_chars=max_chars))
	return sentences


def _chunk_article_semantically(
	article: Dict[str, Any],
	client: Any,
	model: str,
	text_field: str,
	target_chars: int,
	max_chars: int,
	similarity_threshold: float,
	max_retries: int,
	pause_seconds: float,
	embed_batch_size: int,
) -> List[Dict[str, Any]]:
	content = str(article.get(text_field, "") or "").strip()
	if not content:
		return []

	sentences = _split_sentences(content, max_chars=max_chars)
	if not sentences:
		return []

	sentence_embeddings: List[List[float]] = []
	effective_batch_size = max(1, embed_batch_size)
	for start in range(0, len(sentences), effective_batch_size):
		batch_sentences = sentences[start:start + effective_batch_size]
		sentence_embeddings.extend(
			_embed_texts_with_retry(
				client=client,
				texts=batch_sentences,
				model=model,
				max_retries=max_retries,
			)
		)
		if pause_seconds > 0:
			time.sleep(pause_seconds)

	chunks: List[Dict[str, Any]] = []
	current_sentences: List[str] = [sentences[0]]
	current_embedding = sentence_embeddings[0]

	for idx in range(1, len(sentences)):
		next_sentence = sentences[idx]
		next_embedding = sentence_embeddings[idx]
		current_text = " ".join(current_sentences)
		candidate_text = f"{current_text} {next_sentence}".strip()
		similarity = _cosine_similarity(current_embedding, next_embedding)

		should_append = True
		if len(candidate_text) > max_chars:
			should_append = False
		elif len(current_text) >= target_chars and similarity < similarity_threshold:
			should_append = False

		if should_append:
			current_sentences.append(next_sentence)
			current_embedding = list((np.array(current_embedding) + np.array(next_embedding)) / 2.0)
		else:
			chunks.append({"text": current_text})
			current_sentences = [next_sentence]
			current_embedding = next_embedding

	if current_sentences:
		chunks.append({"text": " ".join(current_sentences).strip()})

	article_number = article.get("article_number") or article.get("article") or "unknown"
	total_chunks = len(chunks)
	final_chunks: List[Dict[str, Any]] = []
	for index, chunk in enumerate(chunks, start=1):
		chunk_record: Dict[str, Any] = {
			"chunk_id": f"{article_number}_{index:03d}",
			"chunk_index": index,
			"chunk_count": total_chunks,
			"article_number": article_number,
			"document_name": article.get("document_name", ""),
			"livre": article.get("livre", ""),
			"titre": article.get("titre", ""),
			"chapitre": article.get("chapitre", ""),
			"section": article.get("section", ""),
			"sous_section": article.get("sous_section", ""),
			"pages": article.get("pages", ""),
			"source_relative_path": article.get("source_relative_path", ""),
			"content": chunk["text"],
		}
		final_chunks.append(chunk_record)

	return final_chunks


def _write_json_atomic(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def _write_final_outputs(output_path: Path, chunks: List[Dict[str, Any]]) -> None:
	_write_json_atomic(output_path, chunks)

	csv_path = output_path.with_suffix(".csv")
	df = pd.DataFrame(
		chunks,
		columns=[
			"chunk_id",
			"chunk_index",
			"chunk_count",
			"article_number",
			"document_name",
			"livre",
			"titre",
			"chapitre",
			"section",
			"sous_section",
			"pages",
			"source_relative_path",
			"content",
		],
	)
	df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def _checkpoint_file_path(checkpoint_dir: Path, input_path: Path, next_article_index: int) -> Path:
	return checkpoint_dir / f"{input_path.stem}_semantic_checkpoint_{next_article_index:06d}.json"


def write_checkpoint(
	checkpoint_dir: Path,
	input_path: Path,
	output_path: Path,
	next_article_index: int,
	total_articles: int,
	chunks: List[Dict[str, Any]],
	settings: Dict[str, Any],
) -> Path:
	checkpoint_payload = {
		"input_path": str(input_path),
		"output_path": str(output_path),
		"next_article_index": next_article_index,
		"total_articles": total_articles,
		"chunks": chunks,
		"settings": settings,
		"created_at": time.time(),
	}
	checkpoint_path = _checkpoint_file_path(checkpoint_dir, input_path, next_article_index)
	_write_json_atomic(checkpoint_path, checkpoint_payload)
	return checkpoint_path


def find_latest_checkpoint(checkpoint_dir: Path, input_stem: str) -> Path:
	pattern = f"{input_stem}_semantic_checkpoint_*.json"
	candidates = list(checkpoint_dir.glob(pattern))
	if not candidates:
		raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir} for stem '{input_stem}'")

	def _extract_index(path: Path) -> int:
		match = re.search(r"_semantic_checkpoint_(\d{6})\.json$", path.name)
		return int(match.group(1)) if match else -1

	return max(candidates, key=_extract_index)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
	with checkpoint_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, dict):
		raise ValueError(f"Invalid checkpoint format in {checkpoint_path}")
	return payload


def _pick_setting(checkpoint_settings: Dict[str, Any], key: str, cli_value: Any) -> Any:
	if cli_value is not None:
		return cli_value
	return checkpoint_settings.get(key)


def semantic_chunk_articles(
	input_path: Path,
	output_path: Path,
	api_key: str,
	model: str,
	text_field: str,
	target_chars: int,
	max_chars: int,
	similarity_threshold: float,
	max_retries: int,
	pause_seconds: float,
	embed_batch_size: int = 16,
	checkpoint_dir: Optional[Path] = None,
	checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
	start_article_index: int = 0,
	seed_chunks: Optional[List[Dict[str, Any]]] = None,
) -> None:
	with input_path.open("r", encoding="utf-8-sig") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError(f"Expected a JSON array in {input_path}")

	articles: List[Dict[str, Any]] = [item for item in payload if isinstance(item, dict)]
	client = genai.Client(api_key=api_key)
	all_chunks: List[Dict[str, Any]] = list(seed_chunks or [])

	settings = {
		"model": model,
		"text_field": text_field,
		"target_chars": target_chars,
		"max_chars": max_chars,
		"similarity_threshold": similarity_threshold,
		"max_retries": max_retries,
		"pause_seconds": pause_seconds,
		"embed_batch_size": embed_batch_size,
		"checkpoint_every": checkpoint_every,
	}

	print(f"Loaded {len(articles)} article(s) from {input_path.name}")
	if start_article_index > 0:
		print(f"Resuming from article index {start_article_index + 1}")

	failures = 0

	for idx in range(start_article_index, len(articles)):
		article = articles[idx]
		article_number = article.get("article_number") or article.get("article") or "unknown"
		print(f"[{idx + 1}/{len(articles)}] Chunking article {article_number}...", flush=True)
		try:
			article_chunks = _chunk_article_semantically(
				article=article,
				client=client,
				model=model,
				text_field=text_field,
				target_chars=target_chars,
				max_chars=max_chars,
				similarity_threshold=similarity_threshold,
				max_retries=max_retries,
				pause_seconds=pause_seconds,
				embed_batch_size=embed_batch_size,
			)
			all_chunks.extend(article_chunks)
		except Exception as exc:
			failures += 1
			print(f"[{idx + 1}/{len(articles)}] FAILED article {article_number}: {str(exc)[:220]}")

		next_article_index = idx + 1
		if checkpoint_dir and checkpoint_every > 0 and (next_article_index % checkpoint_every == 0):
			checkpoint_path = write_checkpoint(
				checkpoint_dir=checkpoint_dir,
				input_path=input_path,
				output_path=output_path,
				next_article_index=next_article_index,
				total_articles=len(articles),
				chunks=all_chunks,
				settings=settings,
			)
			print(f"Checkpoint saved: {checkpoint_path}")
			_write_final_outputs(output_path, all_chunks)

	_write_final_outputs(output_path, all_chunks)

	if checkpoint_dir:
		final_checkpoint = write_checkpoint(
			checkpoint_dir=checkpoint_dir,
			input_path=input_path,
			output_path=output_path,
			next_article_index=len(articles),
			total_articles=len(articles),
			chunks=all_chunks,
			settings=settings,
		)
		print(f"Final checkpoint saved: {final_checkpoint}")

	print(f"\nSaved {len(all_chunks)} semantic chunk(s) to {output_path}")
	print(f"Saved CSV to {output_path.with_suffix('.csv')}")
	if failures > 0:
		print(f"Chunking completed with {failures} failed article(s).")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Semantically chunk extracted article JSON while preserving hierarchy metadata."
	)
	parser.add_argument("input", type=Path, help="Input articles JSON file or PDF file")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Output JSON file (default: output/chunks/<input_stem>_semantic_chunks.json)",
	)
	parser.add_argument(
		"-k",
		"--api-key",
		default=None,
		help="Google API key (or set GOOGLE_API_KEY / GEMINI_API_KEY in .env)",
	)
	parser.add_argument(
		"-m",
		"--model",
		default=None,
		help=f"Embedding model name (default: {DEFAULT_MODEL}; on --resume uses checkpoint setting)",
	)
	parser.add_argument(
		"-f",
		"--field",
		default=None,
		help="Article field to chunk (default: content; on --resume uses checkpoint setting)",
	)
	parser.add_argument(
		"--target-chars",
		type=int,
		default=None,
		help="Preferred chunk size in characters before semantic split decisions (default: 900)",
	)
	parser.add_argument(
		"--max-chars",
		type=int,
		default=None,
		help="Hard max chunk size in characters (default: 1400)",
	)
	parser.add_argument(
		"--similarity-threshold",
		type=float,
		default=None,
		help="Cosine similarity threshold for keeping adjacent sentences together (default: 0.72)",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=None,
		help="Retries for transient API errors (default: 5)",
	)
	parser.add_argument(
		"--pause",
		type=float,
		default=None,
		help="Pause in seconds between API calls (default: 0.15)",
	)
	parser.add_argument(
		"--embed-batch-size",
		type=int,
		default=None,
		help="Number of sentences per embedding request (default: 16)",
	)
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=None,
		help="Directory for checkpoint files (default: output/chunks/checkpoints/<input_stem>)",
	)
	parser.add_argument(
		"--checkpoint-every",
		type=int,
		default=None,
		help=f"Save checkpoint every N processed articles (default: {DEFAULT_CHECKPOINT_EVERY})",
	)
	parser.add_argument(
		"--max-pages",
		type=int,
		default=0,
		help="When input is a PDF, extract only first N pages before chunking (default: 0 = all pages)",
	)
	parser.add_argument(
		"--extract-output-dir",
		type=Path,
		default=Path("output/extracted"),
		help="When input is a PDF, directory for extracted articles JSON/CSV",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Resume from latest checkpoint for the given input articles JSON",
	)
	parser.add_argument(
		"--clean",
		action="store_true",
		help="Clean an existing semantic chunks JSON file and regenerate CSV",
	)
	args = parser.parse_args()

	input_path = args.input.resolve()
	if not input_path.exists():
		print(f"Error: Input file not found: {input_path}", file=sys.stderr)
		sys.exit(1)

	if input_path.suffix.lower() == ".pdf":
		if args.clean:
			print("Error: --clean requires a semantic chunks JSON file, not a PDF.", file=sys.stderr)
			sys.exit(1)
		if args.resume:
			print("Error: --resume requires extracted articles JSON, not a PDF.", file=sys.stderr)
			sys.exit(1)

		from app import process_pdf_to_outputs

		extract_output_dir = args.extract_output_dir.resolve()
		extracted_json, extracted_csv = process_pdf_to_outputs(
			pdf_path=input_path,
			output_dir=extract_output_dir,
			max_pages=max(0, args.max_pages),
		)
		print(f"Extraction done -> {extracted_json}")
		print(f"Extraction CSV -> {extracted_csv}")
		input_path = extracted_json

	if args.clean:
		if args.output:
			output_path = args.output.resolve()
		else:
			output_path = input_path
		clean_semantic_chunks_file(input_path=input_path, output_path=output_path)
		return

	_load_env_file(Path.cwd() / ".env")
	api_key = _resolve_api_key(args.api_key)
	if not api_key:
		print("Error: Google API key required. Use --api-key or set GOOGLE_API_KEY / GEMINI_API_KEY in .env", file=sys.stderr)
		sys.exit(1)

	if args.output:
		output_path = args.output.resolve()
	else:
		output_path = Path.cwd() / "output" / "chunks" / f"{input_path.stem}_semantic_chunks.json"

	if args.checkpoint_dir:
		checkpoint_dir = args.checkpoint_dir.resolve()
	else:
		checkpoint_dir = Path.cwd() / "output" / "chunks" / "checkpoints" / input_path.stem

	if args.resume:
		checkpoint_path = find_latest_checkpoint(checkpoint_dir, input_path.stem)
		checkpoint = load_checkpoint(checkpoint_path)
		settings = checkpoint.get("settings") if isinstance(checkpoint.get("settings"), dict) else {}

		next_article_index = int(checkpoint.get("next_article_index", 0))
		seed_chunks = checkpoint.get("chunks") if isinstance(checkpoint.get("chunks"), list) else []

		if args.output:
			output_path = args.output.resolve()
		else:
			checkpoint_output_raw = str(checkpoint.get("output_path", "") or "").strip()
			if checkpoint_output_raw:
				output_path = Path(checkpoint_output_raw).resolve()
			else:
				output_path = Path.cwd() / "output" / "chunks" / f"{input_path.stem}_semantic_chunks.json"

		model = _pick_setting(settings, "model", args.model) or DEFAULT_MODEL
		text_field = _pick_setting(settings, "text_field", args.field) or "content"
		target_chars = int(_pick_setting(settings, "target_chars", args.target_chars) or 900)
		max_chars = int(_pick_setting(settings, "max_chars", args.max_chars) or 1400)
		similarity_threshold = float(_pick_setting(settings, "similarity_threshold", args.similarity_threshold) or 0.72)
		max_retries = int(_pick_setting(settings, "max_retries", args.max_retries) or 5)
		pause_seconds = float(_pick_setting(settings, "pause_seconds", args.pause) or 0.15)
		embed_batch_size = int(_pick_setting(settings, "embed_batch_size", args.embed_batch_size) or 16)
		checkpoint_every = int(_pick_setting(settings, "checkpoint_every", args.checkpoint_every) or DEFAULT_CHECKPOINT_EVERY)

		print(f"Using checkpoint: {checkpoint_path}")
		print(f"Resuming from article index: {next_article_index + 1}")
		print(f"Loaded partial chunks: {len(seed_chunks)}")

		semantic_chunk_articles(
			input_path=input_path,
			output_path=output_path,
			api_key=api_key,
			model=model,
			text_field=text_field,
			target_chars=max(200, target_chars),
			max_chars=max(300, max_chars),
			similarity_threshold=max(0.0, min(1.0, similarity_threshold)),
			max_retries=max(0, max_retries),
			pause_seconds=max(0.0, pause_seconds),
			embed_batch_size=max(1, embed_batch_size),
			checkpoint_dir=checkpoint_dir,
			checkpoint_every=max(0, checkpoint_every),
			start_article_index=max(0, next_article_index),
			seed_chunks=seed_chunks,
		)
		return

	model = args.model or DEFAULT_MODEL
	text_field = args.field or "content"
	target_chars = args.target_chars if args.target_chars is not None else 900
	max_chars = args.max_chars if args.max_chars is not None else 1400
	similarity_threshold = args.similarity_threshold if args.similarity_threshold is not None else 0.72
	max_retries = args.max_retries if args.max_retries is not None else 5
	pause_seconds = args.pause if args.pause is not None else 0.15
	embed_batch_size = args.embed_batch_size if args.embed_batch_size is not None else 16
	checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else DEFAULT_CHECKPOINT_EVERY

	semantic_chunk_articles(
		input_path=input_path,
		output_path=output_path,
		api_key=api_key,
		model=model,
		text_field=text_field,
		target_chars=max(200, target_chars),
		max_chars=max(300, max_chars),
		similarity_threshold=max(0.0, min(1.0, similarity_threshold)),
		max_retries=max(0, max_retries),
		pause_seconds=max(0.0, pause_seconds),
		embed_batch_size=max(1, embed_batch_size),
		checkpoint_dir=checkpoint_dir,
		checkpoint_every=max(0, checkpoint_every),
	)


if __name__ == "__main__":
	main()
