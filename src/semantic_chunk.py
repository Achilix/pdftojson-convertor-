import argparse
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


def _embed_text_with_retry(
	client: Any,
	text: str,
	model: str,
	max_retries: int,
) -> List[float]:
	attempt = 0
	while True:
		try:
			response = client.models.embed_content(
				model=f"models/{model}" if not model.startswith("models/") else model,
				contents=text,
			)
			if not response.embeddings:
				raise RuntimeError("No embedding returned")

			embedding = response.embeddings[0]
			if hasattr(embedding, "values"):
				return list(embedding.values)
			if isinstance(embedding, (list, tuple)):
				return list(embedding)
			raise RuntimeError(f"Unexpected embedding type: {type(embedding)}")
		except Exception as exc:
			attempt += 1
			if attempt > max_retries or not _is_retryable_error(exc):
				raise
			base_delay = min(30.0, 2 ** (attempt - 1))
			jitter = random.uniform(0.0, 0.35)
			time.sleep(base_delay + jitter)


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
) -> List[Dict[str, Any]]:
	content = str(article.get(text_field, "") or "").strip()
	if not content:
		return []

	sentences = _split_sentences(content, max_chars=max_chars)
	if not sentences:
		return []

	sentence_embeddings: List[List[float]] = []
	for sentence in sentences:
		sentence_embeddings.append(
			_embed_text_with_retry(
				client=client,
				text=sentence,
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

	article_id = article.get("id")
	article_number = article.get("article_number") or article.get("article") or "unknown"
	total_chunks = len(chunks)
	final_chunks: List[Dict[str, Any]] = []
	for index, chunk in enumerate(chunks, start=1):
		chunk_record: Dict[str, Any] = {
			"chunk_id": f"{article_number}_{index:03d}",
			"chunk_index": index,
			"chunk_count": total_chunks,
			"article_id": article_id,
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
			"article_id",
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
	checkpoint_dir: Optional[Path] = None,
	checkpoint_every: int = 5,
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
		"checkpoint_every": checkpoint_every,
	}

	print(f"Loaded {len(articles)} article(s) from {input_path.name}")
	if start_article_index > 0:
		print(f"Resuming from article index {start_article_index + 1}")

	for idx in range(start_article_index, len(articles)):
		article = articles[idx]
		article_number = article.get("article_number") or article.get("article") or "unknown"
		print(f"[{idx + 1}/{len(articles)}] Chunking article {article_number}...", flush=True)
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
		)
		all_chunks.extend(article_chunks)

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


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Semantically chunk extracted article JSON while preserving hierarchy metadata."
	)
	parser.add_argument("input", type=Path, help="Input articles JSON file")
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
		default=DEFAULT_MODEL,
		help=f"Embedding model name (default: {DEFAULT_MODEL})",
	)
	parser.add_argument(
		"-f",
		"--field",
		default="content",
		help="Article field to chunk (default: content)",
	)
	parser.add_argument(
		"--target-chars",
		type=int,
		default=900,
		help="Preferred chunk size in characters before semantic split decisions (default: 900)",
	)
	parser.add_argument(
		"--max-chars",
		type=int,
		default=1400,
		help="Hard max chunk size in characters (default: 1400)",
	)
	parser.add_argument(
		"--similarity-threshold",
		type=float,
		default=0.72,
		help="Cosine similarity threshold for keeping adjacent sentences together (default: 0.72)",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=5,
		help="Retries for transient API errors (default: 5)",
	)
	parser.add_argument(
		"--pause",
		type=float,
		default=0.15,
		help="Pause in seconds between API calls (default: 0.15)",
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
		default=5,
		help="Save checkpoint every N processed articles (default: 5)",
	)
	args = parser.parse_args()

	_load_env_file(Path.cwd() / ".env")
	api_key = _resolve_api_key(args.api_key)
	if not api_key:
		print("Error: Google API key required. Use --api-key or set GOOGLE_API_KEY / GEMINI_API_KEY in .env", file=sys.stderr)
		sys.exit(1)

	input_path = args.input.resolve()
	if not input_path.exists():
		print(f"Error: Input file not found: {input_path}", file=sys.stderr)
		sys.exit(1)

	if args.output:
		output_path = args.output.resolve()
	else:
		output_path = Path.cwd() / "output" / "chunks" / f"{input_path.stem}_semantic_chunks.json"

	if args.checkpoint_dir:
		checkpoint_dir = args.checkpoint_dir.resolve()
	else:
		checkpoint_dir = Path.cwd() / "output" / "chunks" / "checkpoints" / input_path.stem

	semantic_chunk_articles(
		input_path=input_path,
		output_path=output_path,
		api_key=api_key,
		model=args.model,
		text_field=args.field,
		target_chars=max(200, args.target_chars),
		max_chars=max(300, args.max_chars),
		similarity_threshold=max(0.0, min(1.0, args.similarity_threshold)),
		max_retries=max(0, args.max_retries),
		pause_seconds=max(0.0, args.pause),
		checkpoint_dir=checkpoint_dir,
		checkpoint_every=max(0, args.checkpoint_every),
	)


if __name__ == "__main__":
	main()
