import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from embed import DEFAULT_MODEL, embed_text


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


def _resolve_api_key(cli_api_key: str | None) -> str:
	if cli_api_key:
		return cli_api_key
	return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def _load_articles(input_path: Path) -> List[Dict[str, Any]]:
	with input_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError(f"Expected a JSON array in {input_path}")

	articles: List[Dict[str, Any]] = []
	for item in payload:
		if isinstance(item, dict):
			articles.append(item)
	return articles


def _write_articles(output_path: Path, articles: List[Dict[str, Any]]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as handle:
		json.dump(articles, handle, ensure_ascii=False, indent=2)
	tmp_path.replace(output_path)


def _has_embedding(article: Dict[str, Any]) -> bool:
	embedding = article.get("embedding")
	if embedding is None:
		return False

	if isinstance(embedding, list):
		if not embedding:
			return False
		first = embedding[0]
		# Standard form: [float, ...]
		if isinstance(first, (int, float)):
			return True
		# Legacy nested form: [["values", [float, ...]], ...]
		if isinstance(first, (list, tuple)) and len(first) == 2 and first[0] == "values":
			return isinstance(first[1], (list, tuple)) and len(first[1]) > 0

	return False


def _article_label(article: Dict[str, Any], index: int) -> str:
	if article.get("id") is not None:
		return str(article.get("id"))
	if article.get("article_number") is not None:
		return str(article.get("article_number"))
	return f"index-{index}"


def _is_retryable_error(exc: Exception) -> bool:
	message = str(exc).lower()
	return any(hint in message for hint in RETRYABLE_HINTS)


def _embed_with_retry(text: str, api_key: str, model: str, max_retries: int) -> List[float]:
	attempt = 0
	while True:
		try:
			return embed_text(text, api_key, model)
		except Exception as exc:
			attempt += 1
			if attempt > max_retries or not _is_retryable_error(exc):
				raise

			base_delay = min(30.0, 2 ** (attempt - 1))
			jitter = random.uniform(0.0, 0.35)
			time.sleep(base_delay + jitter)


def embed_missing_articles(
	input_path: Path,
	output_path: Path,
	api_key: str,
	model: str,
	text_field: str,
	max_retries: int,
	checkpoint_every: int,
	pause_seconds: float,
) -> None:
	articles = _load_articles(input_path)
	total = len(articles)
	already_embedded = sum(1 for article in articles if _has_embedding(article))
	missing = total - already_embedded

	print(f"Loaded {total} article(s) from {input_path.name}")
	print(f"Already embedded: {already_embedded}")
	print(f"Missing embeddings: {missing}\n")

	if missing == 0:
		_write_articles(output_path, articles)
		print(f"No missing embeddings. File written unchanged to {output_path}")
		return

	processed_missing = 0
	failures = 0

	for index, article in enumerate(articles, start=1):
		if _has_embedding(article):
			continue

		processed_missing += 1
		label = _article_label(article, index)
		text_to_embed = str(article.get(text_field, "") or "").strip()
		if not text_to_embed:
			print(f"[{processed_missing}/{missing}] Article {label}: SKIPPED (empty '{text_field}')")
			failures += 1
			continue

		print(f"[{processed_missing}/{missing}] Embedding article {label}...", end=" ", flush=True)
		try:
			embedding = _embed_with_retry(text_to_embed, api_key, model, max_retries=max_retries)
			article["embedding"] = embedding
			print(f"OK ({len(embedding)} dims)")
		except Exception as exc:
			failures += 1
			print(f"FAILED: {str(exc)[:180]}")

		if checkpoint_every > 0 and processed_missing % checkpoint_every == 0:
			_write_articles(output_path, articles)
			print(f"Checkpoint saved: {output_path}")

		if pause_seconds > 0:
			time.sleep(pause_seconds)

	_write_articles(output_path, articles)
	successes = missing - failures
	print("\nDone")
	print(f"Embedded now: {successes}/{missing}")
	print(f"Still missing: {failures}")
	print(f"Output: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Embed only articles missing an embedding field."
	)
	parser.add_argument("input", type=Path, help="Input JSON file (partially embedded or raw articles)")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Output JSON file (default: overwrite input file)",
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
		help="Field to embed (default: content)",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=5,
		help="Retries for transient API errors (default: 5)",
	)
	parser.add_argument(
		"--checkpoint-every",
		type=int,
		default=25,
		help="Save output every N processed missing records (default: 25)",
	)
	parser.add_argument(
		"--pause",
		type=float,
		default=0.3,
		help="Pause in seconds between embedding calls (default: 0.3)",
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

	output_path = args.output.resolve() if args.output else input_path
	embed_missing_articles(
		input_path=input_path,
		output_path=output_path,
		api_key=api_key,
		model=args.model,
		text_field=args.field,
		max_retries=max(0, args.max_retries),
		checkpoint_every=max(0, args.checkpoint_every),
		pause_seconds=max(0.0, args.pause),
	)


if __name__ == "__main__":
	main()