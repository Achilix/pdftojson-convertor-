import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
	import google.genai as genai
except ImportError as exc:
	raise RuntimeError("Missing google-genai. Install it with: pip install google-genai") from exc


DEFAULT_MODEL = "gemini-embedding-001"


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


def _resolve_api_key(cli_api_key: str | None) -> str:
	if cli_api_key:
		return cli_api_key

	return (
		os.environ.get("GOOGLE_API_KEY")
		or os.environ.get("GEMINI_API_KEY")
		or ""
	)


def _normalize_model_name(model: str) -> str:
	return f"models/{model}" if not model.startswith("models/") else model


def _extract_embedding_values(embedding: Any) -> List[float]:
	if hasattr(embedding, "values"):
		return list(embedding.values)
	if isinstance(embedding, (list, tuple)):
		return list(embedding)
	raise RuntimeError(f"Unexpected embedding type: {type(embedding)}")


def embed_texts(
	texts: List[str],
	api_key: str,
	model: str = DEFAULT_MODEL,
	batch_size: int = 16,
) -> List[List[float]]:
	"""Generate embedding vectors for many texts using Gemini batching."""
	if not texts:
		return []

	if not api_key:
		raise ValueError("API key is required. Set GOOGLE_API_KEY or pass --api-key")

	effective_batch_size = max(1, int(batch_size or 1))
	model_name = _normalize_model_name(model)
	client = genai.Client(api_key=api_key)

	result: List[List[float]] = [[] for _ in texts]
	non_empty_items: List[Tuple[int, str]] = [
		(idx, str(text or ""))
		for idx, text in enumerate(texts)
		if str(text or "").strip()
	]

	for start in range(0, len(non_empty_items), effective_batch_size):
		batch_items = non_empty_items[start:start + effective_batch_size]
		batch_texts = [text for _, text in batch_items]

		try:
			response = client.models.embed_content(
				model=model_name,
				contents=batch_texts,
			)
		except Exception as exc:
			raise RuntimeError(f"Error calling Google Generative AI embedding API: {exc}") from exc

		if not response.embeddings:
			raise RuntimeError("No embedding returned")

		if len(response.embeddings) != len(batch_texts):
			raise RuntimeError(
				f"Embedding count mismatch: expected {len(batch_texts)}, got {len(response.embeddings)}"
			)

		for (original_index, _), embedding in zip(batch_items, response.embeddings):
			result[original_index] = _extract_embedding_values(embedding)

	return result


def embed_text(text: str, api_key: str, model: str = DEFAULT_MODEL) -> List[float]:
	"""Generate an embedding vector for text using Gemini."""
	if not text or not text.strip():
		return []

	vectors = embed_texts([text], api_key=api_key, model=model, batch_size=1)
	if not vectors or not vectors[0]:
		print(f"Warning: No embedding returned for text: {text[:50]}...")
		return []
	return vectors[0]


def _flush_pending_embeddings(
	pending: List[Tuple[int, Dict[str, Any], str, str]],
	embedded_articles: List[Dict[str, Any]],
	total: int,
	api_key: str,
	model: str,
) -> None:
	if not pending:
		return

	start_index = pending[0][0]
	end_index = pending[-1][0]
	batch_texts = [text for _, _, text, _ in pending]
	print(
		f"[{start_index}-{end_index}/{total}] Embedding batch ({len(pending)} article(s))...",
		end=" ",
		flush=True,
	)

	try:
		batch_embeddings = embed_texts(
			batch_texts,
			api_key=api_key,
			model=model,
			batch_size=max(1, len(batch_texts)),
		)

		for (_, article, _, _), embedding in zip(pending, batch_embeddings):
			article_with_embedding = dict(article)
			article_with_embedding["embedding"] = embedding
			embedded_articles.append(article_with_embedding)

		dims = len(batch_embeddings[0]) if batch_embeddings else 0
		print(f"OK ({dims} dims each)")
	except Exception as exc:
		print(f"BATCH FAILED: {str(exc)[:120]}")
		for article_index, article, text_to_embed, label in pending:
			print(f"[{article_index}/{total}] Embedding article {label}...", end=" ", flush=True)
			try:
				embedding = embed_text(text_to_embed, api_key, model)
				article_with_embedding = dict(article)
				article_with_embedding["embedding"] = embedding
				embedded_articles.append(article_with_embedding)
				print(f"OK ({len(embedding)} dims)")
			except Exception as inner_exc:
				embedded_articles.append(article)
				print(f"FAILED: {str(inner_exc)[:120]}")


def _load_articles_json(input_path: Path) -> List[Dict[str, Any]]:
	if input_path.suffix.lower() != ".json":
		raise ValueError(f"Expected a .json file, got: {input_path.name}")

	# utf-8-sig tolerates BOM-prefixed JSON files.
	with input_path.open("r", encoding="utf-8-sig") as handle:
		articles = json.load(handle)

	if not isinstance(articles, list):
		raise ValueError(f"Expected JSON array in {input_path}, got {type(articles)}")

	normalized: List[Dict[str, Any]] = []
	for item in articles:
		if isinstance(item, dict):
			normalized.append(item)
	return normalized


def _write_json_atomic(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)
	tmp_path.replace(path)


def _resolve_checkpoint_dir(
	input_path: Path,
	output_path: Path,
	checkpoint_dir: Path | None,
) -> Path:
	if checkpoint_dir is not None:
		return checkpoint_dir
	return output_path.parent / "checkpoints" / input_path.stem


def _write_embedding_checkpoint(
	checkpoint_dir: Path,
	input_path: Path,
	output_path: Path,
	processed_articles: int,
	total_articles: int,
	articles: List[Dict[str, Any]],
	settings: Dict[str, Any],
) -> Path:
	checkpoint_payload = {
		"input_path": str(input_path),
		"output_path": str(output_path),
		"processed_articles": processed_articles,
		"total_articles": total_articles,
		"articles": articles,
		"settings": settings,
		"created_at": time.time(),
	}
	checkpoint_path = checkpoint_dir / f"{input_path.stem}_embedding_checkpoint_{processed_articles:06d}.json"
	_write_json_atomic(checkpoint_path, checkpoint_payload)
	return checkpoint_path


def embed_articles(
	input_path: Path,
	output_path: Path,
	api_key: str,
	model: str = DEFAULT_MODEL,
	text_field: str = "content",
	checkpoint_every: int = 0,
	batch_size: int = 16,
	checkpoint_dir: Path | None = None,
) -> None:
	"""Load articles from JSON, embed content, and save results."""
	articles = _load_articles_json(input_path)
	print(f"Loading {len(articles)} articles from {input_path.name}")

	embedded_articles: List[Dict[str, Any]] = []
	pending: List[Tuple[int, Dict[str, Any], str, str]] = []
	effective_batch_size = max(1, batch_size)
	last_checkpoint_index = 0
	effective_checkpoint_dir: Path | None = None
	if checkpoint_every > 0:
		effective_checkpoint_dir = _resolve_checkpoint_dir(
			input_path=input_path,
			output_path=output_path,
			checkpoint_dir=checkpoint_dir,
		)

	settings = {
		"model": model,
		"text_field": text_field,
		"batch_size": effective_batch_size,
		"checkpoint_every": checkpoint_every,
	}

	for i, article in enumerate(articles, 1):
		text_to_embed = str(article.get(text_field, "") or "")
		if not text_to_embed.strip():
			print(f"Warning: Article {i} has no '{text_field}' field, skipping embedding")
			embedded_articles.append(article)
		else:
			label = article.get("article_number") or article.get("article") or "unknown"
			pending.append((i, article, text_to_embed, str(label)))

		if len(pending) >= effective_batch_size:
			_flush_pending_embeddings(
				pending=pending,
				embedded_articles=embedded_articles,
				total=len(articles),
				api_key=api_key,
				model=model,
			)
			pending = []

		if checkpoint_every > 0 and i % checkpoint_every == 0:
			if pending:
				_flush_pending_embeddings(
					pending=pending,
					embedded_articles=embedded_articles,
					total=len(articles),
					api_key=api_key,
					model=model,
				)
				pending = []

			output_path.parent.mkdir(parents=True, exist_ok=True)
			with output_path.open("w", encoding="utf-8") as handle:
				json.dump(embedded_articles, handle, ensure_ascii=False, indent=2)
			print(f"Checkpoint saved: {output_path}")
			if effective_checkpoint_dir is not None:
				checkpoint_path = _write_embedding_checkpoint(
					checkpoint_dir=effective_checkpoint_dir,
					input_path=input_path,
					output_path=output_path,
					processed_articles=i,
					total_articles=len(articles),
					articles=embedded_articles,
					settings=settings,
				)
				last_checkpoint_index = i
				print(f"Checkpoint snapshot: {checkpoint_path}")

	if pending:
		_flush_pending_embeddings(
			pending=pending,
			embedded_articles=embedded_articles,
			total=len(articles),
			api_key=api_key,
			model=model,
		)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(embedded_articles, handle, ensure_ascii=False, indent=2)

	if checkpoint_every > 0 and effective_checkpoint_dir is not None and last_checkpoint_index != len(articles):
		checkpoint_path = _write_embedding_checkpoint(
			checkpoint_dir=effective_checkpoint_dir,
			input_path=input_path,
			output_path=output_path,
			processed_articles=len(articles),
			total_articles=len(articles),
			articles=embedded_articles,
			settings=settings,
		)
		print(f"Final checkpoint snapshot: {checkpoint_path}")

	print(f"\nSaved {len(embedded_articles)} articles with embeddings to {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate embeddings for article JSON files using Google Generative AI (Gemini)"
	)
	parser.add_argument("input", type=Path, help="Input JSON file with articles to embed")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Output JSON file (default: input_embedded.json)",
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
		help="Field name to extract text from (default: content)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=16,
		help="Number of texts per embedding request (default: 16)",
	)
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=None,
		help="Directory to store checkpoint snapshots (default: output/embeddings/checkpoints/<input_stem>)",
	)
	parser.add_argument(
		"--checkpoint-every",
		type=int,
		default=0,
		help="Save output every N processed articles (default: 0, disabled)",
	)

	args = parser.parse_args()
	_load_env_file(Path.cwd() / ".env")
	api_key = _resolve_api_key(args.api_key)

	if not api_key:
		print(
			"Error: Google API key required. Use --api-key or set GOOGLE_API_KEY / GEMINI_API_KEY in .env",
			file=sys.stderr,
		)
		sys.exit(1)

	input_path = args.input.resolve()
	if not input_path.exists():
		print(f"Error: Input file not found: {input_path}", file=sys.stderr)
		sys.exit(1)

	if args.output:
		output_path = args.output.resolve()
	else:
		output_path = Path.cwd() / "output" / "embeddings" / f"{input_path.stem}_embedded.json"

	print(f"Model: {args.model}")
	print(f"Text field: {args.field}\n")
	embed_articles(
		input_path,
		output_path,
		api_key,
		args.model,
		args.field,
		max(0, args.checkpoint_every),
		max(1, args.batch_size),
		args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
	)


if __name__ == "__main__":
	main()
