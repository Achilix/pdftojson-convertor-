import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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

	return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def embed_text(text: str, api_key: str, model: str = DEFAULT_MODEL) -> List[float]:
	"""Generate an embedding vector for text using Gemini."""
	if not text or not text.strip():
		return []

	if not api_key:
		raise ValueError("API key is required. Set GOOGLE_API_KEY or pass --api-key")

	client = genai.Client(api_key=api_key)

	try:
		response = client.models.embed_content(
			model=f"models/{model}" if not model.startswith("models/") else model,
			contents=text,
		)

		if not response.embeddings:
			print(f"Warning: No embedding returned for text: {text[:50]}...")
			return []

		embedding = response.embeddings[0]
		if hasattr(embedding, "values"):
			return list(embedding.values)
		if isinstance(embedding, (list, tuple)):
			return list(embedding)

		raise RuntimeError(f"Unexpected embedding type: {type(embedding)}")
	except Exception as exc:
		raise RuntimeError(f"Error calling Google Generative AI embedding API: {exc}") from exc


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


def embed_articles(
	input_path: Path,
	output_path: Path,
	api_key: str,
	model: str = DEFAULT_MODEL,
	text_field: str = "content",
) -> None:
	"""Load articles from JSON, embed content, and save results."""
	articles = _load_articles_json(input_path)
	print(f"Loading {len(articles)} articles from {input_path.name}")

	embedded_articles: List[Dict[str, Any]] = []
	for i, article in enumerate(articles, 1):
		text_to_embed = str(article.get(text_field, "") or "")
		if not text_to_embed.strip():
			print(f"Warning: Article {i} has no '{text_field}' field, skipping embedding")
			embedded_articles.append(article)
			continue

		label = article.get("article_number") or article.get("article") or "unknown"
		print(f"[{i}/{len(articles)}] Embedding article {label}...", end=" ", flush=True)
		try:
			embedding = embed_text(text_to_embed, api_key, model)
			article_with_embedding = dict(article)
			article_with_embedding["embedding"] = embedding
			embedded_articles.append(article_with_embedding)
			print(f"OK ({len(embedding)} dims)")
		except Exception as exc:
			print(f"FAILED: {str(exc)[:120]}")
			embedded_articles.append(article)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(embedded_articles, handle, ensure_ascii=False, indent=2)

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
		output_path = input_path.parent / f"{input_path.stem}_embedded.json"

	print(f"Model: {args.model}")
	print(f"Text field: {args.field}\n")
	embed_articles(input_path, output_path, api_key, args.model, args.field)


if __name__ == "__main__":
	main()
