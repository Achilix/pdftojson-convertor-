import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
	import google.genai as genai
except ImportError as exc:
	raise RuntimeError("Missing google-genai. Install it with: pip install google-genai") from exc


DEFAULT_MODEL = "gemini-embedding-001"  # Google's embedding model
DEFAULT_API_KEY = "AIzaSyAYjoO7SFkrtpeGCIwLNLxMIUbVOPJXCZU"


def embed_text(text: str, api_key: str, model: str = DEFAULT_MODEL) -> List[float]:
	"""
	Generate embeddings for text using Google Generative AI (Gemini) with google-genai.
	
	Args:
		text: Text to embed
		api_key: Google API key
		model: Embedding model name (default: text-embedding-004)
		
	Returns:
		List of floats representing the embedding vector
	"""
	if not text or not text.strip():
		return []
	
	if not api_key:
		raise ValueError("API key is required. Set GOOGLE_API_KEY or pass --api-key")
	
	client = genai.Client(api_key=api_key)
	
	try:
		response = client.models.embed_content(
			model=f"models/{model}" if not model.startswith("models/") else model,
			contents=text
		)
		
		if response.embeddings:
			return list(response.embeddings[0])
		else:
			print(f"Warning: No embedding returned for text: {text[:50]}...")
			return []
			
	except Exception as e:
		raise RuntimeError(f"Error calling Google Generative AI embedding API: {str(e)}")


def embed_articles(
	input_path: Path,
	output_path: Path,
	api_key: str,
	model: str = DEFAULT_MODEL,
	text_field: str = "content"
) -> None:
	"""
	Load articles from JSON, embed their content, and save results.
	
	Args:
		input_path: Path to input JSON file with articles
		output_path: Path to output JSON file with embeddings
		api_key: Google API key
		model: Embedding model name
		text_field: Field name to extract text from (default: 'content')
	"""
	# Load input articles
	with input_path.open("r", encoding="utf-8") as f:
		articles = json.load(f)
	
	if not isinstance(articles, list):
		raise ValueError(f"Expected JSON array in {input_path}, got {type(articles)}")
	
	print(f"Loading {len(articles)} articles from {input_path.name}")
	
	# Embed each article
	embedded_articles = []
	for i, article in enumerate(articles, 1):
		if not isinstance(article, dict):
			print(f"Warning: Article {i} is not a dict, skipping")
			continue
		
		# Extract text to embed
		text_to_embed = article.get(text_field, "")
		if not text_to_embed:
			print(f"Warning: Article {i} has no '{text_field}' field, skipping embedding")
			embedded_articles.append(article)
			continue
		
		# Generate embedding
		print(f"[{i}/{len(articles)}] Embedding article {article.get('article', 'unknown')}...", end=" ", flush=True)
		try:
			embedding = embed_text(text_to_embed, api_key, model)
			article_with_embedding = dict(article)
			article_with_embedding["embedding"] = embedding
			embedded_articles.append(article_with_embedding)
			print(f"OK ({len(embedding)} dims)")
		except Exception as e:
			print(f"FAILED: {str(e)[:100]}")
			embedded_articles.append(article)
	
	# Save output
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(embedded_articles, f, ensure_ascii=False, indent=2)
	
	print(f"\n✓ Saved {len(embedded_articles)} articles with embeddings to {output_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Generate embeddings for article JSON files using Google Generative AI (Gemini)"
	)
	parser.add_argument(
		"input",
		type=Path,
		help="Input JSON file with articles to embed"
	)
	parser.add_argument(
		"-o", "--output",
		type=Path,
		default=None,
		help="Output JSON file (default: input_embedded.json)"
	)
	parser.add_argument(
		"-k", "--api-key",
		default=None,
		help="Google API key (or set GOOGLE_API_KEY environment variable)"
	)
	parser.add_argument(
		"-m", "--model",
		default=DEFAULT_MODEL,
		help=f"Embedding model name (default: {DEFAULT_MODEL})"
	)
	parser.add_argument(
		"-f", "--field",
		default="content",
		help="Field name to extract text from (default: content)"
	)
	
	args = parser.parse_args()
	
	# Get API key from args or environment
	api_key = args.api_key
	if not api_key:
		import os
		api_key = os.environ.get("GOOGLE_API_KEY")
	
	# Fall back to default API key if not provided
	if not api_key:
		api_key = DEFAULT_API_KEY
	
	if not api_key:
		print("Error: Google API key required. Use --api-key or set GOOGLE_API_KEY environment variable", file=sys.stderr)
		sys.exit(1)
	
	# Validate input
	input_path = args.input.resolve()
	if not input_path.exists():
		print(f"Error: Input file not found: {input_path}", file=sys.stderr)
		sys.exit(1)
	
	# Determine output path
	if args.output:
		output_path = args.output.resolve()
	else:
		stem = input_path.stem
		output_path = input_path.parent / f"{stem}_embedded.json"
	
	print(f"Model: {args.model}")
	print(f"Text field: {args.field}\n")
	
	embed_articles(input_path, output_path, api_key, args.model, args.field)


if __name__ == "__main__":
	main()
