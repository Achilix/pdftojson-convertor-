import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
	import google.genai as genai
	import numpy as np
except ImportError as exc:
	raise RuntimeError("Missing dependencies. Install with: pip install google-genai numpy") from exc


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


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
	"""
	Calculate cosine similarity between two vectors.
	
	Args:
		vec1: First embedding vector
		vec2: Second embedding vector
		
	Returns:
		Cosine similarity score (0 to 1)
	"""
	v1 = np.array(vec1, dtype=np.float32)
	v2 = np.array(vec2, dtype=np.float32)
	
	norm1 = np.linalg.norm(v1)
	norm2 = np.linalg.norm(v2)
	
	if norm1 == 0 or norm2 == 0:
		return 0.0
	
	return float(np.dot(v1, v2) / (norm1 * norm2))


def embed_query(query: str, api_key: str, model: str = DEFAULT_MODEL) -> List[float]:
	"""
	Generate embedding for a query text.
	
	Args:
		query: Query text to embed
		api_key: Google API key
		model: Embedding model name
		
	Returns:
		Embedding vector (list of floats)
	"""
	if not query or not query.strip():
		raise ValueError("Query cannot be empty")
	
	client = genai.Client(api_key=api_key)
	
	try:
		response = client.models.embed_content(
			model=f"models/{model}" if not model.startswith("models/") else model,
			contents=query
		)
		
		# Extract embedding from response
		if response.embeddings and len(response.embeddings) > 0:
			embedding = response.embeddings[0]
			
			# ContentEmbedding object has .values attribute
			if hasattr(embedding, 'values'):
				return list(embedding.values)
			elif isinstance(embedding, (list, tuple)):
				return list(embedding)
			else:
				raise RuntimeError(f"Unexpected embedding type: {type(embedding)}")
		else:
			raise RuntimeError("No embedding returned")
			
	except Exception as e:
		raise RuntimeError(f"Error embedding query: {e}")


def search_articles(
	embedded_file: Path,
	query: str,
	api_key: str,
	model: str = DEFAULT_MODEL,
	top_k: int = 5,
	threshold: float = 0.0
) -> List[Tuple[Dict[str, Any], float]]:
	"""
	Search articles by cosine similarity.
	
	Args:
		embedded_file: Path to JSON file with embedded articles
		query: Search query text
		api_key: Google API key
		model: Embedding model name
		top_k: Number of top results to return
		threshold: Minimum similarity score (0.0-1.0)
		
	Returns:
		List of tuples (article, similarity_score) sorted by score descending
	"""
	# Load embedded articles
	with embedded_file.open("r", encoding="utf-8") as f:
		articles = json.load(f)
	
	print(f"Loaded {len(articles)} embedded articles")
	
	# Embed the query
	print(f"Embedding query: {query[:100]}...")
	query_embedding = embed_query(query, api_key, model)
	print(f"Query embedding: {len(query_embedding)} dims")
	
	# Calculate similarities
	results = []
	for i, article in enumerate(articles, 1):
		if not isinstance(article, dict):
			continue
		
		# Get embedding from article
		article_embedding = article.get("embedding")
		if not article_embedding:
			continue
		
		# Handle double-nested format: [[['values', [floats...]], [...]]]
		# Extract the actual float array
		if isinstance(article_embedding, (list, tuple)) and len(article_embedding) > 0:
			# First level: article_embedding is [['values', [...]], [...]]
			first_elem = article_embedding[0]
			if isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
				# Check if this is the ['values', [...]] format
				if first_elem[0] == 'values' and isinstance(first_elem[1], (list, tuple)):
					# Extract the float array
					article_embedding = first_elem[1]
		
		# Calculate cosine similarity
		try:
			similarity = cosine_similarity(query_embedding, article_embedding)
		except Exception as e:
			continue
		
		if similarity >= threshold:
			results.append((article, similarity))
		
		if i % 100 == 0:
			print(f"  Processed {i}/{len(articles)} articles...")
	
	# Sort by similarity descending
	results.sort(key=lambda x: x[1], reverse=True)
	
	return results[:top_k]


def format_result(article: Dict[str, Any], similarity: float, rank: int) -> str:
	"""Format a search result for display."""
	article_num = article.get("article_number") or article.get("article", "Unknown")
	content = article.get("content", "")[:200]
	
	# Handle pages field - can be string like "158" or "158-159" or dict
	pages = article.get("pages", "?")
	if isinstance(pages, dict):
		page_start = pages.get("start", "?")
		page_end = pages.get("end", "?")
		pages_str = f"{page_start}-{page_end}"
	else:
		pages_str = str(pages) if pages else "?"
	
	# Only show content line if there's actual content
	content_line = f"Content: {content}...\n" if content else ""
	
	return f"""\
#{rank} - Similarity: {similarity:.4f} ({similarity*100:.2f}%)
Article: {article_num}
Page(s): {pages_str}
{content_line}"""


def main():
	parser = argparse.ArgumentParser(
		description="Search embedded articles using cosine similarity"
	)
	parser.add_argument(
		"embedded_file",
		type=Path,
		help="Path to JSON file with embedded articles (from embed.py)"
	)
	parser.add_argument(
		"-q", "--query",
		default=None,
		help="Search query text (if not provided, will prompt interactively)"
	)
	parser.add_argument(
		"-k", "--top-k",
		type=int,
		default=5,
		help="Number of top results to return (default: 5)"
	)
	parser.add_argument(
		"-t", "--threshold",
		type=float,
		default=0.0,
		help="Minimum similarity score 0.0-1.0 (default: 0.0)"
	)
	parser.add_argument(
		"-a", "--api-key",
		default=None,
		help="Google API key (or set GOOGLE_API_KEY / GEMINI_API_KEY in .env)"
	)
	parser.add_argument(
		"-m", "--model",
		default=DEFAULT_MODEL,
		help=f"Embedding model name (default: {DEFAULT_MODEL})"
	)
	
	args = parser.parse_args()
	
	# Validate input file
	embedded_file = args.embedded_file.resolve()
	if not embedded_file.exists():
		print(f"Error: File not found: {embedded_file}", file=sys.stderr)
		sys.exit(1)
	
	# Get API key
	_load_env_file(Path.cwd() / ".env")
	api_key = _resolve_api_key(args.api_key)
	if not api_key:
		print("Error: Google API key required. Use --api-key or set GOOGLE_API_KEY / GEMINI_API_KEY in .env", file=sys.stderr)
		sys.exit(1)
	
	# Get query
	query = args.query
	if not query:
		print("Enter your search query (or 'quit' to exit):")
		query = input("> ").strip()
		if query.lower() == "quit":
			sys.exit(0)
	
	if not query:
		print("Error: Empty query", file=sys.stderr)
		sys.exit(1)
	
	# Search
	print(f"\nSearching with top_k={args.top_k}, threshold={args.threshold}\n")
	results = search_articles(
		embedded_file,
		query,
		api_key,
		args.model,
		args.top_k,
		args.threshold
	)
	
	# Display results
	if not results:
		print(f"No results found with similarity >= {args.threshold}")
		return
	
	print(f"\nFound {len(results)} result(s):\n")
	for rank, (article, similarity) in enumerate(results, 1):
		print(format_result(article, similarity, rank))
		print("-" * 80)


if __name__ == "__main__":
	main()
