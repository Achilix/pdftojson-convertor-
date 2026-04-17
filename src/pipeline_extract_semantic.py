import argparse
import os
import sys
from pathlib import Path

from app import process_pdf_to_outputs
from semantic_chunk import DEFAULT_MODEL, semantic_chunk_articles


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


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run extraction and semantic chunking in one command."
	)
	parser.add_argument("pdf", type=Path, help="Input PDF file path")
	parser.add_argument(
		"--max-pages",
		type=int,
		default=50,
		help="Process only the first N pages during extraction (default: 50)",
	)
	parser.add_argument(
		"--extract-output-dir",
		type=Path,
		default=Path("output/extracted"),
		help="Directory for extracted articles JSON/CSV",
	)
	parser.add_argument(
		"--chunk-output",
		type=Path,
		default=None,
		help="Output semantic chunks JSON file (default: output/chunks/<extract_stem>_semantic_chunks.json)",
	)
	parser.add_argument(
		"--api-key",
		default=None,
		help="Google API key (or set GOOGLE_API_KEY / GEMINI_API_KEY in .env)",
	)
	parser.add_argument(
		"--model",
		default=DEFAULT_MODEL,
		help=f"Embedding model name (default: {DEFAULT_MODEL})",
	)
	parser.add_argument(
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
		help="Pause in seconds between embedding calls (default: 0.15)",
	)
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=None,
		help="Directory for semantic chunk checkpoints (default: output/chunks/checkpoints/<extract_stem>)",
	)
	parser.add_argument(
		"--checkpoint-every",
		type=int,
		default=5,
		help="Save semantic checkpoint every N processed articles (default: 5)",
	)
	args = parser.parse_args()

	pdf_path = args.pdf.resolve()
	if not pdf_path.exists():
		print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
		sys.exit(1)

	_load_env_file(Path.cwd() / ".env")
	api_key = _resolve_api_key(args.api_key)
	if not api_key:
		print(
			"Error: Google API key required. Use --api-key or set GOOGLE_API_KEY / GEMINI_API_KEY in .env",
			file=sys.stderr,
		)
		sys.exit(1)

	extract_output_dir = args.extract_output_dir.resolve()
	json_file, csv_file = process_pdf_to_outputs(
		pdf_path=pdf_path,
		output_dir=extract_output_dir,
		max_pages=max(0, args.max_pages),
	)
	print(f"Extraction done -> {json_file}")
	print(f"Extraction CSV -> {csv_file}")

	if args.chunk_output:
		chunk_output_path = args.chunk_output.resolve()
	else:
		chunk_output_path = Path.cwd() / "output" / "chunks" / f"{json_file.stem}_semantic_chunks.json"

	if args.checkpoint_dir:
		checkpoint_dir = args.checkpoint_dir.resolve()
	else:
		checkpoint_dir = Path.cwd() / "output" / "chunks" / "checkpoints" / json_file.stem

	semantic_chunk_articles(
		input_path=json_file,
		output_path=chunk_output_path,
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

	print("Pipeline complete.")
	print(f"Semantic chunks JSON -> {chunk_output_path}")
	print(f"Semantic chunks CSV -> {chunk_output_path.with_suffix('.csv')}")


if __name__ == "__main__":
	main()
