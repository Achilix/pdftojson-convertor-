import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from semantic_chunk import (
	DEFAULT_MODEL,
	_load_env_file,
	_resolve_api_key,
	find_latest_checkpoint,
	load_checkpoint,
	semantic_chunk_articles,
)


def _pick_setting(checkpoint_settings: Dict[str, Any], key: str, cli_value: Any) -> Any:
	if cli_value is not None:
		return cli_value
	return checkpoint_settings.get(key)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Resume semantic chunking from the latest checkpoint."
	)
	parser.add_argument("input", type=Path, help="Original input articles JSON file")
	parser.add_argument(
		"--checkpoint-dir",
		type=Path,
		default=None,
		help="Checkpoint directory (default: output/chunks/checkpoints/<input_stem>)",
	)
	parser.add_argument(
		"-k",
		"--api-key",
		default=None,
		help="Google API key (or set GOOGLE_API_KEY / GEMINI_API_KEY in .env)",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Override output JSON path (default: from checkpoint)",
	)
	parser.add_argument(
		"-m",
		"--model",
		default=None,
		help=f"Override embedding model (default: from checkpoint, fallback {DEFAULT_MODEL})",
	)
	parser.add_argument("-f", "--field", default=None, help="Override article text field")
	parser.add_argument("--target-chars", type=int, default=None, help="Override preferred chunk size")
	parser.add_argument("--max-chars", type=int, default=None, help="Override hard max chunk size")
	parser.add_argument("--similarity-threshold", type=float, default=None, help="Override similarity threshold")
	parser.add_argument("--max-retries", type=int, default=None, help="Override max retries")
	parser.add_argument("--pause", type=float, default=None, help="Override pause seconds")
	parser.add_argument(
		"--checkpoint-every",
		type=int,
		default=None,
		help="Override checkpoint save cadence",
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

	if args.checkpoint_dir:
		checkpoint_dir = args.checkpoint_dir.resolve()
	else:
		checkpoint_dir = Path.cwd() / "output" / "chunks" / "checkpoints" / input_path.stem

	checkpoint_path = find_latest_checkpoint(checkpoint_dir, input_path.stem)
	checkpoint = load_checkpoint(checkpoint_path)
	settings = checkpoint.get("settings") if isinstance(checkpoint.get("settings"), dict) else {}

	next_article_index = int(checkpoint.get("next_article_index", 0))
	chunks = checkpoint.get("chunks") if isinstance(checkpoint.get("chunks"), list) else []

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
	checkpoint_every = int(_pick_setting(settings, "checkpoint_every", args.checkpoint_every) or 5)

	print(f"Using checkpoint: {checkpoint_path}")
	print(f"Resuming from article index: {next_article_index + 1}")
	print(f"Loaded partial chunks: {len(chunks)}")

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
		checkpoint_dir=checkpoint_dir,
		checkpoint_every=max(0, checkpoint_every),
		start_article_index=max(0, next_article_index),
		seed_chunks=chunks,
	)


if __name__ == "__main__":
	main()
