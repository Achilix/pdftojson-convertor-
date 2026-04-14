import argparse
import csv
import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Any, Dict, List

try:
	import requests
except ImportError as exc:
	raise RuntimeError("Missing requests. Install it with: pip install requests") from exc


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:latest"


class OutOfCreditsError(RuntimeError):
	"""Raised when the Ollama endpoint rejects access."""


class RateLimitError(RuntimeError):
	"""Raised when Ollama rate limits persist after retries."""


class ModelNotFoundError(RuntimeError):
	"""Raised when the requested Ollama model is not installed on the server."""


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


def _get_ollama_base_url() -> str:
	return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _get_ollama_chat_url() -> str:
	return f"{_get_ollama_base_url()}/api/chat"


def _load_records(input_path: Path) -> List[Dict[str, Any]]:
	with input_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if not isinstance(payload, list):
		raise ValueError(f"Expected a JSON array in {input_path}")

	records: List[Dict[str, Any]] = []
	for item in payload:
		if not isinstance(item, dict):
			raise ValueError(f"Expected each JSON item to be an object in {input_path}")
		records.append(dict(item))

	return records


def _chunk_records(records: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
	return [records[index:index + chunk_size] for index in range(0, len(records), chunk_size)]


def _trim_article(article: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"id": article.get("id"),
		"article_number": article.get("article_number"),
		"document_name": article.get("document_name"),
		"livre": article.get("livre", ""),
		"titre": article.get("titre", ""),
		"chapitre": article.get("chapitre", ""),
		"section": article.get("section", ""),
		"sous_section": article.get("sous_section", ""),
		"pages": article.get("pages", ""),
		"content": article.get("content", ""),
	}


def _build_article_prompt(article: Dict[str, Any]) -> str:
	article_text = json.dumps(article, ensure_ascii=False, indent=2)
	return (
		"You are generating study questions from legal article text. "
		"Write exactly 10 concise questions in French that test understanding of the article. "
		"Do not answer the questions. Do not repeat the article text. Keep the questions grounded in the provided content. "
		"Return valid JSON only, with this structure: {\"id\": ..., \"article_number\": ..., \"questions\": [\"...\", ... 10 items ...]}. "
		"Use the article id exactly as provided.\n\n"
		"Article:\n"
		f"{article_text}"
	)


def _get_ollama_models() -> List[str]:
	try:
		response = requests.get(f"{_get_ollama_base_url()}/api/tags", timeout=30)
		response.raise_for_status()
		payload = response.json()
		models = payload.get("models", []) if isinstance(payload, dict) else []
		return [str(item.get("name", "")).strip() for item in models if str(item.get("name", "")).strip()]
	except Exception:
		return []


def _repair_invalid_json_escapes(text: str) -> str:
	"""Repair invalid backslash escapes so JSON parsing is more tolerant of model output."""
	out: List[str] = []
	in_string = False
	index = 0
	while index < len(text):
		char = text[index]

		if not in_string:
			out.append(char)
			if char == '"':
				in_string = True
			index += 1
			continue

		if char == '"':
			out.append(char)
			in_string = False
			index += 1
			continue

		if char != "\\":
			out.append(char)
			index += 1
			continue

		if index + 1 >= len(text):
			out.append("\\\\")
			index += 1
			continue

		next_char = text[index + 1]
		if next_char in '"\\/bfnrt':
			out.append("\\")
			out.append(next_char)
			index += 2
			continue

		if next_char == "u":
			hex_digits = text[index + 2:index + 6]
			if len(hex_digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_digits):
				out.append(text[index:index + 6])
				index += 6
				continue

		# Invalid escape (for example \l): preserve it as a literal backslash + char.
		out.append("\\\\")
		index += 1

	return "".join(out)


def _parse_ollama_json(content: str) -> Dict[str, Any]:
	clean_content = content.strip()
	candidates: List[str] = [clean_content]

	if clean_content.startswith("```"):
		fence_lines = clean_content.splitlines()
		if len(fence_lines) >= 3 and fence_lines[-1].strip() == "```":
			candidates.append("\n".join(fence_lines[1:-1]).strip())

	start = clean_content.find("{")
	end = clean_content.rfind("}")
	if start != -1 and end != -1 and end > start:
		candidates.append(clean_content[start:end + 1])

	last_error: Exception | None = None
	for candidate in candidates:
		if not candidate:
			continue
		for attempt in (candidate, _repair_invalid_json_escapes(candidate)):
			try:
				parsed = json.loads(attempt)
				if not isinstance(parsed, dict):
					raise ValueError("Model JSON payload must be an object.")
				return parsed
			except Exception as exc:
				last_error = exc

	raise RuntimeError(f"Model did not return valid JSON: {clean_content[:500]}") from last_error


def _call_ollama(model: str, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
	headers = {
		"Content-Type": "application/json",
	}
	payload = {
		"model": model,
		"messages": [
			{"role": "system", "content": "You output strict JSON only."},
			{"role": "user", "content": prompt},
		],
		"stream": False,
		"options": {
			"temperature": temperature,
		},
	}

	response = requests.post(_get_ollama_chat_url(), headers=headers, json=payload, timeout=120)
	if response.status_code in (401, 403):
		raise OutOfCreditsError(
			"Ollama access was rejected by the endpoint."
		)
	if response.status_code == 404:
		available_models = _get_ollama_models()
		available_hint = f" Available models: {available_models}" if available_models else ""
		raise ModelNotFoundError(
			f"Model '{model}' was not found on the Ollama server.{available_hint}"
		)
	response.raise_for_status()
	data = response.json()

	message = data.get("message") or {}
	content = message.get("content", "")
	if not content:
		raise RuntimeError(f"Ollama returned empty content: {data}")

	return _parse_ollama_json(content)


def _normalize_questions(questions: Any) -> List[str]:
	if not isinstance(questions, list):
		return []
	return [str(question).strip() for question in questions if str(question).strip()]


def _merge_unique_questions(existing: List[str], additional: List[str], limit: int) -> List[str]:
	merged: List[str] = []
	seen: set[str] = set()
	for question in existing + additional:
		normalized = question.strip()
		if not normalized:
			continue
		key = normalized.casefold()
		if key in seen:
			continue
		seen.add(key)
		merged.append(normalized)
		if len(merged) >= limit:
			break
	return merged


def _build_top_up_prompt(article: Dict[str, Any], existing_questions: List[str], needed: int) -> str:
	article_text = json.dumps(article, ensure_ascii=False, indent=2)
	existing_text = json.dumps(existing_questions, ensure_ascii=False, indent=2)
	return (
		"You are continuing question generation for one legal article. "
		f"Generate exactly {needed} additional concise questions in French. "
		"The new questions must be different from the existing ones. "
		"Do not answer questions. Return valid JSON only with this structure: "
		"{\"questions\": [\"...\", ...]}.\n\n"
		"Article:\n"
		f"{article_text}\n\n"
		"Existing questions (do not repeat):\n"
		f"{existing_text}"
	)


def _normalize_single_output(batch_output: Dict[str, Any], questions_per_article: int) -> Dict[str, Any]:
	if not isinstance(batch_output, dict):
		raise ValueError("Expected output JSON to be an object.")

	normalized_questions = _normalize_questions(batch_output.get("questions", []))
	if len(normalized_questions) > questions_per_article:
		normalized_questions = normalized_questions[:questions_per_article]

	return {
		"id": batch_output.get("id"),
		"article_number": batch_output.get("article_number"),
		"questions": normalized_questions,
	}


def _generate_questions_for_article(
	model: str,
	article: Dict[str, Any],
	questions_per_article: int,
	max_retries: int,
) -> Dict[str, Any]:
	prompt = _build_article_prompt(article)
	last_error: Exception | None = None
	next_retry_delay = 2.0
	best_partial_questions: List[str] = []
	for attempt in range(1, max_retries + 1):
		try:
			result = _call_ollama(model=model, prompt=prompt)
			normalized = _normalize_single_output(result, questions_per_article=questions_per_article)
			questions = _merge_unique_questions([], normalized.get("questions", []), questions_per_article)

			if len(questions) < questions_per_article:
				missing = questions_per_article - len(questions)
				top_up_prompt = _build_top_up_prompt(article, questions, missing)
				top_up_result = _call_ollama(model=model, prompt=top_up_prompt)
				top_up_questions = _normalize_questions(top_up_result.get("questions", []))
				questions = _merge_unique_questions(questions, top_up_questions, questions_per_article)

			if len(questions) < questions_per_article:
				best_partial_questions = _merge_unique_questions(best_partial_questions, questions, questions_per_article)
				raise ValueError(
					f"Article id {article.get('id')} returned {len(questions)} questions; expected {questions_per_article}."
				)

			return {
				"id": normalized.get("id", article.get("id")),
				"article_number": normalized.get("article_number", article.get("article_number")),
				"questions": questions,
			}
		except Exception as exc:
			if isinstance(exc, ModelNotFoundError):
				raise
			if isinstance(exc, requests.exceptions.HTTPError):
				status_code = exc.response.status_code if exc.response is not None else None
				if status_code == 429:
					retry_after = None
					if exc.response is not None:
						retry_after_value = exc.response.headers.get("Retry-After", "").strip()
						if retry_after_value:
							try:
								retry_after = float(retry_after_value)
							except ValueError:
								retry_after = None

					if attempt < max_retries:
						delay = retry_after if retry_after is not None else next_retry_delay
						delay = max(1.0, min(delay, 60.0))
						print(
							f"Rate limited (429) on article id {article.get('id')}. "
							f"Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})."
						)
						time.sleep(delay)
						next_retry_delay = min(next_retry_delay * 2, 60.0)
						continue

					raise RateLimitError(
						"The Ollama endpoint returned 429 Too Many Requests repeatedly. "
						"Increase --pause-seconds (try 1-3), lower request frequency, or retry later."
					) from exc
			last_error = exc
			if attempt < max_retries:
				time.sleep(2)
				continue
			if len(best_partial_questions) >= questions_per_article:
				return {
					"id": article.get("id"),
					"article_number": article.get("article_number"),
					"questions": best_partial_questions[:questions_per_article],
				}
			raise RuntimeError(
				f"Failed to generate questions for article id {article.get('id')} after {max_retries} attempts: {exc}"
			) from exc

	raise RuntimeError(f"Failed to generate questions for article id {article.get('id')}: {last_error}")


def _write_json(output_path: Path, payload: Any) -> None:
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(output_path: Path, articles: List[Dict[str, Any]], questions_per_article: int) -> None:
	fields = ["article_id"] + [f"question_{i}" for i in range(1, questions_per_article + 1)]
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fields, restval="")
		writer.writeheader()
		for article in articles:
			row = {"article_id": article.get("article_id", "")}
			for index, question in enumerate(article.get("questions", []), start=1):
				if index > questions_per_article:
					break
				row[f"question_{index}"] = question
			writer.writerow(row)


def _write_question_outputs(
	input_path: Path,
	output_dir: Path,
	model: str,
	questions_per_article: int,
	chunk_size: int,
	articles: List[Dict[str, Any]],
	status: str,
	stopped_reason: str = "",
) -> None:
	output_payload = {
		"source_file": input_path.name,
		"model": model,
		"questions_per_article": questions_per_article,
		"chunk_size": chunk_size,
		"total_articles": len(articles),
		"status": status,
		"stopped_reason": stopped_reason,
		"articles": articles,
	}
	json_path = output_dir / f"{input_path.stem}_questions.json"
	_write_json(json_path, output_payload)
	print(f"Wrote {json_path} ({len(articles)} articles)")

	csv_path = output_dir / f"{input_path.stem}_questions.csv"
	_write_csv(csv_path, articles, questions_per_article)
	print(f"Wrote {csv_path} ({len(articles)} articles)")


def _write_batch_outputs(
	input_path: Path,
	batch_dir: Path,
	batch_index: int,
	model: str,
	questions_per_article: int,
	chunk_size: int,
	articles: List[Dict[str, Any]],
) -> None:
	batch_payload = {
		"source_file": input_path.name,
		"model": model,
		"questions_per_article": questions_per_article,
		"chunk_size": chunk_size,
		"batch_index": batch_index,
		"batch_articles": len(articles),
		"articles": articles,
	}
	batch_dir.mkdir(parents=True, exist_ok=True)
	json_path = batch_dir / f"{input_path.stem}_batch_{batch_index:03d}.json"
	_write_json(json_path, batch_payload)
	print(f"Saved batch {batch_index} -> {json_path}")

	csv_path = batch_dir / f"{input_path.stem}_batch_{batch_index:03d}.csv"
	_write_csv(csv_path, articles, questions_per_article)
	print(f"Saved batch {batch_index} -> {csv_path}")


def _print_progress(
	processed: int,
	total: int,
	batch_index: int,
	total_batches: int,
	last_article_id: Any,
	started_at: float,
) -> None:
	percent = (processed / total * 100.0) if total else 0.0
	elapsed = time.time() - started_at
	line = (
		f"\rProgress: {processed}/{total} ({percent:5.1f}%)"
		f" | Batch {batch_index}/{total_batches}"
		f" | Last article id: {last_article_id}"
		f" | Elapsed: {elapsed:6.1f}s"
	)
	sys.stdout.write(line)
	sys.stdout.flush()


def process_file(
	input_path: Path,
	output_dir: Path,
	model: str,
	chunk_size: int,
	questions_per_article: int,
	pause_seconds: float,
	max_retries: int,
	min_total_questions: int,
	max_total_questions: int,
) -> None:
	if min_total_questions < 1:
		raise ValueError("--min-total-questions must be >= 1")
	if max_total_questions < min_total_questions:
		raise ValueError("--max-total-questions must be >= --min-total-questions")
	if questions_per_article < 1:
		raise ValueError("--questions-per-article must be >= 1")

	records = _load_records(input_path)
	if not records:
		print(f"Skipping empty file: {input_path}")
		return

	max_articles_allowed = max_total_questions // questions_per_article
	if max_articles_allowed < 1:
		raise ValueError(
			"The configured max total questions is too low for the current questions per article. "
			"Increase --max-total-questions or lower --questions-per-article."
		)
	min_articles_required = math.ceil(min_total_questions / questions_per_article)
	if len(records) < min_articles_required:
		raise ValueError(
			f"Input has only {len(records)} articles, which cannot satisfy at least "
			f"{min_total_questions} questions with {questions_per_article} per article."
		)

	target_articles = min(len(records), max_articles_allowed)

	output_dir.mkdir(parents=True, exist_ok=True)
	chunks = _chunk_records(records, chunk_size)
	all_articles: List[Dict[str, Any]] = []
	status = "completed"
	stop_reason = ""
	total_articles = target_articles
	processed_articles = 0
	skipped_articles = 0
	started_at = time.time()
	batch_dir = output_dir / f"{input_path.stem}_batches"
	json_output_path = output_dir / f"{input_path.stem}_questions.json"
	csv_output_path = output_dir / f"{input_path.stem}_questions.csv"
	print(f"Source document file: {input_path}")
	print(f"Output JSON: {json_output_path}")
	print(f"Output CSV: {csv_output_path}")
	print(
		f"Question budget per document: {min_total_questions}-{max_total_questions}. "
		f"Target successful articles: {total_articles} (about {total_articles * questions_per_article} questions)."
	)

	for chunk_index, chunk in enumerate(chunks, start=1):
		if processed_articles >= target_articles:
			break
		print(f"Processing batch {chunk_index}/{len(chunks)} ({len(chunk)} articles)")
		batch_articles: List[Dict[str, Any]] = []
		for article in chunk:
			if processed_articles >= target_articles:
				break
			trimmed = _trim_article(article)
			try:
				generated = _generate_questions_for_article(
					model=model,
					article=trimmed,
					questions_per_article=questions_per_article,
					max_retries=max_retries,
				)
			except OutOfCreditsError as exc:
				status = "stopped"
				stop_reason = str(exc)
				print(f"Stopping early at article id {trimmed.get('id')}: {stop_reason}")
				break
			except ModelNotFoundError as exc:
				status = "stopped"
				stop_reason = str(exc)
				print(f"Stopping early at article id {trimmed.get('id')}: {stop_reason}")
				break
			except RateLimitError as exc:
				status = "stopped"
				stop_reason = str(exc)
				print(f"Stopping early at article id {trimmed.get('id')}: {stop_reason}")
				break
			except KeyboardInterrupt:
				status = "stopped"
				stop_reason = "Interrupted by user (Ctrl+C)."
				print("Interrupted by user. Writing partial results.")
				break
			except Exception as exc:
				skipped_articles += 1
				print(f"Skipping article id {trimmed.get('id')} after retries: {exc}")
				continue
			all_articles.append(
				{
					"article_id": generated.get("id", trimmed.get("id", "")),
					"questions": generated.get("questions", []),
				}
			)
			batch_articles.append(
				{
					"article_id": generated.get("id", trimmed.get("id", "")),
					"questions": generated.get("questions", []),
				}
			)
			processed_articles += 1
			_print_progress(
				processed=processed_articles,
				total=total_articles,
				batch_index=chunk_index,
				total_batches=len(chunks),
				last_article_id=trimmed.get("id"),
				started_at=started_at,
			)

			if pause_seconds > 0:
				time.sleep(pause_seconds)

		if status == "stopped":
			print()
			break

		if batch_articles:
			print()
			_write_batch_outputs(
				input_path=input_path,
				batch_dir=batch_dir,
				batch_index=chunk_index,
				model=model,
				questions_per_article=questions_per_article,
				chunk_size=chunk_size,
				articles=batch_articles,
			)
		else:
			print("No successful generations in this batch.")

	if status != "stopped" and processed_articles < min_articles_required:
		status = "stopped"
		stop_reason = (
			f"Could only generate {processed_articles * questions_per_article} questions "
			f"({processed_articles} articles). Minimum required is {min_total_questions} questions."
		)
		print(stop_reason)

	if skipped_articles:
		print(f"Skipped articles due to repeated failures: {skipped_articles}")

	if status == "stopped":
		_write_question_outputs(
			input_path=input_path,
			output_dir=output_dir,
			model=model,
			questions_per_article=questions_per_article,
			chunk_size=chunk_size,
			articles=all_articles,
			status=status,
			stopped_reason=stop_reason,
		)
		return

	print()
	_write_question_outputs(
		input_path=input_path,
		output_dir=output_dir,
		model=model,
		questions_per_article=questions_per_article,
		chunk_size=chunk_size,
		articles=all_articles,
		status="completed",
		stopped_reason="",
	)


def process_directory(
	input_dir: Path,
	output_dir: Path,
	model: str,
	chunk_size: int,
	questions_per_article: int,
	pause_seconds: float,
	max_retries: int,
	min_total_questions: int,
	max_total_questions: int,
) -> None:
	json_files = sorted(path for path in input_dir.glob("*_with_ids.json") if path.is_file())
	if not json_files:
		raise FileNotFoundError(f"No *_with_ids.json files found in directory: {input_dir}")

	print(f"Found {len(json_files)} JSON files in {input_dir}")
	completed = 0
	failed: List[Path] = []
	for json_file in json_files:
		try:
			process_file(
				json_file,
				output_dir,
				model=model,
				chunk_size=chunk_size,
				questions_per_article=questions_per_article,
				pause_seconds=pause_seconds,
				max_retries=max_retries,
				min_total_questions=min_total_questions,
				max_total_questions=max_total_questions,
			)
			completed += 1
		except Exception as exc:
			failed.append(json_file)
			print(f"Failed {json_file}: {exc}")

	if failed:
		raise RuntimeError(f"Completed {completed}/{len(json_files)} files. Failed files: {[path.name for path in failed]}")


def main() -> None:
	_load_env_file(Path(__file__).resolve().parent / ".env")
	default_model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

	parser = argparse.ArgumentParser(description="Generate 10 questions per article using Ollama.")
	parser.add_argument(
		"input",
		help="Path to one JSON file of extracted articles.",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory to write question JSON/CSV files. Defaults to output/questions.",
	)
	parser.add_argument(
		"--model",
		default=default_model,
		help="Ollama model to use (default: OLLAMA_MODEL from .env, else qwen2.5:latest).",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=5,
		help="Number of articles processed per internal batch (default: 5). Batch files are saved separately and merged at the end.",
	)
	parser.add_argument(
		"--questions-per-article",
		type=int,
		default=10,
		help="Number of questions to request per article (default: 10).",
	)
	parser.add_argument(
		"--pause-seconds",
		type=float,
		default=0.0,
		help="Optional pause between batches to reduce rate-limit pressure.",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=3,
		help="Maximum retry attempts for an incomplete model batch (default: 3).",
	)
	parser.add_argument(
		"--min-total-questions",
		type=int,
		default=100,
		help="Minimum total questions to target per document (default: 100).",
	)
	parser.add_argument(
		"--max-total-questions",
		type=int,
		default=200,
		help="Maximum total questions allowed per document (default: 200).",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	if input_path.is_dir():
		raise ValueError(
			f"Input must be a single JSON file, not a directory: {input_path}. "
			"Run this script once per file."
		)

	if not input_path.exists():
		raise FileNotFoundError(f"Input path not found: {input_path}")

	if args.output_dir:
		output_dir = Path(args.output_dir)
	else:
		if input_path.parent.name == "with_ids":
			output_dir = input_path.parent.parent / "questions"
		else:
			output_dir = input_path.parent / "questions"

	process_file(
		input_path,
		output_dir,
		model=args.model,
		chunk_size=args.chunk_size,
		questions_per_article=args.questions_per_article,
		pause_seconds=args.pause_seconds,
		max_retries=args.max_retries,
		min_total_questions=args.min_total_questions,
		max_total_questions=args.max_total_questions,
	)


if __name__ == "__main__":
	main()