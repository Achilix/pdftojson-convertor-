from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

try:
	import google.genai as genai
except ImportError:
	genai = None

try:
	from .recherche import DEFAULT_MODEL, _iter_embedded_files, _load_env_file, _resolve_api_key, search_articles
except ImportError:
	from recherche import DEFAULT_MODEL, _iter_embedded_files, _load_env_file, _resolve_api_key, search_articles


_load_env_file(Path.cwd() / ".env")

EMBEDDINGS_SOURCE = Path(os.environ.get("EMBEDDINGS_DIR", "output/embeddings")).resolve()
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_VERIFY_MODEL = "gemini-2.0-flash"

CLOSE_FILTER_THRESHOLDS = {
	"off": 0.0,
	"loose": 0.2,
	"balanced": 0.35,
	"strict": 0.5,
}


def _resolve_close_filter_threshold(close_filter: str) -> float:
	key = (close_filter or "").strip().lower()
	if key not in CLOSE_FILTER_THRESHOLDS:
		raise ValueError("close_filter must be one of: off, loose, balanced, strict")
	return CLOSE_FILTER_THRESHOLDS[key]


def _closeness_label(similarity: float) -> str:
	if similarity >= 0.6:
		return "very-close"
	if similarity >= 0.45:
		return "close"
	if similarity >= 0.3:
		return "moderate"
	return "weak"


def _to_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return value != 0
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return False


def _extract_json_object(text: str) -> str:
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end < start:
		raise ValueError("Verification model did not return JSON")
	return text[start : end + 1]


def _verify_results_with_gemini(
	query: str,
	results: List[Dict[str, Any]],
	api_key: str,
	verify_top_n: int,
	verify_model: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
	if genai is None:
		raise RuntimeError("google-genai is required for AI verification. Install with: pip install google-genai")

	checked_count = min(max(verify_top_n, 1), len(results))
	target = []
	for index, item in enumerate(results[:checked_count]):
		target.append(
			{
				"index": index,
				"article_number": item.get("article_number"),
				"document_name": item.get("document_name"),
				"content": str(item.get("content", ""))[:900],
			}
		)

	prompt = (
		"You are validating legal semantic-search matches. "
		"Given a query and candidate passages, decide if each passage is relevant to answering the query. "
		"Return ONLY valid JSON with this exact shape: "
		"{\"overall\":\"high|medium|low\",\"explanation\":\"short text\","
		"\"items\":[{\"index\":0,\"is_relevant\":true,\"score\":0.0,\"reason\":\"short\"}]}. "
		"Score must be between 0 and 1."
		f"\n\nQuery:\n{query}\n\nCandidates:\n{json.dumps(target, ensure_ascii=False)}"
	)

	client = genai.Client(api_key=api_key)
	response = client.models.generate_content(
		model=f"models/{verify_model}" if not verify_model.startswith("models/") else verify_model,
		contents=prompt,
	)

	response_text = getattr(response, "text", "") or ""
	parsed = json.loads(_extract_json_object(response_text))
	items = parsed.get("items", []) if isinstance(parsed, dict) else []
	by_index = {
		int(item.get("index")): item
		for item in items
		if isinstance(item, dict) and isinstance(item.get("index"), int)
	}

	updated_results: List[Dict[str, Any]] = []
	relevant_count = 0
	for index, item in enumerate(results):
		entry = dict(item)
		if index < checked_count:
			judge = by_index.get(index, {})
			is_relevant = bool(judge.get("is_relevant", False))
			score = judge.get("score", 0.0)
			try:
				score = max(0.0, min(1.0, float(score)))
			except (TypeError, ValueError):
				score = 0.0
			entry["ai_is_relevant"] = is_relevant
			entry["ai_relevance_score"] = score
			entry["ai_reason"] = str(judge.get("reason", ""))[:200]
			if is_relevant:
				relevant_count += 1
		updated_results.append(entry)

	overall = "medium"
	if checked_count > 0:
		ratio = relevant_count / checked_count
		if ratio >= 0.75:
			overall = "high"
		elif ratio <= 0.34:
			overall = "low"

	if isinstance(parsed, dict) and isinstance(parsed.get("overall"), str):
		model_overall = parsed["overall"].strip().lower()
		if model_overall in {"high", "medium", "low"}:
			overall = model_overall

	summary = {
		"enabled": True,
		"model": verify_model,
		"checked_count": checked_count,
		"relevant_count": relevant_count,
		"overall": overall,
		"explanation": str(parsed.get("explanation", "")).strip()[:300] if isinstance(parsed, dict) else "",
	}
	return summary, updated_results


def _html_response(handler: BaseHTTPRequestHandler, html: str, status_code: int = 200) -> None:
	body = html.encode("utf-8")
	handler.send_response(status_code)
	handler.send_header("Content-Type", "text/html; charset=utf-8")
	handler.send_header("Content-Length", str(len(body)))
	handler.send_header("Access-Control-Allow-Origin", "*")
	handler.end_headers()
	handler.wfile.write(body)


def _frontend_html() -> str:
	return """<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<title>Legal Search</title>
	<style>
		:root {
			--bg: #f2ecdf;
			--surface: #ffffff;
			--line: #d7cfc2;
			--text: #111111;
			--muted: #585858;
			--accent: #111111;
			--accent-soft: #ece4d7;
			--radius: 18px;
		}
		* { box-sizing: border-box; }
		body {
			margin: 0;
			font-family: Georgia, "Times New Roman", serif;
			color: var(--text);
			background:
				radial-gradient(circle at top left, rgba(17, 17, 17, 0.1), transparent 35%),
				radial-gradient(circle at bottom right, rgba(120, 95, 60, 0.08), transparent 32%),
				linear-gradient(180deg, #faf8f3 0%, var(--bg) 100%);
		}
		.container {
			width: min(920px, calc(100% - 28px));
			margin: 34px auto 44px;
			display: grid;
			gap: 16px;
		}
		.panel {
			background: var(--surface);
			border: 1px solid var(--line);
			border-radius: var(--radius);
			padding: 20px;
			box-shadow: 0 14px 30px rgba(30, 26, 20, 0.08);
		}
		h1 {
			margin: 0 0 10px;
			font-size: clamp(34px, 8vw, 64px);
			line-height: 0.95;
		}
		.description {
			margin: 0;
			color: var(--muted);
			font-size: 16px;
			line-height: 1.6;
		}
		.search-row {
			display: grid;
			grid-template-columns: 1fr auto;
			gap: 10px;
			margin-top: 14px;
		}
		.examples {
			display: flex;
			gap: 8px;
			flex-wrap: wrap;
			margin-top: 10px;
		}
		input, button {
			font: inherit;
			border-radius: 12px;
			padding: 12px 14px;
		}
		input {
			border: 1px solid var(--line);
		}
		button {
			border: 0;
			background: var(--accent);
			color: #ffffff;
			cursor: pointer;
			min-width: 110px;
			transition: transform 140ms ease, opacity 140ms ease;
		}
		button:hover {
			opacity: 0.94;
			transform: translateY(-1px);
		}
		.example-btn {
			background: var(--accent-soft);
			color: #1d1d1d;
			min-width: auto;
			padding: 8px 11px;
			font-size: 13px;
		}
		.status {
			margin-top: 10px;
			border-radius: 12px;
			padding: 8px 10px;
			font-size: 13px;
			border: 1px solid var(--line);
			background: #f6f2eb;
			color: var(--muted);
		}
		.status.error { color: #842029; background: #fdecef; border-color: #f4c7cf; }
		.status.ok { color: #0f5132; background: #ecf7ef; border-color: #bddac7; }
		.results { display: grid; gap: 10px; }
		.card {
			border: 1px solid var(--line);
			border-radius: 12px;
			padding: 14px;
			background: #ffffff;
			position: relative;
		}
		.card.closest {
			border-color: #bda57f;
			background: linear-gradient(180deg, #fffaf2 0%, #ffffff 100%);
		}
		.closest-badge {
			display: inline-block;
			font-size: 11px;
			font-weight: 700;
			letter-spacing: 0.05em;
			text-transform: uppercase;
			padding: 4px 8px;
			border-radius: 999px;
			background: #1c1a17;
			color: #f9f4ea;
			margin-bottom: 8px;
		}
		.card h3 { margin: 0 0 6px; font-size: 17px; }
		.card .small { margin-top: 8px; font-size: 12px; color: var(--muted); }
		@media (max-width: 720px) {
			.search-row { grid-template-columns: 1fr; }
			button { width: 100%; }
		}
	</style>
</head>
<body>
	<main class="container">
		<section class="panel">
			<h1>Legal Search</h1>
			<p class="description">Ask a legal question and get the closest passages from your embedded documents.</p>
			<div class="search-row">
				<input id="query" type="text" placeholder="Type your legal query" />
				<button id="searchBtn" type="button">Search</button>
			</div>
			<div class="examples">
				<button class="example-btn" type="button" data-query="obligation d'ouvrir un compte">Example: Compte</button>
				<button class="example-btn" type="button" data-query="resiliation de contrat commercial">Example: Contrat</button>
				<button class="example-btn" type="button" data-query="delai de paiement et penalites">Example: Paiement</button>
			</div>
			<div id="status" class="status">Ready.</div>
		</section>

		<section class="panel">
			<div id="results" class="results">
				<div class="status">Run a search to see results.</div>
			</div>
		</section>
	</main>

	<script>
		(function() {
			const queryInput = document.getElementById('query');
			const searchBtn = document.getElementById('searchBtn');
			const exampleButtons = Array.from(document.querySelectorAll('.example-btn'));
			const statusEl = document.getElementById('status');
			const resultsEl = document.getElementById('results');

			function setStatus(message, kind) {
				statusEl.textContent = message;
				statusEl.className = kind ? 'status ' + kind : 'status';
			}

			function escapeHtml(value) {
				return String(value)
					.replace(/&/g, '&amp;')
					.replace(/</g, '&lt;')
					.replace(/>/g, '&gt;')
					.replace(/"/g, '&quot;')
					.replace(/'/g, '&#39;');
			}

			function renderResults(items) {
				if (!Array.isArray(items) || items.length === 0) {
					resultsEl.innerHTML = '<div class="status">No matches found.</div>';
					return;
				}

				const html = items.map(function(item, idx) {
					const score = Number(item.similarity || 0) * 100;
					const content = item.content ? String(item.content).slice(0, 420) : '';
					const title = 'Article ' + (item.article_number || 'Unknown');
					const doc = item.document_name || 'Unknown document';
					const source = item.embedded_file || 'n/a';
					const closest = idx === 0;
					return '<article class="card' + (closest ? ' closest' : '') + '">'
						+ (closest ? '<div class="closest-badge">Closest Match</div>' : '')
						+ '<h3>#' + (idx + 1) + ' - ' + escapeHtml(title) + '</h3>'
						+ '<p><strong>Similarity:</strong> ' + score.toFixed(1) + '% | <strong>Document:</strong> ' + escapeHtml(doc) + '</p>'
						+ '<p>' + escapeHtml(content) + (item.content && String(item.content).length > 420 ? '...' : '') + '</p>'
						+ '<p class="small">Source file: ' + escapeHtml(source) + '</p>'
						+ '</article>';
				}).join('');

				resultsEl.innerHTML = html;
			}

			async function runSearch() {
				const query = queryInput.value.trim();
				if (!query) {
					setStatus('Enter a query first.', 'error');
					return;
				}

				searchBtn.disabled = true;
				setStatus('Searching...');

				try {
					const response = await fetch('/search', {
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ query: query })
					});

					const data = await response.json();
					if (!response.ok) {
						throw new Error(data && data.error ? data.error : 'Search failed');
					}

					setStatus('Found ' + (data.result_count || 0) + ' result(s).', 'ok');
					renderResults(data.results || []);
				} catch (err) {
					const message = (err && err.message) ? err.message : 'Search failed';
					setStatus(message, 'error');
					resultsEl.innerHTML = '<div class="status error">' + escapeHtml(message) + '</div>';
				} finally {
					searchBtn.disabled = false;
				}
			}

			searchBtn.addEventListener('click', runSearch);
			exampleButtons.forEach(function(btn) {
				btn.addEventListener('click', function() {
					queryInput.value = String(btn.dataset.query || '');
					runSearch();
				});
			});
			queryInput.addEventListener('keydown', function(event) {
				if (event.key === 'Enter') {
					runSearch();
				}
			});
		})();
	</script>
</body>
</html>"""


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: Dict[str, Any]) -> None:
	body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
	handler.send_response(status_code)
	handler.send_header("Content-Type", "application/json; charset=utf-8")
	handler.send_header("Content-Length", str(len(body)))
	handler.send_header("Access-Control-Allow-Origin", "*")
	handler.end_headers()
	handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
	content_length = int(handler.headers.get("Content-Length", "0") or 0)
	if content_length <= 0:
		return {}

	raw_body = handler.rfile.read(content_length)
	if not raw_body:
		return {}

	try:
		payload = json.loads(raw_body.decode("utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError("Request body must be valid JSON") from exc

	if not isinstance(payload, dict):
		raise ValueError("Request body must be a JSON object")

	return payload


def _first_value(values: Dict[str, list[str]], key: str, default: Any = None) -> Any:
	if key not in values or not values[key]:
		return default
	return values[key][0]


def _parse_search_request(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
	parsed_url = urlparse(handler.path)
	query_params = parse_qs(parsed_url.query)
	body_payload: Dict[str, Any] = {}

	if handler.command == "POST":
		body_payload = _read_json_body(handler)

	def _value(key: str, default: Any = None) -> Any:
		if key in body_payload:
			return body_payload[key]
		return _first_value(query_params, key, default)

	return {
		"query": _value("query", ""),
		"top_k": _value("top_k", 5),
		"threshold": _value("threshold", 0.0),
		"close_filter": _value("close_filter", "balanced"),
		"verify_results": _value("verify_results", False),
		"verify_top_n": _value("verify_top_n", 3),
		"verify_model": _value("verify_model", DEFAULT_VERIFY_MODEL),
		"model": _value("model", DEFAULT_MODEL),
		"api_key": _value("api_key", ""),
	}


def _to_relative_source_path(path_value: str) -> str:
	try:
		path_obj = Path(path_value)
		if not path_obj.is_absolute():
			return path_obj.as_posix()

		resolved = path_obj.resolve()
		workspace_root = Path.cwd().resolve()
		try:
			return resolved.relative_to(workspace_root).as_posix()
		except ValueError:
			pass

		try:
			return resolved.relative_to(EMBEDDINGS_SOURCE.parent).as_posix()
		except ValueError:
			return path_obj.name or str(path_value).replace("\\", "/")
	except Exception:
		return str(path_value).replace("\\", "/")


def _serialize_result(article: Dict[str, Any], similarity: float) -> Dict[str, Any]:
	payload = {
		key: value
		for key, value in article.items()
		if key not in {"embedding", "_embedded_file"}
	}
	payload["similarity"] = similarity
	payload["closeness_label"] = _closeness_label(similarity)
	embedded_file = article.get("_embedded_file")
	if embedded_file:
		payload["embedded_file"] = _to_relative_source_path(str(embedded_file))
	return payload


class SearchAPIHandler(BaseHTTPRequestHandler):
	server_version = "LegalSearchAPI/1.0"

	def do_OPTIONS(self) -> None:
		self.send_response(204)
		self.send_header("Access-Control-Allow-Origin", "*")
		self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		self.send_header("Access-Control-Allow-Headers", "Content-Type")
		self.end_headers()

	def do_GET(self) -> None:
		self._route()

	def do_POST(self) -> None:
		self._route()

	def _route(self) -> None:
		parsed_url = urlparse(self.path)
		if parsed_url.path in {"/", "/index.html"}:
			self._handle_root()
			return

		if parsed_url.path == "/health":
			self._handle_health()
			return

		if parsed_url.path == "/search":
			self._handle_search()
			return

		_json_response(self, 404, {"error": "Not found"})

	def _handle_health(self) -> None:
		embedded_files = _iter_embedded_files(EMBEDDINGS_SOURCE) if EMBEDDINGS_SOURCE.exists() else []
		_json_response(
			self,
			200,
			{
				"status": "ok",
				"embeddings_source": str(EMBEDDINGS_SOURCE),
				"embedded_file_count": len(embedded_files),
			},
		)

	def _handle_root(self) -> None:
		_json_response(
			self,
			200,
			{
				"status": "ok",
				"message": "Built-in UI removed. Use your React/Next frontend with this API.",
				"endpoints": {
					"health": "/health",
					"search": "/search",
				},
			},
		)

	def _handle_search(self) -> None:
		if not EMBEDDINGS_SOURCE.exists():
			_json_response(
				self,
				400,
				{
					"error": f"Embeddings source not found: {EMBEDDINGS_SOURCE}",
				},
			)
			return

		try:
			params = _parse_search_request(self)
			query = str(params["query"]).strip()
			if not query:
				raise ValueError("Query cannot be empty")

			top_k = int(params["top_k"])
			if top_k < 1:
				raise ValueError("top_k must be at least 1")

			threshold = float(params["threshold"])
			if threshold < 0.0 or threshold > 1.0:
				raise ValueError("threshold must be between 0.0 and 1.0")

			close_filter = str(params["close_filter"])
			close_filter_threshold = _resolve_close_filter_threshold(close_filter)
			effective_threshold = max(threshold, close_filter_threshold)
			verify_results = _to_bool(params["verify_results"])
			verify_top_n = int(params["verify_top_n"])
			if verify_top_n < 1:
				raise ValueError("verify_top_n must be at least 1")
			verify_model = str(params["verify_model"]).strip() or DEFAULT_VERIFY_MODEL

			api_key = _resolve_api_key(str(params["api_key"]).strip() or None)
			if not api_key:
				raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY, or send api_key in the request.")

			results = search_articles(
				EMBEDDINGS_SOURCE,
				query,
				api_key,
				str(params["model"]).strip() or DEFAULT_MODEL,
				top_k,
				effective_threshold,
			)

			serialized_results = [
				_serialize_result(article, similarity)
				for article, similarity in results
			]

			ai_verification: Dict[str, Any] = {
				"enabled": False,
				"model": verify_model,
				"checked_count": 0,
				"relevant_count": 0,
				"overall": "not-run",
				"explanation": "",
			}

			if verify_results and serialized_results:
				try:
					ai_verification, serialized_results = _verify_results_with_gemini(
						query,
						serialized_results,
						api_key,
						verify_top_n,
						verify_model,
					)
				except Exception as verify_exc:
					ai_verification = {
						"enabled": False,
						"model": verify_model,
						"checked_count": 0,
						"relevant_count": 0,
						"overall": "not-run",
						"explanation": "AI verification unavailable; returning similarity-ranked results.",
						"error": str(verify_exc),
					}
		except ValueError as exc:
			_json_response(self, 400, {"error": str(exc)})
			return
		except Exception as exc:
			_json_response(self, 500, {"error": str(exc)})
			return

		_json_response(
			self,
			200,
			{
				"query": query,
				"model": str(params["model"]).strip() or DEFAULT_MODEL,
				"top_k": top_k,
				"threshold": threshold,
				"close_filter": close_filter,
				"close_filter_threshold": close_filter_threshold,
				"effective_threshold": effective_threshold,
				"top_similarity": results[0][1] if results else 0.0,
				"verify_results": verify_results,
				"verify_top_n": verify_top_n,
				"verify_model": verify_model,
				"ai_verification": ai_verification,
				"embeddings_source": str(EMBEDDINGS_SOURCE),
				"result_count": len(results),
				"results": serialized_results,
			},
		)

	def log_message(self, format: str, *args: Any) -> None:
		return


def run_server() -> None:
	server = ThreadingHTTPServer((HOST, PORT), SearchAPIHandler)
	print(f"Search API listening on http://{HOST}:{PORT}")
	print(f"Searching embedded files under: {EMBEDDINGS_SOURCE}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nStopping server...")
	finally:
		server.server_close()


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the legal semantic search API server")
	parser.add_argument(
		"--host",
		default=HOST,
		help=f"Host/interface to bind (default: {HOST})",
	)
	parser.add_argument(
		"--port",
		type=int,
		default=PORT,
		help=f"Port to listen on (default: {PORT})",
	)
	parser.add_argument(
		"--embeddings-dir",
		default=str(EMBEDDINGS_SOURCE),
		help="Directory containing embedded JSON files (default: output/embeddings or EMBEDDINGS_DIR env var)",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	HOST = args.host
	PORT = args.port
	EMBEDDINGS_SOURCE = Path(args.embeddings_dir).resolve()
	run_server()