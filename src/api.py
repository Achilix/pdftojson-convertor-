from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

try:
	from .recherche import DEFAULT_MODEL, _iter_embedded_files, _load_env_file, _resolve_api_key, search_articles
except ImportError:
	from recherche import DEFAULT_MODEL, _iter_embedded_files, _load_env_file, _resolve_api_key, search_articles


_load_env_file(Path.cwd() / ".env")

EMBEDDINGS_SOURCE = Path(os.environ.get("EMBEDDINGS_DIR", "output/embeddings")).resolve()
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


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
			--bg: #f4f1ea;
			--surface: #ffffff;
			--surface-2: #f7f4ef;
			--text: #111111;
			--muted: #666666;
			--line: #d8d2c6;
			--accent: #111111;
			--accent-soft: #e8e1d5;
			--radius: 18px;
		}
		* { box-sizing: border-box; }
		body {
			margin: 0;
			font-family: Georgia, "Times New Roman", serif;
			background:
				radial-gradient(circle at top left, rgba(17, 17, 17, 0.06), transparent 30%),
				linear-gradient(180deg, #fbfaf7 0%, var(--bg) 100%);
			color: var(--text);
		}
		.container {
			width: min(1100px, calc(100% - 28px));
			margin: 26px auto 40px;
			display: grid;
			gap: 16px;
		}
		.panel {
			background: var(--surface);
			border: 1px solid var(--line);
			border-radius: var(--radius);
			padding: 16px;
		}
		h1 {
			margin: 0 0 8px;
			font-size: clamp(34px, 6vw, 58px);
			line-height: 0.95;
		}
		p { margin: 0; color: var(--muted); line-height: 1.6; }
		.controls {
			display: grid;
			grid-template-columns: 1fr 130px 130px;
			gap: 10px;
			margin-top: 12px;
		}
		input, textarea, button {
			font: inherit;
			border-radius: 14px;
			border: 1px solid var(--line);
			padding: 11px 12px;
		}
		textarea {
			width: 100%;
			min-height: 100px;
			resize: vertical;
			margin-top: 10px;
		}
		.toolbar {
			display: flex;
			gap: 10px;
			margin-top: 10px;
			align-items: center;
			flex-wrap: wrap;
		}
		button {
			background: var(--accent);
			color: #fff;
			cursor: pointer;
			border: 0;
			padding: 11px 14px;
		}
		button.secondary {
			background: var(--accent-soft);
			color: var(--text);
		}
		.status {
			background: var(--surface-2);
			border: 1px solid var(--line);
			padding: 8px 10px;
			border-radius: 12px;
			font-size: 13px;
			color: var(--muted);
		}
		.status.error { color: #842029; background: #fdecef; }
		.status.ok { color: #0f5132; background: #ecf7ef; }
		.meta {
			display: grid;
			grid-template-columns: repeat(4, minmax(0, 1fr));
			gap: 10px;
		}
		.meta div {
			padding: 10px;
			border: 1px solid var(--line);
			border-radius: 12px;
			background: var(--surface-2);
			font-size: 13px;
			color: var(--muted);
		}
		.results { display: grid; gap: 10px; }
		.card {
			border: 1px solid var(--line);
			border-radius: 14px;
			padding: 12px;
			background: var(--surface);
		}
		.card h3 { margin: 0 0 8px; font-size: 17px; }
		.card .small { font-size: 12px; color: var(--muted); margin-top: 8px; }
		@media (max-width: 900px) {
			.controls, .meta { grid-template-columns: 1fr; }
		}
	</style>
</head>
<body>
	<main class="container">
		<section class="panel">
			<h1>Legal Search</h1>
			<p>Search across all embedded files. The server key is loaded from .env automatically.</p>

			<div class="controls">
				<input id="query" type="text" placeholder="Type your legal query" />
				<input id="topK" type="number" min="1" value="5" />
				<input id="threshold" type="number" min="0" max="1" step="0.01" value="0.2" />
			</div>
			<textarea id="queryLong" placeholder="Optional longer query text"></textarea>

			<div class="toolbar">
				<button id="searchBtn" type="button">Search</button>
				<button id="demoBtn" class="secondary" type="button">Load Example</button>
				<span id="status" class="status">Ready.</span>
			</div>
		</section>

		<section class="panel">
			<div class="meta">
				<div>Endpoint: <strong>/search</strong></div>
				<div>Method: <strong>POST</strong></div>
				<div>Source: <strong id="sourceLabel">...</strong></div>
				<div>Files: <strong id="fileCount">...</strong></div>
			</div>
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
			const queryLongInput = document.getElementById('queryLong');
			const topKInput = document.getElementById('topK');
			const thresholdInput = document.getElementById('threshold');
			const searchBtn = document.getElementById('searchBtn');
			const demoBtn = document.getElementById('demoBtn');
			const statusEl = document.getElementById('status');
			const resultsEl = document.getElementById('results');
			const fileCountEl = document.getElementById('fileCount');
			const sourceLabelEl = document.getElementById('sourceLabel');

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
					return '<article class="card">'
						+ '<h3>#' + (idx + 1) + ' - ' + escapeHtml(title) + '</h3>'
						+ '<p><strong>Similarity:</strong> ' + score.toFixed(1) + '% | <strong>Document:</strong> ' + escapeHtml(doc) + '</p>'
						+ '<p>' + escapeHtml(content) + (item.content && String(item.content).length > 420 ? '...' : '') + '</p>'
						+ '<p class="small">Source file: ' + escapeHtml(source) + '</p>'
						+ '</article>';
				}).join('');

				resultsEl.innerHTML = html;
			}

			async function loadHealth() {
				try {
					const response = await fetch('/health');
					const data = await response.json();
					fileCountEl.textContent = String(data.embedded_file_count || '0');
					sourceLabelEl.textContent = String(data.embeddings_source || 'unknown');
				} catch (err) {
					fileCountEl.textContent = 'unavailable';
					sourceLabelEl.textContent = 'unavailable';
				}
			}

			async function runSearch() {
				const longQuery = queryLongInput.value.trim();
				const shortQuery = queryInput.value.trim();
				const query = longQuery || shortQuery;
				const topK = Number(topKInput.value || 5);
				const threshold = Number(thresholdInput.value || 0);

				if (!query) {
					setStatus('Enter a query first.', 'error');
					return;
				}

				searchBtn.disabled = true;
				demoBtn.disabled = true;
				setStatus('Searching...');

				try {
					const response = await fetch('/search', {
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ query: query, top_k: topK, threshold: threshold })
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
					demoBtn.disabled = false;
				}
			}

			searchBtn.addEventListener('click', runSearch);
			demoBtn.addEventListener('click', function() {
				queryInput.value = "obligation d'ouvrir un compte";
				setStatus('Example loaded. Click Search.', 'ok');
			});
			queryInput.addEventListener('keydown', function(event) {
				if (event.key === 'Enter') {
					runSearch();
				}
			});

			loadHealth();
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
		"model": _value("model", DEFAULT_MODEL),
		"api_key": _value("api_key", ""),
	}


def _serialize_result(article: Dict[str, Any], similarity: float) -> Dict[str, Any]:
	payload = {
		key: value
		for key, value in article.items()
		if key not in {"embedding", "_embedded_file"}
	}
	payload["similarity"] = similarity
	embedded_file = article.get("_embedded_file")
	if embedded_file:
		payload["embedded_file"] = embedded_file
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
			self._handle_frontend()
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

	def _handle_frontend(self) -> None:
		_html_response(self, _frontend_html())

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

			api_key = _resolve_api_key(str(params["api_key"]).strip() or None)
			if not api_key:
				raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY, or send api_key in the request.")

			results = search_articles(
				EMBEDDINGS_SOURCE,
				query,
				api_key,
				str(params["model"]).strip() or DEFAULT_MODEL,
				top_k,
				threshold,
			)
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
				"embeddings_source": str(EMBEDDINGS_SOURCE),
				"result_count": len(results),
				"results": [
					_serialize_result(article, similarity)
					for article, similarity in results
				],
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