"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type SearchResult = {
  article_number?: string | number;
  document_name?: string;
  content?: string;
  similarity?: number;
  embedded_file?: string;
  ai_is_relevant?: boolean;
  ai_relevance_score?: number;
  ai_reason?: string;
};

type AIVerification = {
  enabled?: boolean;
  overall?: "high" | "medium" | "low" | string;
  explanation?: string;
  checked_count?: number;
  relevant_count?: number;
};

type SearchResponse = {
  result_count?: number;
  results?: SearchResult[];
  ai_verification?: AIVerification;
  error?: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

const EXAMPLE_QUERIES = [
  "obligation d'ouvrir un compte",
  "resiliation de contrat commercial",
  "delai de paiement et penalites",
];

export default function Home() {
  const [query, setQuery] = useState("");
  const [verifyEnabled, setVerifyEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready.");
  const [statusKind, setStatusKind] = useState<"idle" | "ok" | "error">("idle");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [toast, setToast] = useState("");
  const [toastKind, setToastKind] = useState<"ok" | "error">("ok");
  const toastTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  const statusClass = useMemo(() => {
    if (statusKind === "ok") return "status status-ok";
    if (statusKind === "error") return "status status-error";
    return "status";
  }, [statusKind]);

  const showToast = (message: string, kind: "ok" | "error" = "ok") => {
    setToast(message);
    setToastKind(kind);

    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToast("");
      toastTimerRef.current = null;
    }, 2800);
  };

  const toRelativeSource = (pathValue?: string) => {
    if (!pathValue) return "n/a";
    const normalized = String(pathValue).replace(/\\/g, "/");
    const outputIndex = normalized.toLowerCase().indexOf("output/");
    if (outputIndex >= 0) {
      return normalized.slice(outputIndex);
    }
    const parts = normalized.split("/");
    return parts[parts.length - 1] || normalized;
  };

  const runSearch = async (nextQuery?: string) => {
    const currentQuery = (nextQuery ?? query).trim();
    if (!currentQuery) {
      setStatusKind("error");
      setStatus("Enter a query first.");
      return;
    }

    setLoading(true);
    setStatusKind("idle");
    setStatus("Searching...");

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: currentQuery,
          verify_results: verifyEnabled,
          verify_top_n: 5,
        }),
      });

      const data = (await response.json()) as SearchResponse;
      if (!response.ok) {
        throw new Error(data.error || "Search failed");
      }

      const nextResults = Array.isArray(data.results) ? data.results : [];
      const verification = data.ai_verification;
      setResults(nextResults);
      setStatusKind("ok");

      let statusText = `Found ${data.result_count || nextResults.length} result(s).`;
      if (verifyEnabled && verification?.enabled) {
        statusText += ` AI check: ${verification.relevant_count || 0}/${verification.checked_count || 0} relevant (${verification.overall || "unknown"}).`;
      } else if (
        verifyEnabled &&
        !verification?.enabled &&
        verification?.explanation
      ) {
        statusText += ` ${verification.explanation}`;
      }
      setStatus(statusText);

      const topResult = nextResults[0];
      if (topResult) {
        if (verifyEnabled && topResult.ai_is_relevant === false) {
          showToast(
            "Top match may be weak. Try a more specific query.",
            "error",
          );
        } else {
          const topMessage =
            verifyEnabled && verification?.enabled
              ? `Closest match verified: ${verification.overall || "medium"} confidence.`
              : "This is the closest match.";
          showToast(topMessage, "ok");
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Search failed";
      setResults([]);
      setStatusKind("error");
      setStatus(message);
      showToast(message, "error");
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await runSearch();
  };

  return (
    <main className="page-shell">
      {toast ? <div className={`toast toast-${toastKind}`}>{toast}</div> : null}
      <section className="hero-card">
        <p className="eyebrow">Legal AI Search</p>
        <h1>Search Your Embedded Legal Corpus</h1>
        <p className="description">
          Next.js frontend connected to your Python API. Ask in French or
          English and get the closest legal passages.
        </p>

        <form className="search-row" onSubmit={onSubmit}>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type your legal query"
            aria-label="Search query"
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        <div className="examples">
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              type="button"
              className="example-btn"
              onClick={() => {
                setQuery(example);
                void runSearch(example);
              }}
              disabled={loading}
            >
              {example}
            </button>
          ))}
        </div>

        <label className="verify-toggle">
          <input
            type="checkbox"
            checked={verifyEnabled}
            onChange={(event) => setVerifyEnabled(event.target.checked)}
            disabled={loading}
          />
          Verify results with Gemini (uses quota)
        </label>

        <div className={statusClass}>{status}</div>
      </section>

      <section className="results-wrap" aria-live="polite">
        {results.length === 0 ? (
          <div className="status">Run a search to see results.</div>
        ) : (
          results.map((item, index) => {
            const similarity = Number(item.similarity || 0) * 100;
            const content = item.content
              ? String(item.content).slice(0, 460)
              : "";
            const isTop = index === 0;

            return (
              <article
                key={`${item.article_number || "article"}-${index}`}
                className={`result-card${isTop ? " result-top" : ""}`}
              >
                {isTop ? (
                  <div className="closest-badge">Closest Match</div>
                ) : null}
                {typeof item.ai_is_relevant === "boolean" ? (
                  <div
                    className={`verify-pill ${item.ai_is_relevant ? "verify-pass" : "verify-fail"}`}
                  >
                    {item.ai_is_relevant
                      ? "AI: Relevant"
                      : "AI: Not fully relevant"}
                    {typeof item.ai_relevance_score === "number"
                      ? ` (${(item.ai_relevance_score * 100).toFixed(0)}%)`
                      : ""}
                  </div>
                ) : null}
                <h2>Article {item.article_number || "Unknown"}</h2>
                <p>
                  <strong>Similarity:</strong> {similarity.toFixed(1)}% |{" "}
                  <strong>Document:</strong>{" "}
                  {item.document_name || "Unknown document"}
                </p>
                <p>
                  {content}
                  {item.content && String(item.content).length > 460
                    ? "..."
                    : ""}
                </p>
                <p className="small">
                  Source file: {toRelativeSource(item.embedded_file)}
                </p>
                {item.ai_reason ? (
                  <p className="small">AI note: {item.ai_reason}</p>
                ) : null}
              </article>
            );
          })
        )}
      </section>
    </main>
  );
}
