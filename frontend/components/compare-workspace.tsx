"use client";

import { useEffect, useMemo, useState } from "react";
import { apiPost, type CompareResponse, type QueryMetricsResponse, type RetrievedDocument } from "@/lib/api";
import { TopicChip } from "@/components/topic-chip";

type CompareWorkspaceProps = {
  initialQuery: string;
};

function abstractSnippet(document: RetrievedDocument) {
  const source = document.metadata.abstract || document.retrieval_text || "";
  if (source.length <= 180) {
    return source;
  }
  return `${source.slice(0, 177)}...`;
}

function buildOverlapMap(
  results: {
    method: string;
    documents: RetrievedDocument[];
  }[]
) {
  const counts = new Map<string, number>();

  for (const result of results) {
    for (const doc of result.documents.slice(0, 5)) {
      counts.set(doc.pmid, (counts.get(doc.pmid) ?? 0) + 1);
    }
  }

  return counts;
}

function CompareColumn({
  method,
  items,
  overlapCounts
}: {
  method: string;
  items: RetrievedDocument[];
  overlapCounts: Map<string, number>;
}) {
  return (
    <section className="rounded-[1.6rem] border border-slate-200/80 bg-white/78 p-5 backdrop-blur">
      <div className="border-b border-slate-200/80 pb-4">
        <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Method</p>
        <h2 className="mt-2 font-display text-2xl text-ink">{method.toUpperCase()}</h2>
      </div>

      <div className="mt-5 space-y-4">
        {items.slice(0, 5).map((item, index) => {
          const overlapCount = overlapCounts.get(item.pmid) ?? 1;

          return (
            <article
              key={`${method}-${item.pmid}`}
              className="rounded-[1.2rem] border border-slate-200/80 bg-slate-50/65 p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Rank {index + 1}</p>
                <p className="text-sm font-medium text-sea">{item.score.toFixed(2)}</p>
              </div>

              <h3 className="mt-3 text-sm font-semibold leading-6 text-ink">
                {item.metadata.title || item.retrieval_text}
              </h3>
              <p className="mt-3 line-clamp-4 text-sm leading-6 text-slate-600">
                {abstractSnippet(item)}
              </p>

              <div className="mt-4 flex flex-wrap gap-2">
                <TopicChip label={`PMID ${item.pmid}`} tone="soft" />
                {overlapCount > 1 ? (
                  <TopicChip label={`Overlap in ${overlapCount} methods`} tone="accent" />
                ) : null}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function CompareWorkspace({ initialQuery }: CompareWorkspaceProps) {
  const [query, setQuery] = useState(initialQuery ?? "");
  const [results, setResults] = useState<CompareResponse["results"]>([]);
  const [queryMetrics, setQueryMetrics] = useState<QueryMetricsResponse | null>(null);
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const overlapCounts = useMemo(() => buildOverlapMap(results), [results]);
  const sharedPmids = useMemo(
    () =>
      Array.from(overlapCounts.entries())
        .filter(([, count]) => count > 1)
        .length,
    [overlapCounts]
  );

  const summarySentence =
    sharedPmids > 0
      ? `${sharedPmids} PubMed documents appear in more than one top-5 list, while the rest show where each retrieval method explores a different evidence neighborhood.`
      : "The top-5 lists are fully distinct, which makes the retrieval differences unusually easy to inspect.";

  async function runCompare(nextQuery: string) {
    const normalizedQuery = typeof nextQuery === "string" ? nextQuery.trim() : "";
    if (normalizedQuery.length < 2) {
      setError("Enter at least 2 characters before comparing.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [response, metrics] = await Promise.all([
        apiPost<CompareResponse>("/compare", {
          query: normalizedQuery,
          dataset_name: "unified",
          top_k: 5,
          methods: ["tfidf", "bm25", "dense", "hybrid"]
        }),
        apiPost<QueryMetricsResponse>("/metrics/query", {
          query: normalizedQuery
        })
      ]);
      setResults(response.results);
      setNote(response.note);
      setQueryMetrics(metrics);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Compare request failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (typeof initialQuery === "string" && initialQuery.trim().length >= 2) {
      void runCompare(initialQuery);
    }
  }, [initialQuery]);

  return (
    <div className="space-y-8">
      <section className="rounded-[2rem] border border-white/70 bg-[linear-gradient(145deg,rgba(255,255,255,0.94),rgba(237,244,245,0.72))] p-6 shadow-soft backdrop-blur md:p-8">
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-sea">Shared query</p>
            <div className="mt-4 rounded-[1.5rem] border border-slate-200/80 bg-white/92 p-4">
              <div className="flex flex-col gap-3 rounded-[1.25rem] border border-slate-200 bg-slate-50/75 p-4 md:flex-row md:items-center">
                <div className="h-3 w-3 rounded-full bg-coral" />
                <input
                  aria-label="Compare query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  className="min-w-0 flex-1 bg-transparent text-base text-slate-700 outline-none md:text-lg"
                />
                <button
                  onClick={() => void runCompare(query)}
                  className="rounded-full bg-ink px-5 py-3 text-xs font-semibold uppercase tracking-[0.24em] text-white"
                >
                  Compare
                </button>
              </div>
            </div>
          </div>

          <div className="rounded-[1.5rem] bg-ink p-5 text-white">
            <p className="text-xs uppercase tracking-[0.28em] text-sand/80">Top-line read</p>
            <p className="mt-3 text-base leading-7 text-slate-100">{summarySentence}</p>
            {note ? <p className="mt-3 text-sm leading-6 text-slate-300">{note}</p> : null}
          </div>
        </div>
      </section>

      {loading ? (
        <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm text-slate-600 backdrop-blur">
          Loading live comparison results...
        </section>
      ) : null}

      {error ? (
        <section className="rounded-[1.75rem] border border-coral/30 bg-coral/10 p-6 text-sm text-coral">
          {error}
        </section>
      ) : null}

      {!loading && !error ? (
        results.length > 0 ? (
          <>
            {queryMetrics?.found && queryMetrics.payload.rows ? (
              <section className="overflow-x-auto rounded-[1.75rem] border border-slate-200/80 bg-white/72 backdrop-blur">
                <div className="flex items-center justify-between border-b border-slate-200/80 px-6 py-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.24em] text-sea">Query-specific evaluation</p>
                    <h2 className="mt-1 font-display text-2xl text-ink">
                      Unified benchmark metrics for this exact BioASQ query
                    </h2>
                  </div>
                  <div className="text-right text-sm text-slate-600">
                    <p>Type: {queryMetrics.payload.query_type}</p>
                    <p>Relevant docs: {queryMetrics.payload.num_relevant_docs}</p>
                  </div>
                </div>
                <div className="min-w-[980px]">
                  <div className="grid grid-cols-[1.05fr,0.72fr,0.72fr,0.72fr,0.72fr,0.72fr,0.72fr] px-6 py-4 text-xs uppercase tracking-[0.24em] text-slate-500">
                    <p>Method</p>
                    <p>P@5</p>
                    <p>P@10</p>
                    <p>Recall@10</p>
                    <p>MRR</p>
                    <p>nDCG@5</p>
                    <p>nDCG@10</p>
                  </div>
                  {queryMetrics.payload.rows.map((row) => (
                    <div
                      key={row.method}
                      className="grid grid-cols-[1.05fr,0.72fr,0.72fr,0.72fr,0.72fr,0.72fr,0.72fr] border-t border-slate-200/70 px-6 py-4 text-sm text-slate-700"
                    >
                      <p className="font-display text-xl text-ink">{row.method.toUpperCase()}</p>
                      <p>{row.precision_at_5.toFixed(3)}</p>
                      <p>{row.precision_at_10.toFixed(3)}</p>
                      <p>{row.recall_at_10.toFixed(3)}</p>
                      <p>{row.mrr.toFixed(3)}</p>
                      <p>{row.ndcg_at_5.toFixed(3)}</p>
                      <p>{row.ndcg_at_10.toFixed(3)}</p>
                    </div>
                  ))}
                </div>
              </section>
            ) : queryMetrics && !queryMetrics.found ? (
              <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm leading-7 text-slate-600 backdrop-blur">
                Query-specific evaluation is only available when the compare query exactly matches a
                BioASQ query in the unified dataset.
              </section>
            ) : null}

            <section className="grid gap-5 xl:grid-cols-4">
              {results.map((result) => (
                <CompareColumn
                  key={result.method}
                  method={result.method}
                  items={result.documents}
                  overlapCounts={overlapCounts}
                />
              ))}
            </section>
          </>
        ) : (
          <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm leading-7 text-slate-600 backdrop-blur">
            No comparison results were returned for this query. Try a broader biomedical question.
          </section>
        )
      ) : null}
    </div>
  );
}
