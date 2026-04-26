"use client";

import { useEffect, useMemo, useState } from "react";
import { AIPolishPanel } from "@/components/ai-polish-panel";
import {
  apiPost,
  type AIPolishResponse,
  type ClusterResponse,
  type RetrievedDocument,
  type SearchResponse
} from "@/lib/api";
import { TopicChip } from "@/components/topic-chip";

type SearchExperienceProps = {
  initialQuery: string;
  methods: {
    id: string;
    label: string;
  }[];
};

type ViewMode = "ranked" | "topic";

function methodIdToLabel(id: string, methods: SearchExperienceProps["methods"]) {
  return methods.find((method) => method.id === id)?.label ?? id.toUpperCase();
}

function abstractSnippet(document: RetrievedDocument) {
  const source = document.metadata.abstract || document.retrieval_text || "";
  if (source.length <= 500) {
    return source;
  }
  return `${source.slice(0, 497)}...`;
}

function SearchResultCard({
  item,
  methodLabel,
  clusterLabel
}: {
  item: RetrievedDocument;
  methodLabel: string;
  clusterLabel?: string;
}) {
  const [expanded, setExpanded] = useState(false);

  const fullText = item.metadata.abstract || item.retrieval_text || "";
  const displayText = expanded ? fullText : abstractSnippet(item);

  return (
    <article className="rounded-[1.5rem] border border-slate-200/80 bg-white/82 p-5 backdrop-blur">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <TopicChip label={methodLabel} tone="soft" />
            {clusterLabel ? <TopicChip label={clusterLabel} tone="default" /> : null}
            {item.metadata.mesh_terms?.[0] ? (
              <TopicChip label={item.metadata.mesh_terms[0]} tone="default" />
            ) : null}
          </div>
          <h3 className="font-display text-2xl leading-8 text-ink">
            {item.metadata.title || item.retrieval_text}
          </h3>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Score</p>
          <p className="mt-1 font-display text-2xl text-sea">{item.score.toFixed(2)}</p>
        </div>
      </div>

      <p className="mt-4 text-sm leading-7 text-slate-700">{displayText}</p>

      {fullText.length > 500 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-sm text-sea hover:underline"
        >
          {expanded ? "Read less" : "Read more"}
        </button>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-4 border-t border-slate-200/80 pt-4 text-sm text-slate-600">
        <p>
          <span className="font-medium text-ink">PMID</span> {item.pmid}
        </p>
        <p>
          <span className="font-medium text-ink">Journal</span> {item.metadata.journal || "Unknown"}
        </p>
        <p>
          <span className="font-medium text-ink">Year</span> {item.metadata.year ?? "Unknown"}
        </p>
        <a
          href={`https://pubmed.ncbi.nlm.nih.gov/${item.pmid}/`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sea hover:underline"
        >
          View on PubMed
        </a>
      </div>
    </article>
  );
}

export function SearchExperience({ initialQuery, methods }: SearchExperienceProps) {
  const [query, setQuery] = useState(initialQuery ?? "");
  const [selectedMethod, setSelectedMethod] = useState(methods[1]?.id ?? methods[0]?.id ?? "bm25");
  const [viewMode, setViewMode] = useState<ViewMode>("ranked");
  const [results, setResults] = useState<RetrievedDocument[]>([]);
  const [clusters, setClusters] = useState<ClusterResponse["cluster_summaries"]>([]);
  const [aiPolish, setAiPolish] = useState<AIPolishResponse | null>(null);
  const [statusNote, setStatusNote] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const featuredQueries = [
    "heart attack",
    "myocardial infarction",
    "diabetes treatment",
    "drug adverse effects",
    "breast cancer biomarkers"
  ];

  const selectedMethodLabel = useMemo(
    () => methodIdToLabel(selectedMethod, methods),
    [methods, selectedMethod]
  );

  async function runSearch(nextQuery: string, method: string) {
    const normalizedQuery = typeof nextQuery === "string" ? nextQuery.trim() : "";
    if (normalizedQuery.length < 2) {
      setError("Enter at least 2 characters before searching.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [searchResponse, clusterResponse, polishResponse] = await Promise.all([
        apiPost<SearchResponse>("/search", {
          query: normalizedQuery,
          method,
          dataset_name: "pubmed_subset",
          top_k: 10
        }),
        apiPost<ClusterResponse>("/clusters", {
          query: normalizedQuery,
          method,
          dataset_name: "pubmed_subset",
          top_k: 20,
          num_clusters: 5,
          vector_space: "tfidf"
        }),
        apiPost<AIPolishResponse>("/ai-polish", {
          query: normalizedQuery,
          method,
          dataset_name: "pubmed_subset",
          top_k: 3,
          include_matches: true,
          refine_clusters: false
        })
      ]);

      setResults(searchResponse.documents);
      setStatusNote(searchResponse.note);
      setClusters(clusterResponse.cluster_summaries);
      setAiPolish(polishResponse);
      if (clusterResponse.status !== "success" && clusterResponse.error) {
        setError(clusterResponse.error);
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Search request failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (typeof initialQuery === "string" && initialQuery.trim().length >= 2) {
      void runSearch(initialQuery, selectedMethod);
    }
    // selectedMethod intentionally omitted here so first load happens once.
    // Subsequent method changes go through the select handler.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery]);

  return (
    <div className="space-y-8">
      <section className="overflow-hidden rounded-[2rem] border border-white/70 bg-[linear-gradient(145deg,rgba(255,255,255,0.94),rgba(246,231,207,0.72))] p-6 shadow-soft backdrop-blur md:p-8">
        <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-sea">Curated PubMed subset search</p>
            <div className="mt-4 rounded-[1.6rem] border border-slate-200/80 bg-white/92 p-4">
              <div className="flex flex-col gap-3 rounded-[1.25rem] border border-slate-200 bg-slate-50/75 p-4 md:flex-row md:items-center">
                <div className="h-3 w-3 rounded-full bg-coral" />
                <input
                  aria-label="Biomedical search query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  className="min-w-0 flex-1 bg-transparent text-base text-slate-700 outline-none md:text-lg"
                />
                <button
                  onClick={() => void runSearch(query, selectedMethod)}
                  className="rounded-full bg-ink px-5 py-3 text-xs font-semibold uppercase tracking-[0.24em] text-white"
                >
                  Search
                </button>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-[0.75fr_0.75fr_1fr]">
                <label className="space-y-2 text-sm text-slate-600">
                  <span className="text-xs uppercase tracking-[0.22em] text-slate-500">
                    Retrieval method
                  </span>
                  <select
                    value={selectedMethod}
                    onChange={(event) => {
                      const nextMethod = event.target.value;
                      setSelectedMethod(nextMethod);
                      void runSearch(query, nextMethod);
                    }}
                    className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-ink outline-none"
                  >
                    {methods.map((method) => (
                      <option key={method.id} value={method.id}>
                        {method.label}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="space-y-2 text-sm text-slate-600">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-500">View mode</p>
                  <div className="flex rounded-2xl border border-slate-200 bg-white p-1">
                    {([
                      ["ranked", "Ranked list"],
                      ["topic", "Grouped by topic"]
                    ] as const).map(([mode, label]) => (
                      <button
                        key={mode}
                        onClick={() => setViewMode(mode)}
                        className={`flex-1 rounded-[0.9rem] px-4 py-3 text-sm transition ${
                          viewMode === mode ? "bg-ink text-white" : "text-slate-600"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Suggested queries</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {featuredQueries.map((item, index) => (
                      <button
                        key={item}
                        onClick={() => {
                          setQuery(item);
                          void runSearch(item, selectedMethod);
                        }}
                        className="text-left"
                      >
                        <TopicChip label={item} tone={index === 0 ? "accent" : "default"} />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-[1.5rem] bg-ink p-5 text-white">
              <p className="text-xs uppercase tracking-[0.28em] text-sand/80">Current mode</p>
              <p className="mt-3 font-display text-3xl">{selectedMethodLabel}</p>
              <p className="mt-3 text-sm leading-7 text-slate-200">
                {statusNote ||
                  "Search operates over the curated PubMed subset for practical exploration."}
              </p>
            </div>
            <div className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
              <div className="rounded-[1.4rem] border border-slate-200/80 bg-white/76 p-4">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Visible results</p>
                <p className="mt-2 font-display text-3xl text-ink">{results.length}</p>
              </div>
              <div className="rounded-[1.4rem] border border-slate-200/80 bg-white/76 p-4">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Topic clusters</p>
                <p className="mt-2 font-display text-3xl text-ink">{clusters.length}</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {loading ? (
        <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm text-slate-600 backdrop-blur">
          Loading live retrieval results...
        </section>
      ) : null}

      {error ? (
        <section className="rounded-[1.75rem] border border-coral/30 bg-coral/10 p-6 text-sm text-coral">
          {error}
        </section>
      ) : null}

      {!loading && !error && viewMode === "ranked" ? (
        <section className="space-y-4">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-sea">Results</p>
              <h2 className="mt-2 font-display text-3xl text-ink">Ranked biomedical evidence</h2>
            </div>
            <p className="max-w-xl text-sm leading-6 text-slate-600">
              Each card uses live PubMed metadata from the backend retrieval service.
            </p>
          </div>
          {results.length > 0 ? (
            <div className="space-y-4">
              {results.map((item) => (
                <SearchResultCard key={`${selectedMethod}-${item.pmid}`} item={item} methodLabel={selectedMethodLabel} />
              ))}
            </div>
          ) : (
            <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm leading-7 text-slate-600 backdrop-blur">
              No PubMed results were returned for this query and method. Try a broader biomedical phrase
              or switch retrieval method.
            </section>
          )}
          <AIPolishPanel data={aiPolish} />
        </section>
      ) : null}

      {!loading && !error && viewMode === "topic" ? (
        <section className="space-y-6">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-sea">Topic grouping</p>
            <h2 className="mt-2 font-display text-3xl text-ink">Results grouped by cluster</h2>
          </div>
          {clusters.length > 0 ? (
            <div className="grid gap-5">
              {clusters.map((group) => (
                <section
                  key={group.cluster_id}
                  className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 backdrop-blur"
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Topic cluster</p>
                      <h3 className="mt-2 font-display text-2xl text-ink">
                        {group.representative_keywords.slice(0, 2).join(" / ") || `Cluster ${group.cluster_id}`}
                      </h3>
                    </div>
                    <TopicChip label={`${group.cluster_size} docs`} tone="soft" />
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {group.representative_keywords.map((keyword) => (
                      <TopicChip key={`${group.cluster_id}-${keyword}`} label={keyword} tone="default" />
                    ))}
                  </div>
                  <div className="mt-5 space-y-4">
                    {group.representative_docs.map((item) => (
                      <SearchResultCard
                        key={`${selectedMethod}-${group.cluster_id}-${item.pmid}`}
                        item={item}
                        methodLabel={selectedMethodLabel}
                        clusterLabel={`Cluster ${group.cluster_id}`}
                      />
                    ))}
                  </div>
                </section>
              ))}
            </div>
          ) : (
            <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm leading-7 text-slate-600 backdrop-blur">
              No cluster groups are available for this query yet. Try a broader query or return to
              ranked view.
            </section>
          )}
        </section>
      ) : null}
    </div>
  );
}
