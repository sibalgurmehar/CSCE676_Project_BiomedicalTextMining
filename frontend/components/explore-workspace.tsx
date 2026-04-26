"use client";

import { useEffect, useState } from "react";
import { AIPolishPanel } from "@/components/ai-polish-panel";
import { apiPost, type AIPolishResponse, type ClusterResponse, type RetrievedDocument } from "@/lib/api";
import { TopicChip } from "@/components/topic-chip";

type ExploreWorkspaceProps = {
  initialQuery: string;
};

const exampleLabels = [
  "Treatment Methods",
  "Clinical Outcomes",
  "Side Effects",
  "Trial Design",
  "Biomarkers"
];

function abstractSnippet(document: RetrievedDocument) {
  const source = document.metadata.abstract || document.retrieval_text || "";
  if (source.length <= 170) {
    return source;
  }
  return `${source.slice(0, 167)}...`;
}

export function ExploreWorkspace({ initialQuery }: ExploreWorkspaceProps) {
  const [query, setQuery] = useState(initialQuery ?? "");
  const [clusters, setClusters] = useState<ClusterResponse["cluster_summaries"]>([]);
  const [silhouetteScore, setSilhouetteScore] = useState<number | null>(null);
  const [aiPolish, setAiPolish] = useState<AIPolishResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runClusters(nextQuery: string) {
    const normalizedQuery = typeof nextQuery === "string" ? nextQuery.trim() : "";
    if (normalizedQuery.length < 2) {
      setError("Enter at least 2 characters before clustering.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [response, polish] = await Promise.all([
        apiPost<ClusterResponse>("/clusters", {
          query: normalizedQuery,
          method: "bm25",
          dataset_name: "pubmed_subset",
          top_k: 30,
          num_clusters: 5,
          vector_space: "tfidf"
        }),
        apiPost<AIPolishResponse>("/ai-polish", {
          query: normalizedQuery,
          method: "bm25",
          dataset_name: "pubmed_subset",
          top_k: 3,
          include_matches: false,
          refine_clusters: true
        })
      ]);

      setClusters(response.cluster_summaries);
      setSilhouetteScore(response.silhouette_score);
      setAiPolish(polish);
      if (response.status !== "success" && response.error) {
        setError(response.error);
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Cluster request failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (typeof initialQuery === "string" && initialQuery.trim().length >= 2) {
      void runClusters(initialQuery);
    }
  }, [initialQuery]);

  return (
    <div className="space-y-8">
      <section className="rounded-[2rem] border border-white/70 bg-[linear-gradient(145deg,rgba(255,255,255,0.94),rgba(237,244,245,0.72))] p-6 shadow-soft backdrop-blur md:p-8">
        <div className="grid gap-6 lg:grid-cols-[1.08fr_0.92fr]">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-sea">Clustered exploration</p>
            <div className="mt-4 rounded-[1.5rem] border border-slate-200/80 bg-white/92 p-4">
              <div className="flex flex-col gap-3 rounded-[1.25rem] border border-slate-200 bg-slate-50/75 p-4 md:flex-row md:items-center">
                <div className="h-3 w-3 rounded-full bg-coral" />
                <input
                  aria-label="Explore query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  className="min-w-0 flex-1 bg-transparent text-base text-slate-700 outline-none md:text-lg"
                />
                <button
                  onClick={() => void runClusters(query)}
                  className="rounded-full bg-ink px-5 py-3 text-xs font-semibold uppercase tracking-[0.24em] text-white"
                >
                  Cluster
                </button>
              </div>
            </div>
          </div>

          <div className="rounded-[1.5rem] bg-ink p-5 text-white">
            <p className="text-xs uppercase tracking-[0.28em] text-sand/80">What this shows</p>
            <p className="mt-3 text-sm leading-7 text-slate-200">
              Top PubMed results are grouped into cluster neighborhoods returned by the backend
              clustering pipeline.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {exampleLabels.map((label) => (
                <TopicChip key={label} label={label} tone="default" />
              ))}
            </div>
            {silhouetteScore !== null ? (
              <p className="mt-4 text-sm text-slate-300">
                Silhouette score: {silhouetteScore.toFixed(4)}
              </p>
            ) : null}
          </div>
        </div>
      </section>

      {loading ? (
        <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm text-slate-600 backdrop-blur">
          Loading clustered results...
        </section>
      ) : null}

      {error ? (
        <section className="rounded-[1.75rem] border border-coral/30 bg-coral/10 p-6 text-sm text-coral">
          {error}
        </section>
      ) : null}

      {!loading && !error ? (
        <section className="space-y-6">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-sea">Topic clusters</p>
              <h2 className="mt-2 font-display text-3xl text-ink">
                Top PubMed results grouped by theme
              </h2>
            </div>
            <p className="max-w-xl text-sm leading-6 text-slate-600">
              Each cluster comes from the backend clustering service and includes representative
              keywords, cluster size, and top documents.
            </p>
          </div>

          {clusters.length > 0 ? (
            <div className="grid gap-5">
              {clusters.map((cluster) => (
                <section
                  key={cluster.cluster_id}
                  className="rounded-[1.75rem] border border-slate-200/80 bg-white/74 p-6 backdrop-blur"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                        Cluster {cluster.cluster_id}
                      </p>
                      <h3 className="mt-2 font-display text-3xl text-ink">
                        {cluster.representative_keywords.slice(0, 2).join(" / ") || `Cluster ${cluster.cluster_id}`}
                      </h3>
                      <div className="mt-4 flex flex-wrap gap-2">
                        {cluster.representative_keywords.map((keyword) => (
                          <TopicChip key={`${cluster.cluster_id}-${keyword}`} label={keyword} tone="soft" />
                        ))}
                      </div>
                    </div>

                    <div className="rounded-[1.2rem] bg-mist px-4 py-3 text-sm text-slate-700">
                      <span className="font-semibold text-ink">{cluster.cluster_size}</span> results
                    </div>
                  </div>

                  <div className="mt-6 grid gap-4 md:grid-cols-3">
                    {cluster.representative_docs.map((document) => (
                      <article
                        key={`${cluster.cluster_id}-${document.pmid}`}
                        className="rounded-[1.25rem] border border-slate-200/80 bg-slate-50/70 p-4"
                      >
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          PMID {document.pmid}
                        </p>
                        <h4 className="mt-3 text-sm font-semibold leading-6 text-ink">
                          {document.metadata.title || document.retrieval_text}
                        </h4>
                        <p className="mt-3 text-sm leading-6 text-slate-600">
                          {abstractSnippet(document)}
                        </p>
                      </article>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          ) : (
            <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 text-sm leading-7 text-slate-600 backdrop-blur">
              No clusters were returned for this query. Try a different biomedical phrase or a broader
              topic.
            </section>
          )}
          <AIPolishPanel data={aiPolish} title="AI-assisted cluster refinement" />
        </section>
      ) : null}
    </div>
  );
}
