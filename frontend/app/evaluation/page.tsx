import { SiteShell } from "@/components/site-shell";
import { TopicChip } from "@/components/topic-chip";
import {
  apiGet,
  type ArtifactResponse,
  type SummaryResponse
} from "@/lib/api";

type MetricRow = {
  method: string;
  precision_at_5: number;
  precision_at_10: number;
  recall_at_10: number;
  mrr: number;
  ndcg_at_5: number;
  ndcg_at_10: number;
};

type QueryTypeBucket = {
  num_queries: number;
  best_method_by_ndcg_at_10: string;
  metrics: MetricRow[];
};

type ExampleQuery = {
  query_id: string;
  query_text: string;
  query_type: string;
  ndcg_at_10: number;
  mrr: number;
  precision_at_10: number;
};

export default async function EvaluationPage() {
  const [metrics, diversity, queryTypes, examples] = await Promise.all([
    apiGet<SummaryResponse>("/metrics/summary"),
    apiGet<SummaryResponse>("/diversity/summary"),
    apiGet<SummaryResponse>("/query-types/summary"),
    apiGet<ArtifactResponse>("/metrics/examples")
  ]);

  const metricRows = (Array.isArray(metrics.summary) ? metrics.summary : []) as MetricRow[];
  const queryTypeBuckets = queryTypes.summary as Record<string, QueryTypeBucket>;
  const examplePayload = examples.payload as Record<
    string,
    { best_queries: ExampleQuery[]; worst_queries: ExampleQuery[] }
  >;
  const diversityRows = (Array.isArray(diversity.summary) ? diversity.summary : []) as Array<
    Record<string, number | string>
  >;

  return (
    <SiteShell
      eyebrow="Evaluation"
      title="Metrics, query buckets, and method-level behavior."
      description="Evaluation stays anchored to the unified dataset so every method is judged against the same BioASQ relevance labels joined onto curated PubMed documents."
      aside={
        <div className="space-y-4">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Evaluation note</p>
          <p className="text-sm leading-7 text-slate-700">
            Evaluation metrics are computed on PubMed documents with BioASQ expert relevance
            annotations joined by exact PMID.
          </p>
        </div>
      }
    >
      <section className="overflow-x-auto rounded-[1.75rem] border border-slate-200/80 bg-white/72 backdrop-blur">
        <div className="min-w-[980px]">
          <div className="grid grid-cols-[1.15fr,0.7fr,0.7fr,0.7fr,0.7fr,0.7fr,0.7fr] border-b border-slate-200/80 px-6 py-4 text-xs uppercase tracking-[0.24em] text-slate-500">
            <p>Method</p>
            <p>Precision@5</p>
            <p>Precision@10</p>
            <p>Recall@10</p>
            <p>MRR</p>
            <p>nDCG@5</p>
            <p>nDCG@10</p>
          </div>
          {metricRows.map((row) => (
            <div
              key={row.method}
              className="grid grid-cols-[1.15fr,0.7fr,0.7fr,0.7fr,0.7fr,0.7fr,0.7fr] border-b border-slate-200/60 px-6 py-5 text-sm text-slate-700 last:border-b-0"
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

      <section className="mt-8 grid gap-6 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 backdrop-blur">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Narrative</p>
          <div className="mt-4 space-y-4 text-sm leading-7 text-slate-700">
            {metrics.narrative
              ? Object.entries(metrics.narrative).map(([key, value]) => (
                  <div key={key}>
                    <p className="font-semibold capitalize text-ink">{key.replaceAll("_", " ")}</p>
                    <p>{String(value)}</p>
                  </div>
                ))
              : null}
          </div>
        </div>

        <div className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 backdrop-blur">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Query-type summaries</p>
          <div className="mt-4 space-y-4">
            {Object.entries(queryTypeBuckets).map(([bucket, value]) => (
              <div key={bucket} className="border-t border-slate-200/80 pt-4 first:border-t-0 first:pt-0">
                <div className="flex items-center justify-between gap-4">
                  <p className="font-semibold text-ink">{bucket.replaceAll("_", " ")}</p>
                  <TopicChip label={value.best_method_by_ndcg_at_10.toUpperCase()} tone="accent" />
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">
                  {value.num_queries} queries in this bucket. Best method by nDCG@10:{" "}
                  {value.best_method_by_ndcg_at_10.toUpperCase()}.
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 backdrop-blur">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Best and worst queries</p>
          <div className="mt-4 space-y-4">
            {Object.entries(examplePayload).map(([method, payload]) => (
              <div key={method} className="border-t border-slate-200/80 pt-4 first:border-t-0 first:pt-0">
                <div className="flex items-center justify-between gap-4">
                  <p className="font-display text-2xl text-ink">{method.toUpperCase()}</p>
                  <TopicChip label="Example queries" tone="soft" />
                </div>
                <div className="mt-3 space-y-3 text-sm leading-6 text-slate-700">
                  <div>
                    <p className="font-semibold text-ink">Best</p>
                    <p>{payload.best_queries[0]?.query_text ?? "No example available."}</p>
                  </div>
                  <div>
                    <p className="font-semibold text-ink">Worst</p>
                    <p>{payload.worst_queries[0]?.query_text ?? "No example available."}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-6 backdrop-blur">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Diversity summary</p>
          <div className="mt-4 space-y-4 text-sm leading-7 text-slate-700">
            {diversity.narrative
              ? Object.entries(diversity.narrative).map(([key, value]) => (
                  <div key={key}>
                    <p className="font-semibold capitalize text-ink">{key.replaceAll("_", " ")}</p>
                    <p>{String(value)}</p>
                  </div>
                ))
              : null}
          </div>
          <div className="mt-5 flex flex-wrap gap-2">
            {diversityRows.map((row) => (
              <TopicChip
                key={String(row.method)}
                label={`${String(row.method).toUpperCase()} theme coverage ${Number(row.theme_coverage).toFixed(2)}`}
                tone="default"
              />
            ))}
          </div>
        </div>
      </section>
    </SiteShell>
  );
}
