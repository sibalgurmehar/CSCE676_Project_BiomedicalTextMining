import { TopicChip } from "@/components/topic-chip";
import { type AIPolishResponse } from "@/lib/api";

type AIPolishPanelProps = {
  data: AIPolishResponse | null;
  title?: string;
};

export function AIPolishPanel({ data, title = "AI-assisted polish" }: AIPolishPanelProps) {
  if (!data || !data.enabled) {
    return null;
  }

  return (
    <section className="rounded-[1.75rem] border border-coral/20 bg-white/74 p-6 backdrop-blur">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-coral">AI-assisted</p>
          <h2 className="mt-2 font-display text-2xl text-ink">{title}</h2>
        </div>
        <TopicChip label={data.mode.replaceAll("_", " ")} tone="accent" />
      </div>

      <p className="mt-4 text-sm leading-7 text-slate-700">{data.note}</p>

      {data.rewrite_suggestions.length > 0 ? (
        <div className="mt-5">
          <p className="text-sm font-semibold text-ink">Query rewrite suggestions</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {data.rewrite_suggestions.map((item) => (
              <TopicChip key={item} label={item} tone="soft" />
            ))}
          </div>
        </div>
      ) : null}

      {data.suggested_follow_up_queries.length > 0 ? (
        <div className="mt-5">
          <p className="text-sm font-semibold text-ink">Suggested follow-up queries</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {data.suggested_follow_up_queries.map((item) => (
              <TopicChip key={item} label={item} tone="default" />
            ))}
          </div>
        </div>
      ) : null}

      {data.why_this_matched.length > 0 ? (
        <div className="mt-5 space-y-3">
          <p className="text-sm font-semibold text-ink">Why this matched</p>
          {data.why_this_matched.map((item) => (
            <div key={item.pmid} className="rounded-2xl bg-slate-50/70 p-4 text-sm leading-6 text-slate-700">
              <p className="font-medium text-ink">{item.title}</p>
              <p className="mt-1">PMID {item.pmid}</p>
              <p className="mt-2">{item.explanation}</p>
            </div>
          ))}
        </div>
      ) : null}

      {data.refined_cluster_labels.length > 0 ? (
        <div className="mt-5 space-y-3">
          <p className="text-sm font-semibold text-ink">Refined cluster labels</p>
          {data.refined_cluster_labels.map((item) => (
            <div key={item.cluster_id} className="rounded-2xl bg-slate-50/70 p-4 text-sm leading-6 text-slate-700">
              <p className="font-medium text-ink">{item.label}</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {item.keywords.map((keyword) => (
                  <TopicChip key={`${item.cluster_id}-${keyword}`} label={keyword} tone="soft" />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
