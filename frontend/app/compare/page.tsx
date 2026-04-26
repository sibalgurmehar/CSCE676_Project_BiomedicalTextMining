import { CompareWorkspace } from "@/components/compare-workspace";
import { SiteShell } from "@/components/site-shell";
import { loadSharedSettings } from "@/lib/settings";

export default async function ComparePage() {
  const settings = await loadSharedSettings();

  return (
    <SiteShell
      eyebrow="Compare"
      title="One biomedical query, four retrieval behaviors."
      description="The compare view keeps the unified dataset in focus so relevance, ranking style, and retrieval tradeoffs stay legible."
      aside={
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Current comparison query</p>
          <p className="mt-3 font-display text-3xl text-ink">{settings.frontend.defaultQuery}</p>
          <p className="mt-4 text-sm leading-7 text-slate-700">
            The same query is pushed through TF-IDF, BM25, Dense, and Hybrid so the top-5 result
            lists can be inspected side by side.
          </p>
        </div>
      }
    >
      <CompareWorkspace initialQuery={settings.frontend.defaultQuery} />
    </SiteShell>
  );
}
