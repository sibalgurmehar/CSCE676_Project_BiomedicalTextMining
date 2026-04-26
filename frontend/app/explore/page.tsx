import { ExploreWorkspace } from "@/components/explore-workspace";
import { SiteShell } from "@/components/site-shell";
import { apiGet, type SummaryResponse } from "@/lib/api";

export default async function ExplorePage() {
  const diversity = await apiGet<SummaryResponse>("/diversity/summary");
  const themeCoverage = Array.isArray(diversity.summary) ? diversity.summary : [];

  return (
    <SiteShell
      eyebrow="Explore"
      title="Clustered evidence neighborhoods for exploratory reading."
      description="The exploration view uses the curated PubMed subset for open-ended browsing, cluster grouping, and topic-aware diversification."
      aside={
        <div className="space-y-4">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Theme coverage</p>
          {themeCoverage.map((item) => (
            <div
              key={String(item.method)}
              className="flex items-center justify-between text-sm text-slate-700"
            >
              <span>{String(item.method).toUpperCase()}</span>
              <span className="font-semibold text-ink">
                {Number(item.theme_coverage).toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      }
    >
      <ExploreWorkspace initialQuery="diabetes treatment" />
    </SiteShell>
  );
}
