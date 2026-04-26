import { SearchExperience } from "@/components/search-experience";
import { SiteShell } from "@/components/site-shell";
import { apiGet, type DatasetInfoResponse } from "@/lib/api";
import { loadSharedSettings } from "@/lib/settings";

export default async function HomePage() {
  const [settings, datasetInfo] = await Promise.all([
    loadSharedSettings(),
    apiGet<DatasetInfoResponse>("/dataset-info")
  ]);

  return (
    <SiteShell
      eyebrow="Search"
      title={settings.project.name}
      description={settings.project.researchQuestion}
      aside={
        <div className="space-y-4">
          <p className="text-xs uppercase tracking-[0.28em] text-sea">Retrieval frame</p>
          <div className="grid gap-3 text-sm text-slate-700">
            <div className="rounded-2xl bg-mist p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Curated corpus</p>
              <p className="mt-2 font-display text-2xl text-ink">
                {(datasetInfo.stats.pubmed_subset_docs ?? 0).toLocaleString()}
              </p>
              <p className="mt-1">{datasetInfo.retrieval_corpus}</p>
            </div>
            <div className="rounded-2xl bg-mist p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">BioASQ queries</p>
              <p className="mt-2 font-display text-2xl text-ink">
                {(datasetInfo.stats.surviving_queries ?? datasetInfo.stats.bioasq_questions ?? 0).toLocaleString()}
              </p>
              <p className="mt-1">{datasetInfo.unified_dataset_note}</p>
            </div>
          </div>
        </div>
      }
    >
      <SearchExperience
        initialQuery={settings.frontend.defaultQuery}
        methods={settings.retrievalMethods}
      />
    </SiteShell>
  );
}
