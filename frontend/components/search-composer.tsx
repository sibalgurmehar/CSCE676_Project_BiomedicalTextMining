import { TopicChip } from "@/components/topic-chip";

type SearchComposerProps = {
  query: string;
  methods: {
    id: string;
    label: string;
  }[];
  featuredQueries: string[];
};

export function SearchComposer({ query, methods, featuredQueries }: SearchComposerProps) {
  return (
    <section className="overflow-hidden rounded-[2rem] border border-white/70 bg-[linear-gradient(145deg,rgba(255,255,255,0.92),rgba(246,231,207,0.65))] p-6 shadow-soft backdrop-blur md:p-8">
      <div className="grid gap-8 lg:grid-cols-[1.25fr_0.75fr]">
        <div>
          <div className="rounded-[1.5rem] border border-slate-200/80 bg-white/90 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.9)]">
            <div className="flex items-center gap-3 rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
              <div className="h-3 w-3 rounded-full bg-coral" />
              <p className="flex-1 text-base text-slate-600 md:text-lg">{query}</p>
              <span className="rounded-full bg-ink px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-white">
                Search
              </span>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {methods.map((method) => (
                <TopicChip key={method.id} label={method.label} tone="soft" />
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-slate-500">Suggested queries</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {featuredQueries.map((item, index) => (
                <TopicChip
                  key={item}
                  label={item}
                  tone={index === 0 ? "accent" : "default"}
                />
              ))}
            </div>
          </div>
          <div className="rounded-[1.5rem] bg-ink p-5 text-white">
            <p className="text-xs uppercase tracking-[0.28em] text-sand/80">Interface stance</p>
            <p className="mt-3 text-sm leading-7 text-slate-200">
              One query surface, one corpus, four retrieval methods. Search stays front and center,
              while comparison, clustering, and evaluation live in adjacent views.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
