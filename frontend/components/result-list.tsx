import { TopicChip } from "@/components/topic-chip";

type ResultListProps = {
  title: string;
  caption: string;
  items: {
    pmid: string;
    title: string;
    score: number;
    cluster: string;
    relevant: boolean;
  }[];
};

export function ResultList({ title, caption, items }: ResultListProps) {
  return (
    <section className="rounded-[1.75rem] border border-slate-200/80 bg-white/72 p-5 backdrop-blur">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-2xl text-ink">{title}</h2>
          <p className="text-sm text-slate-600">{caption}</p>
        </div>
      </div>
      <div className="mt-5 space-y-4">
        {items.map((item) => (
          <article key={`${title}-${item.pmid}`} className="border-t border-slate-200/80 pt-4 first:border-t-0 first:pt-0">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">PMID {item.pmid}</p>
                <h3 className="mt-2 text-sm font-semibold leading-6 text-ink">{item.title}</h3>
              </div>
              <p className="text-sm font-medium text-sea">{item.score.toFixed(2)}</p>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <TopicChip label={item.cluster} tone="soft" />
              <TopicChip label={item.relevant ? "BioASQ relevant" : "Exploration doc"} tone={item.relevant ? "accent" : "default"} />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
