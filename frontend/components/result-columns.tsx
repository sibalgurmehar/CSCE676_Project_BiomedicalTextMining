type RetrievedDocument = {
  pmid: string;
  title: string;
  score: number;
  cluster: string;
  relevant: boolean;
};

type MethodResults = {
  method: string;
  top_documents: RetrievedDocument[];
};

export function ResultColumns({ results }: { results: MethodResults[] }) {
  return (
    <div className="mt-6 grid gap-4 xl:grid-cols-2">
      {results.map((group) => (
        <div key={group.method} className="rounded-[1.5rem] bg-mist p-4">
          <div className="flex items-center justify-between gap-4">
            <h3 className="font-display text-xl text-ink">{group.method}</h3>
            <span className="text-xs uppercase tracking-[0.24em] text-slate-500">
              Top {group.top_documents.length}
            </span>
          </div>
          <div className="mt-4 space-y-3">
            {group.top_documents.map((doc) => (
              <article key={`${group.method}-${doc.pmid}`} className="rounded-2xl bg-white p-4">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-xs uppercase tracking-[0.18em] text-slate-500">
                    PMID {doc.pmid}
                  </span>
                  <span
                    className={`rounded-full px-3 py-1 text-[11px] font-medium uppercase tracking-[0.16em] ${
                      doc.relevant
                        ? "bg-sea/10 text-sea"
                        : "bg-slate-200 text-slate-600"
                    }`}
                  >
                    {doc.relevant ? "Relevant" : "Not labeled"}
                  </span>
                </div>
                <p className="mt-3 text-sm font-semibold leading-6 text-ink">{doc.title}</p>
                <div className="mt-3 flex items-center justify-between text-xs uppercase tracking-[0.16em] text-slate-500">
                  <span>Score {doc.score.toFixed(2)}</span>
                  <span>Cluster {doc.cluster}</span>
                </div>
              </article>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

