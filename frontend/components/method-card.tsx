type Method = {
  id: string;
  label: string;
  family: string;
  description: string;
};

export function MethodCard({ method }: { method: Method }) {
  return (
    <article className="group rounded-[1.5rem] border border-slate-200 bg-white/70 p-5 transition duration-300 hover:-translate-y-1 hover:border-sea/30 hover:shadow-soft">
      <p className="text-xs uppercase tracking-[0.24em] text-slate-500">{method.family}</p>
      <h3 className="mt-3 font-display text-2xl text-ink">{method.label}</h3>
      <p className="mt-3 text-sm leading-7 text-slate-700">{method.description}</p>
      <div className="mt-6 h-px bg-gradient-to-r from-sea/40 to-transparent" />
      <p className="mt-4 text-xs uppercase tracking-[0.24em] text-sea">
        Method ID: {method.id}
      </p>
    </article>
  );
}

