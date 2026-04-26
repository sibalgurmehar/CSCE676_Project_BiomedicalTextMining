type TopicChipProps = {
  label: string;
  tone?: "default" | "soft" | "accent";
};

const toneClasses = {
  default: "bg-white text-slate-700 ring-1 ring-slate-200",
  soft: "bg-mist text-sea",
  accent: "bg-coral/10 text-coral"
};

export function TopicChip({ label, tone = "default" }: TopicChipProps) {
  return (
    <span className={`inline-flex rounded-full px-3 py-1.5 text-xs font-medium ${toneClasses[tone]}`}>
      {label}
    </span>
  );
}
