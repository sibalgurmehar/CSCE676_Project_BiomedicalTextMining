import Link from "next/link";
import { ReactNode } from "react";

const navigation = [
  { href: "/", label: "Search" },
  { href: "/compare", label: "Compare" },
  { href: "/explore", label: "Explore" },
  { href: "/evaluation", label: "Evaluation" },
];

type SiteShellProps = {
  children: ReactNode;
  eyebrow?: string;
  title?: string;
  description?: string;
  aside?: ReactNode;
};

export function SiteShell({
  children,
  eyebrow,
  title,
  description,
  aside
}: SiteShellProps) {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/75 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-6 py-4 md:px-10 lg:px-14">
          <div>
            <Link href="/" className="font-display text-2xl font-semibold tracking-tight text-ink">
              BioSeek
            </Link>
            <p className="text-xs uppercase tracking-[0.24em] text-slate-500">
              Biomedical retrieval
            </p>
          </div>
          <nav className="flex flex-wrap items-center justify-end gap-2 text-sm text-slate-600">
            {navigation.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="rounded-full px-4 py-2 transition hover:bg-white hover:text-ink"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 pb-16 pt-8 md:px-10 lg:px-14 lg:pt-10">
        {(title || description || aside) && (
          <section className="mb-10 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="space-y-4">
              {eyebrow ? (
                <p className="text-xs uppercase tracking-[0.32em] text-sea">{eyebrow}</p>
              ) : null}
              {title ? (
                <h1 className="max-w-4xl font-display text-4xl font-semibold tracking-tight text-ink md:text-6xl">
                  {title}
                </h1>
              ) : null}
              {description ? (
                <p className="max-w-3xl text-lg leading-8 text-slate-700">{description}</p>
              ) : null}
            </div>
            {aside ? (
              <div className="rounded-[1.75rem] border border-white/70 bg-white/72 p-6 shadow-soft backdrop-blur">
                {aside}
              </div>
            ) : null}
          </section>
        )}
        {children}
      </main>
    </div>
  );
}
