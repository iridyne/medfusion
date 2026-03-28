import type { ReactNode } from "react";

interface PageChip {
  label: string;
  tone?: "neutral" | "teal" | "amber" | "blue" | "rose";
}

interface PageMetric {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "teal" | "amber" | "blue" | "rose";
}

interface PageScaffoldProps {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
  aside?: ReactNode;
  chips?: PageChip[];
  metrics?: PageMetric[];
  children: ReactNode;
}

function toneClassName(tone?: PageChip["tone"]) {
  return tone ? `is-${tone}` : "is-neutral";
}

export default function PageScaffold({
  eyebrow,
  title,
  description,
  actions,
  aside,
  chips,
  metrics,
  children,
}: PageScaffoldProps) {
  return (
    <div className="page-shell">
      <section className="page-hero surface-frame">
        <div className="page-hero__content">
          <span className="page-hero__eyebrow">{eyebrow}</span>

          {chips?.length ? (
            <div className="page-hero__chips">
              {chips.map((chip) => (
                <span
                  key={`${chip.label}-${chip.tone ?? "neutral"}`}
                  className={`page-chip ${toneClassName(chip.tone)}`}
                >
                  {chip.label}
                </span>
              ))}
            </div>
          ) : null}

          <h1 className="page-hero__title">{title}</h1>
          <p className="page-hero__description">{description}</p>

          {actions ? <div className="page-hero__actions">{actions}</div> : null}
        </div>

        {aside ? <aside className="page-hero__aside">{aside}</aside> : null}
      </section>

      {metrics?.length ? (
        <section className="metric-grid">
          {metrics.map((metric) => (
            <article
              key={metric.label}
              className={`metric-card ${toneClassName(metric.tone)}`}
            >
              <span className="metric-card__label">{metric.label}</span>
              <strong className="metric-card__value">{metric.value}</strong>
              {metric.hint ? (
                <span className="metric-card__hint">{metric.hint}</span>
              ) : null}
            </article>
          ))}
        </section>
      ) : null}

      <div className="page-shell__body">{children}</div>
    </div>
  );
}
