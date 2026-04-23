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

export default function PageScaffold({
  eyebrow,
  title,
  description,
  actions,
  aside,
  metrics,
  children,
}: PageScaffoldProps) {
  return (
    <div className="page-shell">
      <section className={`page-header ${aside ? "page-header--with-aside" : ""}`}>
        <div className="page-header__main surface-frame">
          <div className="page-header__content">
            <span className="page-header__eyebrow">{eyebrow}</span>
            <div className="page-header__headline">
              <h1 className="page-header__title">{title}</h1>
            </div>
            <p className="page-header__description">{description}</p>
          </div>

          {actions ? <div className="page-header__actions">{actions}</div> : null}
        </div>

        {aside ? <aside className="page-header__aside">{aside}</aside> : null}
      </section>

      {metrics?.length ? (
        <section className="metric-grid">
          {metrics.map((metric) => (
            <article
              key={metric.label}
              className="metric-card"
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
