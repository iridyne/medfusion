export const WORKBENCH_FALLBACK_SOURCES = [
  "workflow",
  "experiments",
  "preprocessing",
  "settings",
] as const;

export type WorkbenchFallbackSource = (typeof WORKBENCH_FALLBACK_SOURCES)[number];

const WORKBENCH_FALLBACK_SET = new Set<string>(WORKBENCH_FALLBACK_SOURCES);

export function isWorkbenchFallbackSource(
  value: string | null,
): value is WorkbenchFallbackSource {
  return value !== null && WORKBENCH_FALLBACK_SET.has(value);
}

export function buildWorkbenchFallbackSearch(
  source: WorkbenchFallbackSource,
): string {
  const params = new URLSearchParams({ from: source });
  return params.toString();
}

export function consumeWorkbenchFallback(
  searchParams: URLSearchParams,
): {
  source: WorkbenchFallbackSource | null;
  nextSearchParams: URLSearchParams;
} {
  const nextSearchParams = new URLSearchParams(searchParams);
  const rawSource = nextSearchParams.get("from");
  nextSearchParams.delete("from");

  return {
    source: isWorkbenchFallbackSource(rawSource) ? rawSource : null,
    nextSearchParams,
  };
}
