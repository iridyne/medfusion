export interface TrainingLaunchPrefill {
  experimentName: string;
  backbone: string;
  numClasses: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
}

export type TrainingLaunchSource =
  | "guided-start"
  | "comfyui-bridge"
  | "run-wizard"
  | "model-library"
  | "training-monitor"
  | null;

function parsePositiveNumber(raw: string | null): number | undefined {
  if (!raw) {
    return undefined;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return undefined;
  }

  return parsed;
}

export function buildTrainingPrefillQuery(
  prefill: TrainingLaunchPrefill,
): string {
  const params = new URLSearchParams({
    action: "start",
    experimentName: prefill.experimentName,
    backbone: prefill.backbone,
    numClasses: String(prefill.numClasses),
    epochs: String(prefill.epochs),
    batchSize: String(prefill.batchSize),
    learningRate: String(prefill.learningRate),
  });

  return params.toString();
}

export function parseTrainingPrefillParams(
  searchParams: URLSearchParams,
  backboneOptions: readonly string[],
): Partial<TrainingLaunchPrefill> {
  const prefill: Partial<TrainingLaunchPrefill> = {};
  const experimentName = searchParams.get("experimentName");
  const backbone = searchParams.get("backbone");
  const numClasses = parsePositiveNumber(searchParams.get("numClasses"));
  const epochs = parsePositiveNumber(searchParams.get("epochs"));
  const batchSize = parsePositiveNumber(searchParams.get("batchSize"));
  const learningRate = parsePositiveNumber(searchParams.get("learningRate"));

  if (experimentName) {
    prefill.experimentName = experimentName;
  }
  if (backbone && backboneOptions.includes(backbone)) {
    prefill.backbone = backbone;
  }
  if (numClasses !== undefined) {
    prefill.numClasses = numClasses;
  }
  if (epochs !== undefined) {
    prefill.epochs = epochs;
  }
  if (batchSize !== undefined) {
    prefill.batchSize = batchSize;
  }
  if (learningRate !== undefined) {
    prefill.learningRate = learningRate;
  }

  return prefill;
}

export function consumeTrainingLaunchParams(
  searchParams: URLSearchParams,
  backboneOptions: readonly string[],
): {
  source: TrainingLaunchSource;
  prefill: Partial<TrainingLaunchPrefill>;
  nextSearchParams: URLSearchParams;
} {
  const rawSource = searchParams.get("source");
  const source: TrainingLaunchSource =
    rawSource === "guided-start" ||
    rawSource === "comfyui-bridge" ||
    rawSource === "run-wizard" ||
    rawSource === "model-library" ||
    rawSource === "training-monitor"
      ? rawSource
      : null;
  const nextSearchParams = new URLSearchParams(searchParams);

  nextSearchParams.delete("action");
  nextSearchParams.delete("source");
  nextSearchParams.delete("experimentName");
  nextSearchParams.delete("backbone");
  nextSearchParams.delete("numClasses");
  nextSearchParams.delete("epochs");
  nextSearchParams.delete("batchSize");
  nextSearchParams.delete("learningRate");

  return {
    source,
    prefill: parseTrainingPrefillParams(searchParams, backboneOptions),
    nextSearchParams,
  };
}
