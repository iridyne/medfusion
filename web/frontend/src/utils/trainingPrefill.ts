export interface TrainingLaunchPrefill {
  experimentName: string;
  backbone: string;
  numClasses: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
}

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
