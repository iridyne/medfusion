import type {
  CustomModelEntry,
  ModelCatalogComponent,
  ModelCatalogTemplate,
} from "@/api/models";

const CUSTOM_MODEL_SCHEMA_VERSION = "0.1";

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

const COMPUTE_ORDER: Record<string, number> = {
  light: 1,
  medium: 2,
  heavy: 3,
};

function mergeComputeProfile(
  model: ModelCatalogTemplate,
  selectedUnits: ModelCatalogComponent[],
) {
  const profiles = [
    model.compute_profile,
    ...selectedUnits.map((item) => item.compute_profile),
  ];
  const heaviest = profiles.reduce((current, candidate) =>
    (COMPUTE_ORDER[candidate.tier] || 0) > (COMPUTE_ORDER[current.tier] || 0)
      ? candidate
      : current,
  );

  return {
    tier: heaviest.tier,
    gpu_vram_hint: heaviest.gpu_vram_hint,
    notes: uniqueStrings(profiles.map((item) => item.notes)).join(" / "),
  };
}

export function composeCustomModelEntry(params: {
  id?: string;
  name: string;
  description?: string;
  baseModel: ModelCatalogTemplate;
  unitSelections: Record<string, string>;
  allUnits: ModelCatalogComponent[];
}): CustomModelEntry {
  const selectedUnits = params.allUnits.filter((unit) =>
    Object.values(params.unitSelections).includes(unit.id),
  );
  const mergedPrefill = {
    ...(params.baseModel.wizard_prefill || {}),
    ...selectedUnits.reduce<Record<string, any>>((acc, unit) => {
      Object.assign(acc, unit.wizard_prefill || {});
      return acc;
    }, {}),
    modelTemplateId: params.baseModel.id,
    customModelLabel: params.name,
  };

  const now = new Date().toISOString();
  const id = params.id || `custom-${Date.now()}`;
  return {
    schema_version: CUSTOM_MODEL_SCHEMA_VERSION,
    id,
    source: "custom",
    label: params.name,
    description: params.description || "",
    status: "local_custom",
    based_on_model_id: params.baseModel.id,
    unit_map: params.unitSelections,
    editable_slots: params.baseModel.editable_slots || [],
    component_ids: Object.values(params.unitSelections),
    data_requirements: uniqueStrings([
      ...params.baseModel.data_requirements,
      ...selectedUnits.flatMap((unit) => unit.data_requirements),
    ]),
    compute_profile: mergeComputeProfile(params.baseModel, selectedUnits),
    wizard_prefill: mergedPrefill,
    created_at: now,
    updated_at: now,
  };
}
