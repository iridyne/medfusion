import api from "./index";
import type {
  AdvancedBuilderBlueprint,
  AdvancedBuilderComponent,
  AdvancedBuilderConnectionRule,
  AdvancedBuilderFamily,
  AdvancedBuilderStatus,
} from "@/config/advancedBuilderCatalog";

export interface AdvancedBuilderCompileIssue {
  level: "error" | "warning";
  code?: string;
  message: string;
  path?: string;
  context?: Record<string, unknown>;
  suggestion?: string;
}

export interface AdvancedBuilderCompileResponse {
  preset: "quickstart" | "clinical" | "showcase";
  run_spec: Record<string, unknown> | null;
  experiment_config: Record<string, unknown> | null;
  contract_validation:
    | {
        ok: boolean;
        errors: Array<{
          path: string;
          message: string;
          error_code: string;
          suggestion?: string | null;
        }>;
        warnings: Array<{
          path: string;
          message: string;
          error_code: string;
          suggestion?: string | null;
        }>;
      }
    | null;
  mainline_contract:
    | {
        schema_family: string;
        output_dir: string;
        model: {
          model_type: string;
          vision_backbone: string;
          fusion_type: string;
          num_classes: number;
        };
      }
    | null;
  issues: AdvancedBuilderCompileIssue[];
  chosen_components: Record<string, string>;
}

export interface AdvancedBuilderCatalogResponse {
  familyLabels: Record<AdvancedBuilderFamily, string>;
  statusLabels: Record<AdvancedBuilderStatus, string>;
  components: AdvancedBuilderComponent[];
  connectionRules: AdvancedBuilderConnectionRule[];
  blueprints: AdvancedBuilderBlueprint[];
}

export const compileAdvancedBuilder = async (payload: {
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
  blueprint_id?: string;
}): Promise<AdvancedBuilderCompileResponse> => {
  const response = await api.post("/advanced-builder/compile", payload);
  return response.data;
};

export const startTrainingFromAdvancedBuilder = async (payload: {
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
  blueprint_id?: string;
}) => {
  const response = await api.post("/advanced-builder/start-training", payload);
  return response.data;
};

export const getAdvancedBuilderCatalog = async () => {
  const response = await api.get("/advanced-builder/catalog");
  const data = response.data;
  return {
    familyLabels: data.families,
    statusLabels: data.status_labels,
    components: data.components.map((item: any) => ({
      id: item.id,
      family: item.family,
      label: item.label,
      status: item.status,
      description: item.description,
      schemaPath: item.schema_path,
      inputs: item.inputs || [],
      outputs: item.outputs || [],
      notes: item.notes || [],
      advancedBuilderContract: item.advanced_builder_contract,
    })),
    connectionRules: data.connection_rules.map((item: any) => ({
      fromFamily: item.from_family,
      toFamily: item.to_family,
      status: item.status,
      description: item.description,
    })),
    blueprints: data.blueprints.map((item: any) => ({
      id: item.id,
      label: item.label,
      status: item.status,
      description: item.description,
      components: item.components,
      compilesTo: item.compiles_to,
      blockers: item.blockers,
      recommendedPreset: item.recommended_preset,
    })),
  } satisfies AdvancedBuilderCatalogResponse;
};
