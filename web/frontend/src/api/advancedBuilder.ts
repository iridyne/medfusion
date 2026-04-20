import api from "./index";

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
  return response.data;
};
