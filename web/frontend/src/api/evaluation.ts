import api from "./index";

export interface EvaluationRequest {
  config_path: string;
  checkpoint_path: string;
  output_dir?: string;
  split?: "train" | "val" | "test" | "all";
  attention_samples?: number;
  enable_survival?: boolean;
  survival_time_column?: string;
  survival_event_column?: string;
  enable_importance?: boolean;
  importance_sample_limit?: number;
  import_to_model_library?: boolean;
  name?: string;
  description?: string;
  tags?: string[];
}

export interface EvaluationResponse {
  status: "completed";
  mode: "evaluate_only" | "evaluate_and_import";
  output_dir: string;
  artifact_paths: Record<string, string>;
  metrics: Record<string, any>;
  validation: Record<string, any>;
  summary: {
    experiment_name: string;
    split: string;
    sample_count?: number;
    accuracy?: number;
    auc?: number | null;
    macro_f1?: number;
  };
  model_library_import: {
    imported: boolean;
    model_id: number | null;
    model_name: string | null;
  };
  next_step: string;
}

export async function runIndependentEvaluation(
  payload: EvaluationRequest,
): Promise<EvaluationResponse> {
  const response = await api.post("/evaluation/run", payload);
  return response.data;
}
