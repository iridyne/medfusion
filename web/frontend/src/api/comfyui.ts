import api from "./index";

export interface ComfyUIHealthResponse {
  base_url: string;
  open_url: string;
  recommended_start_command: string;
  handoff_hint: string;
  probe: {
    reachable: boolean;
    status_code: number | null;
    latency_ms: number | null;
    probe_url: string;
    message: string;
    payload_preview?: Record<string, unknown> | null;
  };
}

export interface ComfyUIAdapterProfile {
  id: string;
  label: string;
  description: string;
  blueprint_id: string;
  target_canvas_route: string;
  components: Array<{
    component_id: string;
    label: string;
    family: string;
    family_label: string;
    status: "compile_ready" | "conditional" | "draft_only";
  }>;
  family_chain: Array<{
    family: string;
    label: string;
  }>;
  default_import_prefill: {
    config_path: string;
    checkpoint_path: string;
    output_dir?: string;
    split: "train" | "val" | "test";
    attention_samples: number;
    importance_sample_limit: number;
  };
}

export interface ComfyUIAdapterProfilesResponse {
  mode: string;
  source_boundary: string;
  recommended_steps: string[];
  profiles: ComfyUIAdapterProfile[];
}

export const getComfyUIHealth = async (
  baseUrl?: string,
): Promise<ComfyUIHealthResponse> => {
  const response = await api.get("/comfyui/health", {
    params: baseUrl ? { base_url: baseUrl } : undefined,
  });
  return response.data;
};

export const getComfyUIAdapterProfiles = async (): Promise<ComfyUIAdapterProfilesResponse> => {
  const response = await api.get("/comfyui/adapter-profiles");
  return response.data;
};
