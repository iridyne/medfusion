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

export const getComfyUIHealth = async (
  baseUrl?: string,
): Promise<ComfyUIHealthResponse> => {
  const response = await api.get("/comfyui/health", {
    params: baseUrl ? { base_url: baseUrl } : undefined,
  });
  return response.data;
};
