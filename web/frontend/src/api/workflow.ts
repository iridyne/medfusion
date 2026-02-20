import api from "./index";

export interface WorkflowData {
  nodes: any[];
  edges: any[];
}

export interface Workflow {
  id?: string;
  name: string;
  description?: string;
  nodes: any[];
  edges: any[];
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface WorkflowStatus {
  workflow_id: string;
  status: {
    nodes: Record<string, { status: string; error?: string }>;
    total: number;
    completed: number;
    failed: number;
  };
  results?: any;
}

export interface ExecuteRequest {
  workflow: WorkflowData;
  name?: string;
}

export interface ExecuteResponse {
  workflow_id: string;
  status: string;
  message: string;
}

export const getNodes = async () => {
  const response = await api.get("/workflows/nodes");
  return response.data;
};

export const createWorkflow = async (workflow: Workflow) => {
  const response = await api.post("/workflows/", workflow);
  return response.data;
};

export const getWorkflow = async (workflowId: string) => {
  const response = await api.get(`/workflows/${workflowId}`);
  return response.data;
};

export const validateWorkflow = async (
  workflow: WorkflowData,
): Promise<ValidationResult> => {
  const response = await api.post("/api/workflows/validate", { workflow });
  return response.data;
};

export const executeWorkflow = async (
  request: ExecuteRequest,
): Promise<ExecuteResponse> => {
  const response = await api.post("/api/workflows/execute", request);
  return response.data;
};

export const getWorkflowStatus = async (
  workflowId: string,
): Promise<WorkflowStatus> => {
  const response = await api.get(`/api/workflows/${workflowId}/status`);
  return response.data;
};

export const deleteWorkflow = async (workflowId: string) => {
  const response = await api.delete(`/workflows/${workflowId}`);
  return response.data;
};
