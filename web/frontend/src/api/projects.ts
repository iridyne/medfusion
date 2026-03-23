import api from "./index";

export type ProjectTaskType =
  | "binary_classification"
  | "cox_survival"
  | "multimodal_research";

export interface ProjectTemplate {
  id: string;
  name: string;
  task_type: ProjectTaskType;
  description: string;
  required_fields: string[];
  recommended_backbone?: string;
  recommended_fusion?: string;
  expected_outputs: string[];
  warnings: string[];
}

export interface ProjectSummaryItem {
  id: number;
  name: string;
  architecture?: string;
  accuracy?: number | null;
  loss?: number | null;
  checkpoint_path?: string;
  created_at: string;
}

export interface ProjectRunSummary {
  id: number;
  job_id: string;
  experiment_name?: string;
  backbone?: string;
  status: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_loss?: number | null;
  current_accuracy?: number | null;
  created_at: string;
}

export interface Project {
  id: number;
  name: string;
  description?: string | null;
  task_type: ProjectTaskType;
  template_id: string;
  status: string;
  dataset_id?: number | null;
  dataset_name?: string | null;
  config_path?: string | null;
  output_dir?: string | null;
  latest_job_id?: string | null;
  latest_model_id?: number | null;
  tags: string[];
  project_meta: Record<string, any>;
  job_count: number;
  model_count: number;
  latest_job?: ProjectRunSummary | null;
  latest_model?: ProjectSummaryItem | null;
  jobs?: ProjectRunSummary[];
  models?: ProjectSummaryItem[];
  created_at: string;
  updated_at?: string | null;
}

export interface ProjectCreate {
  name: string;
  description?: string;
  task_type: ProjectTaskType;
  template_id: string;
  dataset_id?: number;
  config_path?: string;
  output_dir?: string;
  tags?: string[];
  project_meta?: Record<string, any>;
  status?: string;
}

export interface ProjectUpdate extends Partial<ProjectCreate> {
  latest_job_id?: string;
  latest_model_id?: number;
}

export const getProjects = async (params?: {
  skip?: number;
  limit?: number;
  task_type?: string;
  status?: string;
}) => {
  const response = await api.get<Project[]>("/projects/", { params });
  return response.data;
};

export const getProject = async (projectId: number) => {
  const response = await api.get<Project>(`/projects/${projectId}`);
  return response.data;
};

export const createProject = async (payload: ProjectCreate) => {
  const response = await api.post<Project>("/projects/", payload);
  return response.data;
};

export const updateProject = async (projectId: number, payload: ProjectUpdate) => {
  const response = await api.patch<Project>(`/projects/${projectId}`, payload);
  return response.data;
};

export const deleteProject = async (projectId: number) => {
  const response = await api.delete(`/projects/${projectId}`);
  return response.data;
};

export const getProjectRuns = async (projectId: number) => {
  const response = await api.get<{ jobs: ProjectRunSummary[]; models: ProjectSummaryItem[] }>(
    `/projects/${projectId}/runs`,
  );
  return response.data;
};

export const getProjectTemplates = async () => {
  const response = await api.get<{ templates: ProjectTemplate[] }>("/projects/templates");
  return response.data.templates;
};

export const getProjectTemplate = async (templateId: string) => {
  const response = await api.get<ProjectTemplate>(`/projects/templates/${templateId}`);
  return response.data;
};

export const exportProjectBundle = async (projectId: number) => {
  const response = await api.post<{ archive_path: string; download_url: string }>(
    `/projects/${projectId}/export`,
  );
  return response.data;
};
