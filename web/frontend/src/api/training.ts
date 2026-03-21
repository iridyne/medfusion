/**
 * 训练 API 客户端
 *
 * 当前与后端 `med_core.web.api.training` 路由对齐：
 * - GET  /api/training/jobs
 * - GET  /api/training/{job_id}/status
 * - POST /api/training/start
 * - POST /api/training/{job_id}/pause
 * - POST /api/training/{job_id}/resume
 * - POST /api/training/{job_id}/stop
 */
import api from "./index";

export interface TrainingJobCreate {
  experiment_name: string;
  training_model_config: Record<string, any>;
  dataset_config: Record<string, any>;
  training_config: Record<string, any>;
}

export interface TrainingJob {
  id: string;
  name: string;
  status: "queued" | "running" | "paused" | "completed" | "failed" | "stopped";
  progress: number;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  startTime: string;
}

interface BackendTrainingJob {
  id: number;
  job_id: string;
  experiment_name: string;
  dataset_name?: string | null;
  backbone?: string | null;
  status: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_loss: number | null;
  current_accuracy: number | null;
  created_at: string;
}

interface BackendTrainingStatus {
  job_id: string;
  experiment_name: string;
  dataset_name?: string | null;
  backbone?: string | null;
  status: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_loss: number | null;
  current_accuracy: number | null;
}

function normalizeJob(job: BackendTrainingJob | BackendTrainingStatus): TrainingJob {
  const jobId = "job_id" in job ? job.job_id : String((job as any).id ?? "");
  return {
    id: jobId,
    name: job.experiment_name || jobId,
    status: job.status as TrainingJob["status"],
    progress: Math.round(job.progress ?? 0),
    epoch: job.current_epoch ?? 0,
    totalEpochs: job.total_epochs ?? 0,
    loss: job.current_loss ?? 0,
    accuracy: job.current_accuracy ?? 0,
    startTime: (job as BackendTrainingJob).created_at ?? "",
  };
}

/**
 * 获取训练任务列表
 */
export const listJobs = async (): Promise<TrainingJob[]> => {
  const response = await api.get<BackendTrainingJob[]>("/training/jobs");
  return response.data.map(normalizeJob);
};

/**
 * 获取训练任务详情
 */
export const getJob = async (jobId: string): Promise<TrainingJob> => {
  const response = await api.get<BackendTrainingStatus>(`/training/${jobId}/status`);
  return normalizeJob(response.data);
};

/**
 * 创建并启动训练任务
 */
export const createJob = async (data: TrainingJobCreate) => {
  const response = await api.post("/training/start", data);
  return response.data;
};

/**
 * 与后端保持兼容：等同 createJob
 */
export const startJob = async (data: TrainingJobCreate) => {
  return createJob(data);
};

/**
 * 暂停训练任务
 */
export const pauseJob = async (jobId: string) => {
  const response = await api.post(`/training/${jobId}/pause`);
  return response.data;
};

/**
 * 恢复训练任务
 */
export const resumeJob = async (jobId: string) => {
  const response = await api.post(`/training/${jobId}/resume`);
  return response.data;
};

/**
 * 停止训练任务
 */
export const stopJob = async (jobId: string) => {
  const response = await api.post(`/training/${jobId}/stop`);
  return response.data;
};

export const trainingApi = {
  listJobs,
  getJob,
  createJob,
  startJob,
  pauseJob,
  resumeJob,
  stopJob,
};

export default trainingApi;
