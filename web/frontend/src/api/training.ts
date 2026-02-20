/**
 * 训练 API 客户端
 *
 * 提供与后端训练 API 交互的所有方法
 */
import api from './index';

export interface TrainingJobCreate {
  name: string;
  config_path?: string;
  config?: Record<string, any>;
  description?: string;
  tags?: string[];
}

export interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  epoch: number;
  total_epochs: number;
  current_loss?: number;
  current_accuracy?: number;
  best_loss?: number;
  best_accuracy?: number;
  config?: Record<string, any>;
  metrics?: Record<string, any>;
  start_time?: string;
  end_time?: string;
  duration?: number;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface TrainingMetrics {
  epoch: number;
  train_loss: number;
  val_loss?: number;
  train_acc?: number;
  val_acc?: number;
  learning_rate?: number;
  timestamp: string;
}

export interface TrainingJobListParams {
  skip?: number;
  limit?: number;
  status?: string;
  sort_by?: string;
  order?: 'asc' | 'desc';
}

/**
 * 获取训练任务列表
 */
export const listJobs = async (params?: TrainingJobListParams) => {
  const response = await api.get('/training/jobs', { params });
  return response;
};

/**
 * 获取训练任务详情
 */
export const getJob = async (jobId: string): Promise<TrainingJob> => {
  const response = await api.get(`/training/jobs/${jobId}`);
  return response.data;
};

/**
 * 创建训练任务
 */
export const createJob = async (data: TrainingJobCreate) => {
  const response = await api.post('/training/jobs', data);
  return response.data;
};

/**
 * 启动训练任务
 */
export const startJob = async (jobId: string) => {
  const response = await api.post(`/training/jobs/${jobId}/start`);
  return response.data;
};

/**
 * 暂停训练任务
 */
export const pauseJob = async (jobId: string) => {
  const response = await api.post(`/training/jobs/${jobId}/pause`);
  return response.data;
};

/**
 * 恢复训练任务
 */
export const resumeJob = async (jobId: string) => {
  const response = await api.post(`/training/jobs/${jobId}/resume`);
  return response.data;
};

/**
 * 停止训练任务
 */
export const stopJob = async (jobId: string) => {
  const response = await api.post(`/training/jobs/${jobId}/stop`);
  return response.data;
};

/**
 * 取消训练任务
 */
export const cancelJob = async (jobId: string) => {
  const response = await api.post(`/training/jobs/${jobId}/cancel`);
  return response.data;
};

/**
 * 删除训练任务
 */
export const deleteJob = async (jobId: string) => {
  const response = await api.delete(`/training/jobs/${jobId}`);
  return response.data;
};

/**
 * 获取训练指标历史
 */
export const getMetrics = async (jobId: string): Promise<TrainingMetrics[]> => {
  const response = await api.get(`/training/jobs/${jobId}/metrics`);
  return response.data;
};

/**
 * 获取训练日志
 */
export const getLogs = async (jobId: string, lines?: number) => {
  const response = await api.get(`/training/jobs/${jobId}/logs`, {
    params: { lines },
  });
  return response.data;
};

/**
 * 下载训练检查点
 */
export const downloadCheckpoint = async (jobId: string, epoch?: number) => {
  const params = epoch ? { epoch } : {};
  const response = await api.get(`/training/jobs/${jobId}/checkpoint`, {
    params,
    responseType: 'blob',
  });

  // 创建下载链接
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `checkpoint_${jobId}_${epoch || 'latest'}.pth`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};

/**
 * 导出训练 API
 */
export const trainingApi = {
  listJobs,
  getJob,
  createJob,
  startJob,
  pauseJob,
  resumeJob,
  stopJob,
  cancelJob,
  deleteJob,
  getMetrics,
  getLogs,
  downloadCheckpoint,
};

export default trainingApi;
