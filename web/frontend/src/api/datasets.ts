/**
 * 数据集 API 客户端
 * 
 * 提供与后端数据集 API 交互的所有方法
 */
import api from './index'

export interface DatasetCreate {
  name: string;
  description?: string;
  data_path: string;
  dataset_type?: "image" | "tabular" | "multimodal";
  status?: "uploading" | "processing" | "ready" | "error";
  num_samples?: number;
  num_classes?: number;
  train_samples?: number;
  val_samples?: number;
  test_samples?: number;
  class_distribution?: Record<string, any>;
  tags?: string[];
  created_by?: string;
}

export interface DatasetUpdate {
  name?: string;
  description?: string;
  num_samples?: number;
  num_classes?: number;
  train_samples?: number;
  val_samples?: number;
  test_samples?: number;
  class_distribution?: Record<string, any>;
  tags?: string[];
}

export interface Dataset {
  id: number;
  name: string;
  description?: string;
  data_path: string;
  dataset_type?: "image" | "tabular" | "multimodal";
  status?: string;
  size_bytes?: number;
  num_samples?: number;
  num_classes?: number;
  train_samples?: number;
  val_samples?: number;
  test_samples?: number;
  class_distribution?: Record<string, any>;
  analysis?: Record<string, any>;
  tags?: string[];
  created_at: string;
  created_by?: string;
}

export interface DatasetListParams {
  skip?: number;
  limit?: number;
  num_classes?: number;
  sort_by?: string;
  order?: 'asc' | 'desc';
}

export interface DatasetStatistics {
  total_datasets: number;
  total_samples: number;
  avg_samples: number;
}

export interface DatasetReadinessCheck {
  key: string;
  label: string;
  status: "pass" | "warning" | "fail";
  detail: string;
}

export interface DatasetInspection {
  path: {
    path: string;
    exists: boolean;
    kind: string;
    size_bytes?: number | null;
    estimated_size_mb?: number | null;
  };
  csv: {
    path?: string | null;
    error?: string | null;
    headers: string[];
    row_count?: number | null;
    preview_rows: Record<string, any>[];
  };
  schema: {
    image_path_column?: string | null;
    target_column?: string | null;
    patient_id_column?: string | null;
    numerical_features?: string[];
    categorical_features?: string[];
    num_classes?: number | null;
    image_dir?: string | null;
  };
  image_probe?: {
    checked: number;
    existing: number;
    missing: number;
    samples: Array<{
      value: string;
      resolved_path: string;
      exists: boolean;
    }>;
  } | null;
  readiness: {
    status: "ready" | "warning" | "blocked";
    can_enter_training: boolean;
    errors: string[];
    warnings: string[];
    checks: DatasetReadinessCheck[];
    summary: {
      headers: number;
      preview_rows: number;
      num_classes?: number | null;
      numerical_features: number;
      categorical_features: number;
    };
    next_step: string;
  };
}

export interface DatasetInspectRequest {
  data_path: string;
  dataset_type?: "image" | "tabular" | "multimodal";
  csv_path?: string;
  image_dir?: string;
  image_path_column?: string;
  target_column?: string;
  patient_id_column?: string;
  numerical_features?: string[];
  categorical_features?: string[];
  num_classes?: number;
}

/**
 * 获取数据集列表
 */
export const getDatasets = async (params?: DatasetListParams) => {
  const response = await api.get('/datasets/', { params })
  return response.data
}

/**
 * 搜索数据集
 */
export const searchDatasets = async (keyword: string, skip = 0, limit = 100) => {
  const response = await api.get('/datasets/search', {
    params: { keyword, skip, limit }
  })
  return response.data
}

/**
 * 获取统计信息
 */
export const getDatasetStatistics = async (): Promise<DatasetStatistics> => {
  const response = await api.get('/datasets/statistics')
  return response.data
}

/**
 * 获取所有类别数
 */
export const getClassCounts = async (): Promise<number[]> => {
  const response = await api.get('/datasets/class-counts')
  return response.data.class_counts
}

/**
 * 获取数据集详情
 */
export const getDataset = async (id: number): Promise<Dataset> => {
  const response = await api.get(`/datasets/${id}`)
  return response.data
}

/**
 * 创建数据集记录
 */
export const createDataset = async (data: DatasetCreate) => {
  const response = await api.post('/datasets/', data)
  return response.data
}

export const inspectDatasetPath = async (
  data: DatasetInspectRequest,
): Promise<DatasetInspection> => {
  const response = await api.post('/datasets/inspect', data)
  return response.data
}

/**
 * 更新数据集信息
 */
export const updateDataset = async (id: number, data: DatasetUpdate) => {
  const response = await api.put(`/datasets/${id}`, data)
  return response.data
}

/**
 * 删除数据集
 */
export const deleteDataset = async (id: number) => {
  const response = await api.delete(`/datasets/${id}`)
  return response.data
}

/**
 * 分析数据集
 */
export const analyzeDataset = async (id: number) => {
  const response = await api.post(`/datasets/${id}/analyze`)
  return response.data
}

export const getDatasetReadiness = async (
  id: number,
): Promise<DatasetInspection & { dataset_id: number; dataset_name: string }> => {
  const response = await api.get(`/datasets/${id}/readiness`)
  return response.data
}

/**
 * 格式化样本数量
 */
export const formatSampleCount = (count?: number): string => {
  if (!count) return 'N/A'
  
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`
  } else if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}K`
  }
  
  return count.toString()
}

/**
 * 计算数据集分割比例
 */
export const calculateSplitRatio = (dataset: Dataset): string => {
  const { train_samples, val_samples, test_samples } = dataset
  
  if (!train_samples && !val_samples && !test_samples) {
    return 'N/A'
  }
  
  const total = (train_samples || 0) + (val_samples || 0) + (test_samples || 0)
  
  if (total === 0) return 'N/A'
  
  const trainRatio = ((train_samples || 0) / total * 100).toFixed(0)
  const valRatio = ((val_samples || 0) / total * 100).toFixed(0)
  const testRatio = ((test_samples || 0) / total * 100).toFixed(0)
  
  return `${trainRatio}/${valRatio}/${testRatio}`
}
