/**
 * 模型 API 客户端
 * 
 * 提供与后端模型 API 交互的所有方法
 */
import api from './index'

export interface ModelCreate {
  name: string;
  description?: string;
  backbone: string;
  num_classes: number;
  accuracy?: number;
  loss?: number;
  metrics?: Record<string, any>;
  format?: string;
  input_shape?: number[];
  trained_epochs?: number;
  tags?: string[];
}

export interface ModelUpdate {
  name?: string;
  description?: string;
  accuracy?: number;
  loss?: number;
  metrics?: Record<string, any>;
  tags?: string[];
}

export interface Model {
  id: number;
  name: string;
  description?: string;
  backbone: string;
  num_classes: number;
  input_shape?: number[];
  accuracy?: number;
  loss?: number;
  metrics?: Record<string, any>;
  model_path: string;
  file_size?: number;
  format?: string;
  training_job_id?: number;
  trained_epochs?: number;
  tags?: string[];
  created_at: string;
  created_by?: string;
}

export interface ModelListParams {
  skip?: number;
  limit?: number;
  backbone?: string;
  format?: string;
  sort_by?: string;
  order?: 'asc' | 'desc';
}

export interface ModelStatistics {
  total_models: number;
  total_size: number;
  avg_accuracy: number;
}

/**
 * 获取模型列表
 */
export const getModels = async (params?: ModelListParams) => {
  const response = await api.get('/models/', { params })
  return response.data
}

/**
 * 搜索模型
 */
export const searchModels = async (keyword: string, skip = 0, limit = 100) => {
  const response = await api.get('/models/search', {
    params: { keyword, skip, limit }
  })
  return response.data
}

/**
 * 获取统计信息
 */
export const getModelStatistics = async (): Promise<ModelStatistics> => {
  const response = await api.get('/models/statistics')
  return response.data
}

/**
 * 获取所有 Backbone
 */
export const getBackbones = async (): Promise<string[]> => {
  const response = await api.get('/models/backbones')
  return response.data.backbones
}

/**
 * 获取所有格式
 */
export const getFormats = async (): Promise<string[]> => {
  const response = await api.get('/models/formats')
  return response.data.formats
}

/**
 * 获取模型详情
 */
export const getModel = async (id: number): Promise<Model> => {
  const response = await api.get(`/models/${id}`)
  return response.data
}

/**
 * 创建模型记录
 */
export const createModel = async (data: ModelCreate) => {
  const response = await api.post('/models/', data)
  return response.data
}

/**
 * 上传模型文件
 */
export const uploadModelFile = async (
  id: number,
  file: File,
  onProgress?: (progress: number) => void
) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post(`/models/${id}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onProgress(progress)
      }
    },
  })
  
  return response.data
}

/**
 * 下载模型文件
 */
export const downloadModel = async (id: number, filename?: string) => {
  const response = await api.get(`/models/${id}/download`, {
    responseType: 'blob',
  })
  
  // 创建下载链接
  const url = window.URL.createObjectURL(new Blob([response.data]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', filename || `model_${id}.pth`)
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}

/**
 * 更新模型信息
 */
export const updateModel = async (id: number, data: ModelUpdate) => {
  const response = await api.put(`/models/${id}`, data)
  return response.data
}

/**
 * 删除模型
 */
export const deleteModel = async (id: number) => {
  const response = await api.delete(`/models/${id}`)
  return response.data
}

/**
 * 格式化文件大小
 */
export const formatFileSize = (bytes?: number): string => {
  if (!bytes) return 'N/A'
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = bytes
  let unitIndex = 0
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }
  
  return `${size.toFixed(2)} ${units[unitIndex]}`
}

/**
 * 格式化参数数量
 */
export const formatParams = (params?: number): string => {
  if (!params) return 'N/A'
  
  if (params >= 1_000_000) {
    return `${(params / 1_000_000).toFixed(1)}M`
  } else if (params >= 1_000) {
    return `${(params / 1_000).toFixed(1)}K`
  }
  
  return params.toString()
}

/**
 * 格式化准确率
 */
export const formatAccuracy = (accuracy?: number): string => {
  if (accuracy === undefined || accuracy === null) return 'N/A'
  return `${(accuracy * 100).toFixed(2)}%`
}

