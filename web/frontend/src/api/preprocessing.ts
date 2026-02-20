/**
 * 预处理 API 客户端
 */

import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1';

// ==================== 类型定义 ====================

export interface PreprocessingConfig {
  size: number;
  normalize: 'minmax' | 'zscore' | 'percentile' | 'none';
  remove_artifacts: boolean;
  enhance_contrast: boolean;
}

export interface PreprocessingTaskCreate {
  name: string;
  description?: string;
  input_dir: string;
  output_dir: string;
  config: PreprocessingConfig;
}

export interface PreprocessingTask {
  id: number;
  task_id: string;
  name: string;
  description?: string;
  input_dir: string;
  output_dir: string;
  config: PreprocessingConfig;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  total_images: number;
  processed_images: number;
  failed_images: number;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  duration?: number;
}

export interface PreprocessingTaskListResponse {
  tasks: PreprocessingTask[];
  total: number;
  skip: number;
  limit: number;
}

export interface PreprocessingStatistics {
  total_tasks: number;
  status_counts: {
    pending: number;
    running: number;
    completed: number;
    failed: number;
    cancelled: number;
  };
  total_processed_images: number;
  total_failed_images: number;
}

export interface PreprocessingResult {
  status: string;
  total_images: number;
  processed_images: number;
  failed_images: number;
  processed_files: string[];
  failed_files: Array<{ file: string; error: string }>;
  duration?: number;
  output_dir?: string;
}

// ==================== API 客户端 ====================

export const preprocessingAPI = {
  /**
   * 启动预处理任务
   */
  start: (data: PreprocessingTaskCreate) =>
    axios.post<PreprocessingTask>(`${API_BASE}/preprocessing/start`, data),

  /**
   * 获取任务状态
   */
  getStatus: (taskId: string) =>
    axios.get<PreprocessingTask>(`${API_BASE}/preprocessing/status/${taskId}`),

  /**
   * 列出预处理任务
   */
  list: (params?: {
    skip?: number;
    limit?: number;
    status?: string;
    sort_by?: string;
    order?: 'asc' | 'desc';
  }) =>
    axios.get<PreprocessingTaskListResponse>(`${API_BASE}/preprocessing/list`, {
      params,
    }),

  /**
   * 取消任务
   */
  cancel: (taskId: string) =>
    axios.post(`${API_BASE}/preprocessing/cancel/${taskId}`),

  /**
   * 删除任务
   */
  delete: (taskId: number) =>
    axios.delete(`${API_BASE}/preprocessing/${taskId}`),

  /**
   * 获取统计信息
   */
  statistics: () =>
    axios.get<PreprocessingStatistics>(`${API_BASE}/preprocessing/statistics`),

  /**
   * 创建 WebSocket 连接监控任务
   */
  connectWebSocket: (
    taskId: string,
    onMessage: (data: any) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
  ): WebSocket => {
    const wsUrl = `ws://localhost:8000/api/v1/preprocessing/ws/${taskId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      if (onClose) onClose();
    };

    return ws;
  },
};

// ==================== 工具函数 ====================

/**
 * 格式化任务状态
 */
export const formatStatus = (status: string): string => {
  const statusMap: Record<string, string> = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  };
  return statusMap[status] || status;
};

/**
 * 获取状态颜色
 */
export const getStatusColor = (status: string): string => {
  const colorMap: Record<string, string> = {
    pending: 'default',
    running: 'processing',
    completed: 'success',
    failed: 'error',
    cancelled: 'warning',
  };
  return colorMap[status] || 'default';
};

/**
 * 格式化归一化方法
 */
export const formatNormalizeMethod = (method: string): string => {
  const methodMap: Record<string, string> = {
    minmax: 'Min-Max 归一化',
    zscore: 'Z-Score 标准化',
    percentile: '百分位归一化',
    none: '不归一化',
  };
  return methodMap[method] || method;
};

/**
 * 格式化持续时间
 */
export const formatDuration = (seconds?: number): string => {
  if (!seconds) return '-';

  if (seconds < 60) {
    return `${seconds.toFixed(1)}秒`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}分${secs}秒`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}小时${minutes}分`;
  }
};

/**
 * 计算成功率
 */
export const calculateSuccessRate = (task: PreprocessingTask): number => {
  if (task.total_images === 0) return 0;
  return (task.processed_images / task.total_images) * 100;
};

/**
 * 验证输入目录
 */
export const validateInputDir = (dir: string): string | null => {
  if (!dir || dir.trim() === '') {
    return '请输入输入目录路径';
  }
  return null;
};

/**
 * 验证输出目录
 */
export const validateOutputDir = (dir: string): string | null => {
  if (!dir || dir.trim() === '') {
    return '请输出输出目录路径';
  }
  return null;
};

/**
 * 验证任务名称
 */
export const validateTaskName = (name: string): string | null => {
  if (!name || name.trim() === '') {
    return '请输入任务名称';
  }
  if (name.length > 255) {
    return '任务名称不能超过 255 个字符';
  }
  return null;
};
