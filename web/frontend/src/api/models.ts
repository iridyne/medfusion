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

export interface ModelImportRequest {
  config_path: string;
  checkpoint_path: string;
  output_dir?: string;
  split?: "train" | "val" | "test";
  attention_samples?: number;
  survival_time_column?: string;
  survival_event_column?: string;
  importance_sample_limit?: number;
  name?: string;
  description?: string;
  tags?: string[];
}

export interface ModelInspectResponse {
  status: "ready" | "warning" | "blocked";
  can_enter_training: boolean;
  issues: {
    errors: Array<{
      path: string;
      message: string;
      error_code: string;
      suggestion?: string | null;
    }>;
    warnings: Array<{
      path: string;
      message: string;
      error_code: string;
      suggestion?: string | null;
    }>;
  };
  checks: Array<{
    key: string;
    label: string;
    status: "pass" | "warning" | "fail";
    detail: string;
  }>;
  summary: {
    num_classes: number;
    backbone: string;
    attention_type: string;
    fusion_type: string;
    tabular_feature_count: number;
    recommended_fusion_hidden_dim: number;
  };
  runtime?: {
    vision_output_dim: number;
    tabular_input_dim: number;
    tabular_output_dim: number;
    fusion_output_dim: number;
    total_params: number;
    trainable_params: number;
    frozen_params: number;
    pretrained_requested: boolean;
    pretrained_materialized: boolean;
    auxiliary_heads: boolean;
  } | null;
  next_step: string;
}

export interface ModelCatalogComponent {
  id: string;
  source: "official" | "custom";
  label: string;
  family: string;
  status: string;
  description: string;
  data_requirements: string[];
  config_requirements: string[];
  compute_profile: {
    tier: string;
    gpu_vram_hint: string;
    notes: string;
  };
  upstream: string[];
  outputs: string[];
  advanced_builder_component_id?: string;
  advanced_builder_contract?: {
    preset_hints: string[];
    compile_boundary: string;
    compile_notes: string[];
    patch_target_hints: Array<{
      path: string;
      mode: string;
      description: string;
    }>;
    warning_metadata: Array<{
      code: string;
      path?: string;
      message: string;
      suggestion?: string;
    }>;
  };
  wizard_prefill?: Record<string, any>;
}

export interface ModelCatalogTemplate {
  id: string;
  source: "official" | "custom";
  label: string;
  status: string;
  description: string;
  component_ids: string[];
  unit_map?: Record<string, string>;
  editable_slots?: string[];
  data_requirements: string[];
  compute_profile: {
    tier: string;
    gpu_vram_hint: string;
    notes: string;
  };
  advanced_builder_blueprint_id?: string;
  advanced_builder_contract?: {
    recommended_preset: string;
    compile_boundary: string;
    compile_notes: string[];
    patch_target_hints: Array<{
      path: string;
      mode: string;
      description: string;
    }>;
  };
  wizard_prefill?: Record<string, any>;
}

export interface ModelCatalogResponse {
  sources: {
    official: {
      enabled: boolean;
      label: string;
      description: string;
      entry_path: string;
    };
    custom: {
      enabled: boolean;
      label: string;
      description: string;
      entry_path: string;
    };
  };
  principles: string[];
  advanced_builder: {
    family_projection: Record<string, { advanced_family: string; label: string }>;
    family_labels: Record<string, string>;
    status_labels: Record<string, string>;
    required_families: string[];
    default_preset: string;
    preset_rules: Array<{
      preset: string;
      priority: number;
      match_any_components: string[];
      description: string;
    }>;
    connection_rules: Array<{
      from_family: string;
      to_family: string;
      status: string;
      description: string;
    }>;
  };
  units: ModelCatalogComponent[];
  models: ModelCatalogTemplate[];
  components: ModelCatalogComponent[];
  templates: ModelCatalogTemplate[];
}

export interface CustomModelEntry {
  schema_version: string;
  id: string;
  source: "custom";
  label: string;
  description: string;
  status: "local_custom";
  based_on_model_id: string;
  unit_map: Record<string, string>;
  editable_slots: string[];
  component_ids: string[];
  data_requirements: string[];
  compute_profile: {
    tier: string;
    gpu_vram_hint: string;
    notes: string;
  };
  wizard_prefill: Record<string, any>;
  created_at: string;
  updated_at: string;
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
  config?: Record<string, any>;
  config_path?: string;
  model_path: string;
  checkpoint_path?: string;
  file_size?: number;
  format?: string;
  training_job_id?: number;
  trained_epochs?: number;
  training_time?: number;
  dataset_name?: string;
  tags?: string[];
  result_files?: Array<{
    key: string;
    label: string;
    path: string;
    exists: boolean;
    download_url?: string;
    preview_url?: string | null;
    is_image?: boolean;
  }>;
  training_history?: {
    entries: Array<{
      epoch: number;
      train_loss: number;
      val_loss: number;
      train_accuracy: number;
      val_accuracy: number;
      learning_rate: number;
      best_so_far?: boolean;
    }>;
    plot_artifact_key?: string | null;
    plot_url?: string | null;
  } | null;
  validation?: {
    dataset?: {
      name?: string | null;
      labels?: string[];
      num_classes?: number;
      sample_count?: number;
      class_distribution?: Array<{
        label: string;
        count: number;
        rate: number;
      }>;
    };
    overview?: {
      sample_count?: number;
      num_classes?: number;
      positive_class_label?: string;
      positive_prevalence?: number;
      accuracy?: number;
      balanced_accuracy?: number;
      precision_macro?: number;
      recall_macro?: number;
      macro_f1?: number;
      weighted_f1?: number;
      auc?: number | null;
      mean_confidence?: number;
      error_count?: number;
      error_rate?: number;
      best_epoch?: number | null;
    };
    per_class?: Array<{
      label: string;
      support: number;
      prevalence: number;
      precision: number;
      recall: number;
      f1_score: number;
      predicted_count: number;
      predicted_rate: number;
    }>;
    prediction_summary?: {
      mean_confidence?: number;
      mean_confidence_correct?: number | null;
      mean_confidence_error?: number | null;
      error_count?: number;
      error_rate?: number;
      top_misclassifications?: Array<{
        actual: string;
        predicted: string;
        count: number;
      }>;
    };
    threshold_analysis?: {
      threshold?: number;
      youden_j?: number;
      sensitivity?: number;
      specificity?: number;
      ppv?: number;
      npv?: number;
      confusion_matrix?: number[][];
    } | null;
    calibration?: {
      positive_class_label?: string;
      brier_score?: number;
      ece?: number;
      n_bins?: number;
      bins?: Array<{
        bin_index: number;
        range_start: number;
        range_end: number;
        count: number;
        mean_confidence: number;
        empirical_accuracy: number;
        gap: number;
      }>;
    } | null;
    survival?: {
      time_column?: string;
      event_column?: string;
      risk_score_source?: string;
      sample_count?: number;
      event_count?: number;
      event_rate?: number;
      censoring_rate?: number;
      median_survival_time?: number | null;
      risk_group_threshold?: number;
      c_index?: number | null;
      risk_distribution?: {
        min?: number;
        median?: number;
        max?: number;
        mean?: number;
        std?: number;
      };
      kaplan_meier?: {
        groups?: Array<{
          label: string;
          count: number;
          event_count: number;
          curve: Array<{
            time: number;
            survival: number;
            at_risk?: number;
            events?: number;
          }>;
        }>;
      };
    } | null;
    global_feature_importance?: {
      method?: string;
      score_name?: string;
      sample_count?: number;
      feature_count?: number;
      top_features?: Array<{
        feature: string;
        mean_abs_contribution: number;
        mean_contribution: number;
        std_contribution: number;
        sample_count: number;
      }>;
      features?: Array<{
        feature: string;
        mean_abs_contribution: number;
        mean_contribution: number;
        std_contribution: number;
        sample_count: number;
      }>;
    } | null;
  } | null;
  source_contract?: {
    source_type: string;
    entrypoint?: string;
    blueprint_id?: string;
    template_id?: string;
    template_label?: string;
    recommended_preset?: string;
    compile_boundary?: string;
    compile_notes?: string[];
    patch_target_hints?: Array<{
      path: string;
      mode: string;
      description: string;
    }>;
    split?: string;
    message?: string;
  } | null;
  visualizations?: {
    roc_curve?: {
      auc?: number | null;
      positive_class_label?: string;
      points: Array<{
        fpr: number;
        tpr: number;
        threshold?: number;
      }>;
      plot_artifact_key?: string | null;
      plot_url?: string | null;
    };
    confusion_matrix?: {
      labels: string[];
      matrix: number[][];
      plot_artifact_key?: string | null;
      plot_url?: string | null;
      normalized_plot_artifact_key?: string | null;
      normalized_plot_url?: string | null;
    };
    attention_maps?: Array<{
      title: string;
      modality: string;
      grid: number[][];
      artifact_key?: string;
      image_url?: string | null;
      mean_attention?: number;
      peak_attention?: number;
    }>;
    attention_statistics?: {
      artifact_key: string;
      image_url: string;
    };
    calibration_curve?: {
      artifact_key: string;
      image_url: string;
    };
    probability_distribution?: {
      artifact_key: string;
      image_url: string;
    };
    survival_curve?: {
      artifact_key: string;
      image_url: string;
      c_index?: number | null;
    };
    risk_score_distribution?: {
      artifact_key: string;
      image_url: string;
    };
    feature_importance_bar?: {
      artifact_key: string;
      image_url: string;
      top_features?: Array<{
        feature: string;
        mean_abs_contribution: number;
      }>;
    };
    feature_importance_beeswarm?: {
      artifact_key: string;
      image_url: string;
    };
    training_curves?: {
      artifact_key: string;
      image_url: string;
    };
    phase_importance?: {
      phase_labels?: string[];
      mean_importance?: Record<string, number>;
      cases?: Array<{
        case_id: string;
        phase_importance: Record<string, number>;
      }>;
      artifact_key?: string | null;
      artifact_url?: string | null;
    };
    case_explanations?: {
      phase_labels?: string[];
      cases?: Array<{
        case_id: string;
        predicted_label?: number;
        pred_probability?: number;
        risk_score?: number;
        phase_importance?: Record<string, number>;
        top_clinical_factors?: string[];
        heatmap_artifacts?: Array<Record<string, any>>;
      }>;
      artifact_key?: string | null;
      artifact_url?: string | null;
    };
    three_phase_heatmaps?: {
      method?: string;
      phase_labels?: string[];
      artifact_key?: string | null;
      artifact_url?: string | null;
      case_count?: number;
      heatmap_count?: number;
      cases?: Array<{
        case_id: string;
        predicted_label?: number;
        pred_probability?: number;
        heatmaps?: Array<{
          phase: string;
          method?: string;
          default_explanation_target?: string;
          image_path?: string;
          slice_index?: number;
          phase_importance?: number;
          targets?: Record<string, { target_class?: number; image_path?: string }>;
        }>;
      }>;
    };
  };
  created_at: string;
  updated_at?: string;
  created_by?: string;
}

export const inspectModelConfig = async (payload: {
  num_classes: number;
  use_auxiliary_heads: boolean;
  vision: {
    backbone: string;
    pretrained: boolean;
    freeze_backbone: boolean;
    feature_dim: number;
    dropout: number;
    attention_type: string;
  };
  tabular: {
    hidden_dims: number[];
    output_dim: number;
    dropout: number;
  };
  fusion: {
    fusion_type: string;
    hidden_dim: number;
    dropout: number;
    num_heads: number;
  };
  numerical_features: string[];
  categorical_features: string[];
  image_size: number;
  use_attention_supervision: boolean;
  num_epochs: number;
}): Promise<ModelInspectResponse> => {
  const response = await api.post("/models/inspect-config", payload)
  return response.data
}

export const getModelCatalog = async (): Promise<ModelCatalogResponse> => {
  const response = await api.get("/models/catalog")
  return response.data
}

export const getCustomModels = async (): Promise<{
  items: CustomModelEntry[];
  trash_items: CustomModelEntry[];
  storage: string;
  root_dir: string;
  schema_version: string;
  format_contract: string;
  history_backend: string;
  supports_export_import: boolean;
  retention_scope: string;
  retention_floor_scope: string;
  retention_policy: {
    mode: "count" | "time";
    max_count: number;
    max_age_days: number;
    min_count_per_model: number;
  };
}> => {
  const response = await api.get("/models/custom")
  return response.data
}

export const saveCustomModelEntry = async (
  entry: CustomModelEntry,
): Promise<{ item: CustomModelEntry; storage: string; schema_version: string; history_backend: string }> => {
  const response = await api.post("/models/custom", entry)
  return response.data
}

export const deleteCustomModelEntry = async (
  id: string,
): Promise<{ deleted: boolean; recycled: boolean; id: string; storage: string; history_backend: string }> => {
  const response = await api.delete(`/models/custom/${id}`)
  return response.data
}

export const undeleteCustomModelEntry = async (
  id: string,
): Promise<{ item: CustomModelEntry; history_backend: string; restore_behavior: string }> => {
  const response = await api.post(`/models/custom/${id}/undelete`)
  return response.data
}

export const getCustomModelHistory = async (
  id: string,
): Promise<{
  id: string;
  items: Array<{ commit: string; committed_at: string; subject: string }>;
  history_backend: string;
}> => {
  const response = await api.get(`/models/custom/${id}/history`)
  return response.data
}

export const restoreCustomModelEntry = async (
  id: string,
  revision: string,
): Promise<{ item: CustomModelEntry; history_backend: string; restore_behavior: string }> => {
  const response = await api.post(`/models/custom/${id}/restore`, { revision })
  return response.data
}

export const getCustomModelRetentionPolicy = async (): Promise<{
  policy: {
    mode: "count" | "time";
    max_count: number;
    max_age_days: number;
    min_count_per_model: number;
  };
  history_backend: string;
  supports_export_import: boolean;
  retention_scope: string;
  retention_floor_scope: string;
}> => {
  const response = await api.get("/models/custom/policy")
  return response.data
}

export const updateCustomModelRetentionPolicy = async (policy: {
  mode: "count" | "time";
  max_count: number;
  max_age_days: number;
  min_count_per_model: number;
}): Promise<{
  policy: {
    mode: "count" | "time";
    max_count: number;
    max_age_days: number;
    min_count_per_model: number;
  };
  history_backend: string;
  retention_scope: string;
  retention_floor_scope: string;
}> => {
  const response = await api.put("/models/custom/policy", policy)
  return response.data
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
 * 导入真实训练结果到模型库
 */
export const importModelRun = async (data: ModelImportRequest): Promise<Model> => {
  const response = await api.post('/models/import-run', data)
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
 * 下载模型结果产物
 */
export const downloadModelArtifact = async (
  id: number,
  artifactKey: string,
  filename?: string
) => {
  const response = await api.get(`/models/${id}/artifacts/${artifactKey}`, {
    responseType: 'blob',
  })

  const url = window.URL.createObjectURL(new Blob([response.data]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', filename || `${artifactKey}_${id}`)
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
