/**
 * API 客户端，支持请求重试和错误处理
 */
import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from "axios";

const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL || "http://localhost:8000";

/**
 * 创建 Axios 实例
 */
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * 请求拦截器
 */
apiClient.interceptors.request.use(
  (config) => {
    // 添加认证 token
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

/**
 * 响应拦截器
 */
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error: AxiosError) => {
    const config = error.config as AxiosRequestConfig & { _retry?: number };

    // 如果是 401 错误，清除 token 并跳转到登录页
    if (error.response?.status === 401) {
      localStorage.removeItem("access_token");
      window.location.href = "/login";
      return Promise.reject(error);
    }

    // 如果是网络错误或 5xx 错误，进行重试
    if (
      !error.response ||
      (error.response.status >= 500 && error.response.status < 600) ||
      error.code === "ECONNABORTED" ||
      error.code === "ERR_NETWORK"
    ) {
      config._retry = config._retry || 0;

      // 最多重试 3 次
      if (config._retry < 3) {
        config._retry++;

        // 指数退避：1s, 2s, 4s
        const delay = Math.pow(2, config._retry - 1) * 1000;
        console.log(
          `Retrying request (attempt ${config._retry}/3) after ${delay}ms`,
        );

        await new Promise((resolve) => setTimeout(resolve, delay));
        return apiClient(config);
      }
    }

    // 如果是 429 (Too Many Requests)，等待后重试
    if (error.response?.status === 429) {
      const retryAfter = error.response.headers["retry-after"];
      const delay = retryAfter ? parseInt(retryAfter) * 1000 : 5000;

      console.log(`Rate limited. Retrying after ${delay}ms`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return apiClient(config);
    }

    return Promise.reject(error);
  },
);

/**
 * 通用错误处理
 */
export const handleApiError = (error: any): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail: string }>;

    if (axiosError.response) {
      // 服务器返回错误
      return (
        axiosError.response.data?.detail ||
        `请求失败: ${axiosError.response.status}`
      );
    } else if (axiosError.request) {
      // 请求已发送但没有收到响应
      return "网络错误，请检查您的网络连接";
    } else {
      // 请求配置错误
      return "请求配置错误";
    }
  }

  return error?.message || "未知错误";
};

/**
 * 设置认证 token
 */
export const setAuthToken = (token: string) => {
  localStorage.setItem("access_token", token);
};

/**
 * 清除认证 token
 */
export const clearAuthToken = () => {
  localStorage.removeItem("access_token");
};

/**
 * 获取认证 token
 */
export const getAuthToken = (): string | null => {
  return localStorage.getItem("access_token");
};

export default apiClient;
