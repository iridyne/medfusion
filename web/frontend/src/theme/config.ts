import { ThemeConfig } from 'antd'

export type ThemeMode = 'light' | 'dark' | 'auto'

// 亮色主题配置
export const lightTheme: ThemeConfig = {
  token: {
    colorPrimary: '#1890ff',
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#ff4d4f',
    colorInfo: '#1890ff',
    colorBgBase: '#ffffff',
    colorTextBase: '#000000',
    borderRadius: 6,
    fontSize: 14,
  },
  components: {
    Layout: {
      colorBgHeader: '#001529',
      colorBgBody: '#f0f2f5',
      colorBgTrigger: '#002140',
    },
    Menu: {
      colorItemBg: '#001529',
      colorItemText: 'rgba(255, 255, 255, 0.65)',
      colorItemTextSelected: '#ffffff',
      colorItemBgSelected: '#1890ff',
      colorItemTextHover: '#ffffff',
    },
    Card: {
      colorBgContainer: '#ffffff',
      boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.03), 0 1px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
    },
  },
}

// 暗色主题配置
export const darkTheme: ThemeConfig = {
  token: {
    colorPrimary: '#177ddc',
    colorSuccess: '#49aa19',
    colorWarning: '#d89614',
    colorError: '#d32029',
    colorInfo: '#177ddc',
    colorBgBase: '#141414',
    colorTextBase: '#ffffff',
    borderRadius: 6,
    fontSize: 14,
  },
  components: {
    Layout: {
      colorBgHeader: '#1f1f1f',
      colorBgBody: '#000000',
      colorBgTrigger: '#262626',
    },
    Menu: {
      colorItemBg: '#1f1f1f',
      colorItemText: 'rgba(255, 255, 255, 0.65)',
      colorItemTextSelected: '#ffffff',
      colorItemBgSelected: '#177ddc',
      colorItemTextHover: '#ffffff',
    },
    Card: {
      colorBgContainer: '#1f1f1f',
      boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3), 0 1px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px 0 rgba(0, 0, 0, 0.2)',
    },
    Table: {
      colorBgContainer: '#1f1f1f',
    },
    Modal: {
      colorBgElevated: '#1f1f1f',
    },
    Drawer: {
      colorBgElevated: '#1f1f1f',
    },
  },
}

// 获取系统主题偏好
export const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'light'

  const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)')
  return darkModeQuery.matches ? 'dark' : 'light'
}

// 获取当前主题
export const getCurrentTheme = (mode: ThemeMode): ThemeConfig => {
  if (mode === 'auto') {
    return getSystemTheme() === 'dark' ? darkTheme : lightTheme
  }
  return mode === 'dark' ? darkTheme : lightTheme
}

// 保存主题设置
export const saveThemeMode = (mode: ThemeMode): void => {
  localStorage.setItem('themeMode', mode)
}

// 加载主题设置
export const loadThemeMode = (): ThemeMode => {
  const saved = localStorage.getItem('themeMode')
  if (saved === 'light' || saved === 'dark' || saved === 'auto') {
    return saved
  }
  return 'light'
}

// 监听系统主题变化
export const watchSystemTheme = (callback: (theme: 'light' | 'dark') => void): (() => void) => {
  if (typeof window === 'undefined') return () => {}

  const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)')

  const handler = (e: MediaQueryListEvent) => {
    callback(e.matches ? 'dark' : 'light')
  }

  darkModeQuery.addEventListener('change', handler)

  return () => {
    darkModeQuery.removeEventListener('change', handler)
  }
}
