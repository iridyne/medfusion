import { defineConfig } from 'vitepress'
import AutoSidebar from 'vite-plugin-vitepress-auto-sidebar'

export default defineConfig({
  title: 'MedFusion',
  description: '高度模块化的医学多模态深度学习研究框架',
  lang: 'zh-CN',

  // 基础路径，部署到 GitHub Pages 时使用
  base: '/medfusion/',

  // Vite 配置
  vite: {
    plugins: [
      AutoSidebar({
        // 自动生成侧边栏
        path: '/contents',
        collapsed: false,
      })
    ]
  },

  // 主题配置
  themeConfig: {
    logo: '/logo.svg',

    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '教程', link: '/contents/tutorials/README' },
      { text: '快速开始', link: '/contents/user-guides/QUICKSTART_GUIDE' },
      { text: 'API 文档', link: '/contents/api/med_core' },
      { text: '指南', link: '/contents/guides/quick_reference' },
      { text: 'GitHub', link: 'https://github.com/iridite/medfusion' }
    ],

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/iridite/medfusion' }
    ],

    // 页脚
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-2026 MedFusion Team'
    },

    // 搜索
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换'
                }
              }
            }
          }
        }
      }
    },

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/iridite/medfusion/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    }
  },

  // Markdown 配置
  markdown: {
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  // 忽略死链接（文档迁移中）
  ignoreDeadLinks: true
})
