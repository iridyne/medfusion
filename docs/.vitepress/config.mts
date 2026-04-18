import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'MedFusion',
  description: '高度模块化的医学多模态深度学习研究框架',
  lang: 'zh-CN',

  // 基础路径，部署到 GitHub Pages 时使用
  base: '/medfusion/',

  // 主题配置
  themeConfig: {
    logo: '/logo.svg',

    // 侧边栏
    sidebar: {
      '/contents/getting-started/': [
        {
          text: '快速入门',
          items: [
            { text: 'CLI 与 Config 使用路径', link: '/contents/getting-started/cli-config-workflow' },
            { text: '如何新建模型与 YAML', link: '/contents/getting-started/model-creation-paths' },
            { text: '环境安装', link: '/contents/getting-started/installation' },
            { text: '快速开始', link: '/contents/getting-started/quickstart' },
            { text: '公开数据集快速验证', link: '/contents/getting-started/public-datasets' },
            { text: '第一个模型', link: '/contents/getting-started/first-model' },
            { text: 'Web UI 快速入门', link: '/contents/getting-started/web-ui' }
          ]
        }
      ],
      '/contents/playbooks/': [
        {
          text: '任务手册',
          items: [
            { text: '总览', link: '/contents/playbooks/README' },
            { text: '最小可复现实验', link: '/contents/playbooks/minimum-reproducible-run' },
            { text: '对外 Demo 路径', link: '/contents/playbooks/external-demo-path' },
            { text: '多 seed 稳定性汇报', link: '/contents/playbooks/multi-seed-stability-report' },
            { text: '结果解读与交付检查', link: '/contents/playbooks/result-interpretation-checklist' }
          ]
        }
      ],
      '/contents/tutorials/': [
        {
          text: '教程导航',
          items: [
            { text: '总览', link: '/contents/tutorials/README' }
          ]
        },
        {
          text: '基础教程',
          collapsed: false,
          items: [
            { text: '配置文件详解', link: '/contents/tutorials/fundamentals/configs' },
            { text: '数据准备', link: '/contents/tutorials/fundamentals/data-prep' },
            { text: '模型构建器', link: '/contents/tutorials/fundamentals/builder-api' },
            { text: '选择骨干网络', link: '/contents/tutorials/fundamentals/backbones' },
            { text: '融合策略', link: '/contents/tutorials/fundamentals/fusion' }
          ]
        },
        {
          text: '训练指南',
          collapsed: false,
          items: [
            { text: '训练工作流', link: '/contents/tutorials/training/workflow' },
            { text: '监控训练', link: '/contents/tutorials/training/monitoring' },
            { text: '超参数调优', link: '/contents/tutorials/training/tuning' }
          ]
        },
        {
          text: '高级特性',
          collapsed: false,
          items: [
            { text: '注意力监督', link: '/contents/tutorials/advanced/attention' },
            { text: '多视图支持', link: '/contents/tutorials/advanced/multiview' }
          ]
        },
        {
          text: '部署指南',
          collapsed: false,
          items: [
            { text: '模型导出', link: '/contents/tutorials/deployment/model-export' },
            { text: 'Docker 部署', link: '/contents/tutorials/deployment/docker' },
            { text: '生产环境清单', link: '/contents/tutorials/deployment/production' }
          ]
        },
        {
          text: '案例研究',
          collapsed: false,
          items: [
            { text: '肺结节检测', link: '/contents/tutorials/case_studies/01_lung_nodule_detection' },
            { text: '乳腺癌分类', link: '/contents/tutorials/case_studies/02_breast_cancer_classification' },
            { text: '生存预测', link: '/contents/tutorials/case_studies/03_survival_prediction' }
          ]
        }
      ],
      '/contents/api/': [
        {
          text: 'API 文档',
          items: [
            { text: 'med_core', link: '/contents/api/med_core' },
            { text: 'models', link: '/contents/api/models' },
            { text: 'backbones', link: '/contents/api/backbones' },
            { text: 'fusion', link: '/contents/api/fusion' },
            { text: 'heads', link: '/contents/api/heads' },
            { text: 'datasets', link: '/contents/api/datasets' },
            { text: 'trainers', link: '/contents/api/trainers' },
            { text: 'preprocessing', link: '/contents/api/preprocessing' },
            { text: 'aggregators', link: '/contents/api/aggregators' },
            { text: 'attention_supervision', link: '/contents/api/attention_supervision' },
            { text: 'evaluation', link: '/contents/api/evaluation' },
            { text: 'utils', link: '/contents/api/utils' },
            { text: 'Web API', link: '/contents/api/web_api' }
          ]
        }
      ],
      '/contents/guides/': [
        {
          text: '核心指南',
          items: [
            { text: '快速参考', link: '/contents/guides/core/quick-reference' },
            { text: 'FAQ 和故障排除', link: '/contents/guides/core/faq' }
          ]
        },
        {
          text: '高级功能',
          collapsed: false,
          items: [
            { text: '分布式训练', link: '/contents/guides/advanced-features/distributed-training' },
            { text: '梯度检查点', link: '/contents/guides/advanced-features/gradient-checkpointing' },
            { text: '模型压缩', link: '/contents/guides/advanced-features/model-compression' },
            { text: '数据缓存', link: '/contents/guides/advanced-features/data-caching' },
            { text: '性能基准测试', link: '/contents/guides/advanced-features/performance-benchmarking' }
          ]
        },
        {
          text: '专题指南',
          collapsed: false,
          items: [
            { text: '注意力机制', link: '/contents/guides/attention/mechanism' },
            { text: '多视图支持', link: '/contents/guides/multiview/overview' }
          ]
        },
        {
          text: '开发指南',
          items: [
            { text: '文档规范', link: '/contents/guides/development/documentation-standards' },
            { text: '贡献指南', link: '/contents/guides/development/contributing' }
          ]
        }
      ],
      '/contents/architecture/': [
        {
          text: '架构文档',
          items: [
            { text: 'Core Runtime Architecture', link: '/contents/architecture/CORE_RUNTIME_ARCHITECTURE' },
            { text: 'Web UI 架构', link: '/contents/architecture/WEB_UI_ARCHITECTURE' },
            { text: '工作流设计（Legacy）', link: '/contents/architecture/WORKFLOW_DESIGN' }
          ]
        }
      ],
      '/contents/reference/': [
        {
          text: '参考文档',
          items: [
            { text: '错误代码', link: '/contents/reference/error_codes' }
          ]
        }
      ]
    },

    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '快速入门', link: '/contents/getting-started/cli-config-workflow' },
      { text: '任务手册', link: '/contents/playbooks/README' },
      { text: '架构', link: '/contents/architecture/CORE_RUNTIME_ARCHITECTURE' },
      { text: 'API 文档', link: '/contents/api/med_core' }
    ],

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/iridyne/medfusion' }
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
      pattern: 'https://github.com/iridyne/medfusion/edit/main/docs/:path',
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
