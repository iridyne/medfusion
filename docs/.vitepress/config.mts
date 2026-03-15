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
      '/contents/tutorials/': [
        {
          text: '教程导航',
          items: [
            { text: '总览', link: '/contents/tutorials/README' }
          ]
        },
        {
          text: '教程模块',
          collapsed: false,
          items: [
            { text: '环境安装', link: '/contents/tutorials/modules/01_installation' },
            { text: '你的第一个模型', link: '/contents/tutorials/modules/02_first_model' },
            { text: '配置文件详解', link: '/contents/tutorials/modules/03_understanding_configs' },
            { text: '数据准备指南', link: '/contents/tutorials/modules/04_data_preparation' },
            { text: '模型构建器 API', link: '/contents/tutorials/modules/05_builder_api' },
            { text: '选择骨干网络', link: '/contents/tutorials/modules/06_choosing_backbones' },
            { text: '融合策略对比', link: '/contents/tutorials/modules/07_fusion_strategies' },
            { text: '训练工作流', link: '/contents/tutorials/modules/08_training_workflow' },
            { text: '监控训练进度', link: '/contents/tutorials/modules/09_monitoring_progress' },
            { text: '超参数调优', link: '/contents/tutorials/modules/10_hyperparameter_tuning' },
            { text: '注意力监督', link: '/contents/tutorials/modules/11_attention_supervision' },
            { text: '多视图支持', link: '/contents/tutorials/modules/12_multiview_support' },
            { text: '模型导出', link: '/contents/tutorials/modules/13_model_export' },
            { text: 'Docker 部署', link: '/contents/tutorials/modules/14_docker_deployment' },
            { text: '生产环境清单', link: '/contents/tutorials/modules/15_production_checklist' }
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
      '/contents/user-guides/': [
        {
          text: '用户指南',
          items: [
            { text: '快速入门指南', link: '/contents/user-guides/QUICKSTART_GUIDE' },
            { text: 'Docker 部署指南', link: '/contents/user-guides/DOCKER_GUIDE' },
            { text: 'Web UI 快速入门', link: '/contents/user-guides/WEB_UI_QUICKSTART' }
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
            { text: 'utils', link: '/contents/api/utils' }
          ]
        }
      ],
      '/contents/guides/': [
        {
          text: '核心指南',
          items: [
            { text: '快速参考', link: '/contents/guides/quick_reference' },
            { text: 'FAQ 和故障排除', link: '/contents/guides/faq_troubleshooting' }
          ]
        },
        {
          text: '高级功能',
          collapsed: false,
          items: [
            { text: '分布式训练', link: '/contents/guides/distributed_training' },
            { text: '梯度检查点', link: '/contents/guides/gradient_checkpointing_guide' },
            { text: '模型压缩', link: '/contents/guides/model_compression' },
            { text: '模型导出', link: '/contents/guides/model_export' },
            { text: '数据缓存', link: '/contents/guides/data_caching' },
            { text: '性能基准测试', link: '/contents/guides/performance_benchmarking' },
            { text: 'CI/CD 流程', link: '/contents/guides/ci_cd' }
          ]
        },
        {
          text: '专题指南',
          collapsed: false,
          items: [
            { text: '注意力机制', link: '/contents/guides/attention/mechanism' },
            { text: '多视图支持', link: '/contents/guides/multiview/overview' }
          ]
        }
      ],
      '/contents/architecture/': [
        {
          text: '架构文档',
          items: [
            { text: 'Web UI 架构', link: '/contents/architecture/WEB_UI_ARCHITECTURE' },
            { text: '工作流设计', link: '/contents/architecture/WORKFLOW_DESIGN' }
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
      { text: '教程', link: '/contents/tutorials/README' },
      { text: '新手指南', link: '/contents/user-guides/QUICKSTART_GUIDE' },
      { text: 'API 文档', link: '/contents/api/med_core' },
      { text: '功能指南', link: '/contents/guides/quick_reference' }
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
