import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Card, Progress, Tag } from "antd";
import {
  ArrowRightOutlined,
  ControlOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  ImportOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";

import { getDatasetStatistics } from "@/api/datasets";
import { getModels, getModelStatistics } from "@/api/models";
import trainingApi from "@/api/training";
import PageScaffold from "@/components/layout/PageScaffold";
import { PRIMARY_ENTRY_COMMAND } from "@/config/navigation";

interface WorkbenchStats {
  totalDatasets: number;
  totalSamples: number;
  totalModels: number;
  avgAccuracy: number;
  runningJobs: number;
  totalJobs: number;
}

const EMPTY_STATS: WorkbenchStats = {
  totalDatasets: 0,
  totalSamples: 0,
  totalModels: 0,
  avgAccuracy: 0,
  runningJobs: 0,
  totalJobs: 0,
};

export default function Workbench() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<WorkbenchStats>(EMPTY_STATS);
  const [latestModelName, setLatestModelName] = useState<string>("-");

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [datasetStats, modelStats, models, jobs] = await Promise.all([
          getDatasetStatistics(),
          getModelStatistics(),
          getModels({ limit: 1 }),
          trainingApi.listJobs(),
        ]);

        setStats({
          totalDatasets: datasetStats.total_datasets ?? 0,
          totalSamples: datasetStats.total_samples ?? 0,
          totalModels: modelStats.total_models ?? 0,
          avgAccuracy: modelStats.avg_accuracy ?? 0,
          runningJobs: jobs.filter((job) => job.status === "running").length,
          totalJobs: jobs.length,
        });
        setLatestModelName(models?.[0]?.name || "-");
      } catch (error) {
        console.error("Failed to load workbench overview:", error);
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, []);

  const runningRatio = useMemo(() => {
    if (stats.totalJobs <= 0) {
      return 0;
    }
    return Math.round((stats.runningJobs / stats.totalJobs) * 100);
  }, [stats.runningJobs, stats.totalJobs]);

  const launchCards = [
    {
      title: "演示型训练",
      description:
        "快速生成训练曲线、ROC 和 attention heatmap，用于对外演示和 OSS 快速上手。",
      tag: "Demo lane",
      actionLabel: "进入演示训练",
      icon: <ExperimentOutlined />,
      onClick: () => navigate("/training?action=start"),
    },
    {
      title: "RunSpec 向导",
      description:
        "用真实 schema 生成训练配置，把 YAML 从入口位移到 artifact 输出与复现位。",
      tag: "RunSpec",
      actionLabel: "打开训练向导",
      icon: <ControlOutlined />,
      onClick: () => navigate("/config"),
    },
    {
      title: "结果导入",
      description:
        "把 CLI 训练出来的 checkpoint、ROC、日志和结果文件并回到结果库进行展示。",
      tag: "Artifact import",
      actionLabel: "导入真实结果",
      icon: <ImportOutlined />,
      onClick: () => navigate("/models?action=import"),
    },
    {
      title: "数据准备",
      description:
        "登记本地目录、观察处理状态，为真实训练和复现实验准备统一入口。",
      tag: "Data intake",
      actionLabel: "去数据管理",
      icon: <DatabaseOutlined />,
      onClick: () => navigate("/datasets"),
    },
  ];

  return (
    <PageScaffold
      eyebrow="Open-Core Research Surface"
      title="把 MedFusion OSS 收拢成一张研究控制面"
      description="这不是一个演示味很重的 SaaS 仪表盘，而是一张面向研究人员和评估者的桌面级工作面。数据、训练、RunSpec 与结果检查在这里汇合，执行和自动化仍然回到 CLI 主链。"
      chips={[
        { label: "Desktop-first cockpit", tone: "teal" },
        { label: "CLI-backed execution", tone: "amber" },
        { label: "Research-grade artifact trail", tone: "blue" },
      ]}
      actions={
        <>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() => navigate("/training?action=start")}
          >
            启动演示训练
          </Button>
          <Button
            size="large"
            icon={<ControlOutlined />}
            onClick={() => navigate("/config")}
          >
            打开 RunSpec 向导
          </Button>
          <Button
            size="large"
            icon={<ImportOutlined />}
            onClick={() => navigate("/models?action=import")}
          >
            导入真实结果
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Recommended entry</span>
          <div className="hero-aside-panel__value">从统一入口启动桌面工作台</div>
          <div className="hero-aside-panel__copy">
            Web 负责浏览、配置和检查，CLI 继续承担远程 GPU、自动化和可复现执行。
          </div>
          <pre className="command-block">{PRIMARY_ENTRY_COMMAND}</pre>
          <div className="surface-note">
            最近结果已索引到 <strong>{latestModelName}</strong>
          </div>
          <Progress
            percent={runningRatio}
            status={stats.runningJobs > 0 ? "active" : "normal"}
            format={() => `${stats.runningJobs}/${stats.totalJobs || 0} 运行中`}
          />
        </div>
      }
      metrics={[
        {
          label: "Datasets indexed",
          value: stats.totalDatasets.toLocaleString(),
          hint: `${stats.totalSamples.toLocaleString()} samples`,
          tone: "blue",
        },
        {
          label: "Result records",
          value: stats.totalModels.toLocaleString(),
          hint: loading ? "同步中…" : `Latest: ${latestModelName}`,
          tone: "teal",
        },
        {
          label: "Accuracy envelope",
          value: `${(stats.avgAccuracy * 100).toFixed(2)}%`,
          hint: "Across imported model artifacts",
          tone: "amber",
        },
        {
          label: "Active jobs",
          value: `${stats.runningJobs}/${stats.totalJobs || 0}`,
          hint: "Running / total jobs",
          tone: "rose",
        },
      ]}
    >
      <Card className="surface-card surface-card--accent">
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Launch lanes</div>
            <h2 className="section-heading__title">四条高频研究链路</h2>
            <p className="section-heading__description">
              从首次演示到真实训练导入，工作台把高频入口固定成可以复用的操作路径。
            </p>
          </div>
          <Tag color="processing">Workbench orchestration</Tag>
        </div>

        <div className="cta-grid">
          {launchCards.map((card) => (
            <div key={card.title} className="cta-card">
              <div className="cta-card__meta">
                <Tag>{card.tag}</Tag>
                <strong>{card.title}</strong>
                <p>{card.description}</p>
              </div>
              <Button icon={card.icon} onClick={card.onClick}>
                {card.actionLabel}
              </Button>
            </div>
          ))}
        </div>
      </Card>

      <div className="split-grid">
        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Workflow posture</div>
              <h2 className="section-heading__title">OSS 工作链路如何分工</h2>
              <p className="section-heading__description">
                不把 CLI 删除，而是把 Web 调整成发现和解释层，把 CLI 保留为执行与复现层。
              </p>
            </div>
          </div>

          <div className="workbench-flow">
            <div className="flow-step">
              <strong>1. 通过 Web 发现入口</strong>
              <p>浏览工作台、选择合适的数据、确认向导 preset 和结果导入路径。</p>
            </div>
            <div className="flow-step">
              <strong>2. 用 RunSpec 固定配置语义</strong>
              <p>表单产出真实配置，并把 CLI / Web 两条链拉回同一套 schema。</p>
            </div>
            <div className="flow-step">
              <strong>3. 结果回流到结果库</strong>
              <p>训练输出、ROC、混淆矩阵、attention 热力图和日志继续在 Web 做检查。</p>
            </div>
          </div>
        </Card>

        <Card className="surface-card surface-card--editorial">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Operator notes</div>
              <h2 className="section-heading__title">当前关注点</h2>
              <p className="section-heading__description">
                这张工作台优先服务评估者、贡献者和研究协作场景，而不是把所有事情塞进单一演示页。
              </p>
            </div>
          </div>

          <div className="editorial-stack">
            <div className="editorial-quote">
              <span className="editorial-quote__mark">/</span>
              <p>
                Web 端现在更像研究桌面的“解释层”，不是替代 CLI，而是把入口、状态和结果做成一套能被理解、能被演示、也能被维护的界面语言。
              </p>
            </div>

            <div className="editorial-grid">
              <div className="surface-note surface-note--dense">
                <strong>入口统一</strong>
                <p>对外推荐 `medfusion start`，对内保留 CLI 执行层。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>配置统一</strong>
                <p>向导与 CLI 共用 RunSpec，避免再维护一套演示型伪配置。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>结果统一</strong>
                <p>真实 artifact 导回结果库，支撑 README 截图、演示和研究复盘。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>信息统一</strong>
                <p>导航、主题、状态和命令提示集中到同一层壳，降低入口切换成本。</p>
              </div>
            </div>

            <Button icon={<ArrowRightOutlined />} onClick={() => navigate("/models")}>
              打开完整结果页
            </Button>
          </div>
        </Card>
      </div>

      <Card className="surface-card surface-card--accent">
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Control deck</div>
            <h2 className="section-heading__title">把 OSS 首屏改成真正的研究控制台</h2>
            <p className="section-heading__description">
              我把最容易被看到的首屏改成三层结构：一层讲入口，一层讲链路，一层讲资产。这样别人第一次打开时，就不会再把它当成普通后台模板。
            </p>
          </div>
          <Tag color="gold">Frontend redesign</Tag>
        </div>

        <div className="insight-grid">
          <div className="insight-panel">
            <span className="insight-panel__kicker">Shell rewrite</span>
            <strong>导航改成 lane-based 研究视图</strong>
            <p>每个工作面都绑定自己的 eyebrow、说明和强调色，壳层不再只是一个空头部。</p>
          </div>
          <div className="insight-panel">
            <span className="insight-panel__kicker">Visual system</span>
            <strong>用暖临床色系替代通用 AI 看板配色</strong>
            <p>保留层次感和微光，但避开俗套的青紫霓虹，看起来更像专业工作站。</p>
          </div>
          <div className="insight-panel">
            <span className="insight-panel__kicker">Information scent</span>
            <strong>命令、状态、资产、工作流同屏出现</strong>
            <p>这样无论是研究者、贡献者还是 OSS 评估者，都能一眼理解这个界面是干嘛的。</p>
          </div>
        </div>
      </Card>
    </PageScaffold>
  );
}
