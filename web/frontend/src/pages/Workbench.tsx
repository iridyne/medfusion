import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Alert, Button, Card, Progress, Tag } from "antd";
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
import {
  WORKBENCH_MODE,
  WORKBENCH_PRIMARY_CARDS,
} from "@/config/workbenchExperience";
import { consumeWorkbenchFallback } from "@/utils/workbenchFallback";

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
  const [searchParams, setSearchParams] = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<WorkbenchStats>(EMPTY_STATS);
  const [latestModelName, setLatestModelName] = useState<string>("-");
  const [redirectNotice, setRedirectNotice] = useState<string | null>(null);

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

  useEffect(() => {
    const { source, nextSearchParams } = consumeWorkbenchFallback(searchParams);
    if (!source) {
      return;
    }

    setRedirectNotice(source);
    setSearchParams(nextSearchParams, { replace: true });
  }, [searchParams, setSearchParams]);

  const runningRatio = useMemo(() => {
    if (stats.totalJobs <= 0) {
      return 0;
    }
    return Math.round((stats.runningJobs / stats.totalJobs) * 100);
  }, [stats.runningJobs, stats.totalJobs]);

  const launchCards = [
    {
      title: "训练监控",
      description:
        "创建训练任务、跟踪实时状态，并把结果稳定回流到结果库。",
      tag: "Mainline",
      actionLabel: "进入训练监控",
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
      eyebrow="Post-run overview"
      title="查看最近一次运行、关键结果和下一步"
      description="Workbench 现在更像运行后的中等概览页：先看最近状态、结果和下一步，再决定是否继续训练、导入产物或切换到自己的 YAML。"
      chips={[
        { label: WORKBENCH_MODE, tone: "teal" },
        { label: WORKBENCH_PRIMARY_CARDS[0], tone: "amber" },
        { label: WORKBENCH_PRIMARY_CARDS[2], tone: "blue" },
      ]}
      actions={
        <>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() => navigate("/training?action=start")}
          >
            启动训练任务
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
          <div className="hero-aside-panel__value">首次进入先走 Getting Started</div>
          <div className="hero-aside-panel__copy">
            `medfusion start` 先把第一次成功讲清楚；回到这里时，重点已经变成最近运行、结果资产和下一步。
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
      {redirectNotice ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={`已回到工作台：${redirectNotice} 当前不作为 OSS 默认主链入口`}
          description="当前默认入口固定为 workbench / datasets / config / training / models / system。workflow 相关能力保留为实验态。"
          closable
          onClose={() => setRedirectNotice(null)}
        />
      ) : null}

      <Card className="surface-card surface-card--accent">
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Next actions</div>
            <h2 className="section-heading__title">从最近运行继续往前走</h2>
            <p className="section-heading__description">
              这里保留高频入口，但不再承担“第一次教你怎么开始”的职责。
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
                这张工作台优先服务评估者、贡献者和研究协作场景，避免脱离主链的页面分叉。
              </p>
            </div>
          </div>

          <div className="editorial-stack">
            <div className="editorial-quote">
              <span className="editorial-quote__mark">/</span>
              <p>
                进入 Workbench 说明第一次引导已经结束。这里更像运行后的回看台：你可以继续训练、读结果、整理资产，但主链执行语义仍然和 CLI 保持一致。
              </p>
            </div>

            <div className="editorial-grid">
              <div className="surface-note surface-note--dense">
                <strong>入口统一</strong>
                <p>第一次进入走 `medfusion start`，之后再回到这张总览页看最近运行。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>配置统一</strong>
                <p>向导与 CLI 共用 RunSpec，避免再维护偏展示用途的分叉配置。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>结果统一</strong>
                <p>真实 artifact 导回结果库，支撑验证、分享和研究复盘。</p>
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
            <div className="section-heading__eyebrow">Overview posture</div>
            <h2 className="section-heading__title">保持中等概览，而不是重型 dashboard</h2>
            <p className="section-heading__description">
              当前页只收口最近状态、关键指标和下一步入口，避免重新长成复杂产品壳。
            </p>
          </div>
          <Tag color="gold">Frontend redesign</Tag>
        </div>

        <div className="insight-grid">
          <div className="insight-panel">
            <span className="insight-panel__kicker">Shell rewrite</span>
              <strong>保持最近运行可见</strong>
              <p>先看最新状态和结果，再决定是继续 quickstart、导入结果，还是换自己的 YAML。</p>
            </div>
            <div className="insight-panel">
              <span className="insight-panel__kicker">Visual system</span>
              <strong>保留必要信息密度</strong>
              <p>关键指标、命令和结果入口同屏，但不把每个工作面都挤成重控制台。</p>
            </div>
            <div className="insight-panel">
              <span className="insight-panel__kicker">Information scent</span>
              <strong>下一步动作保持很少</strong>
              <p>把“去训练、看结果、导入产物”保留作高频入口，不继续扩展页面职责。</p>
            </div>
          </div>
        </Card>
    </PageScaffold>
  );
}
