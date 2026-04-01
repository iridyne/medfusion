import { useNavigate } from "react-router-dom";
import { Alert, Button, Card, Tag } from "antd";
import {
  ArrowLeftOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";

import PageScaffold from "@/components/layout/PageScaffold";
import {
  QUICKSTART_RESULT_PATHS,
  QUICKSTART_STAGES,
  QUICKSTART_TRAINING_PREFILL,
  RECOMMENDED_QUICKSTART_PROFILE,
} from "@/config/quickstartRun";
import { buildTrainingPrefillQuery } from "@/utils/trainingPrefill";

const TRAINING_QUERY = buildTrainingPrefillQuery(QUICKSTART_TRAINING_PREFILL);

export default function QuickstartRun() {
  const navigate = useNavigate();

  return (
    <PageScaffold
      eyebrow="Quickstart run"
      title="第一次运行链路"
      description="这页把推荐 quickstart 拆成四个最小阶段：准备公开数据、校验配置、启动训练、构建结果。Web 负责解释路径，真实执行始终回到同一条 YAML + CLI 主链。"
      chips={[
        { label: "Recommended first run", tone: "amber" },
        { label: "YAML + CLI", tone: "blue" },
        { label: "Artifact contract", tone: "teal" },
      ]}
      actions={
        <>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() =>
              navigate(`/training?source=guided-start&${TRAINING_QUERY}`)
            }
          >
            去训练监控继续
          </Button>
          <Button
            size="large"
            icon={<FileSearchOutlined />}
            onClick={() => navigate("/models")}
          >
            查看结果入口
          </Button>
          <Button
            size="large"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/start")}
          >
            返回 Getting Started
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Recommended profile</span>
          <div className="hero-aside-panel__value">
            {RECOMMENDED_QUICKSTART_PROFILE.title}
          </div>
          <div className="hero-aside-panel__copy">
            {RECOMMENDED_QUICKSTART_PROFILE.description}
          </div>
          <pre className="command-block">
            {RECOMMENDED_QUICKSTART_PROFILE.configPath}
          </pre>
          <div className="surface-note">
            这条链的目标是先交付一组稳定产物，再进入更复杂的 YAML 组装。
          </div>
        </div>
      }
      metrics={[
        {
          label: "Recommended dataset",
          value: RECOMMENDED_QUICKSTART_PROFILE.id,
          hint: "Public dataset profile for first-run validation",
          tone: "amber",
        },
        {
          label: "Core stages",
          value: QUICKSTART_STAGES.length,
          hint: "prepare -> validate-config -> train -> build-results",
          tone: "blue",
        },
        {
          label: "Expected outputs",
          value: QUICKSTART_RESULT_PATHS.length,
          hint: "checkpoint + metrics + reports",
          tone: "teal",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="第一次成功标准"
        description="只要这四步跑通，并且你知道 checkpoint、metrics.json、validation.json、summary.json 和 report.md 在哪里，就算完成了第一轮 MVP onboarding。"
      />

      <Card className="surface-card surface-card--accent" style={{ marginBottom: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Run chain</div>
            <h2 className="section-heading__title">推荐执行顺序</h2>
            <p className="section-heading__description">
              每一步都对应真实命令；你之后从源码或 CLI 运行时，仍然复用这同一组入口。
            </p>
          </div>
          <Tag color="processing">{QUICKSTART_STAGES.length} steps</Tag>
        </div>

        <div className="workbench-flow">
          {QUICKSTART_STAGES.map((stage, index) => (
            <div key={stage.key} className="flow-step">
              <strong>
                {index + 1}. {stage.title}
              </strong>
              <p>{stage.description}</p>
              <pre className="command-block">{stage.command}</pre>
            </div>
          ))}
        </div>
      </Card>

      <div className="split-grid">
        <Card className="surface-card surface-card--editorial">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Training handoff</div>
              <h2 className="section-heading__title">进入训练页时会带上的默认参数</h2>
              <p className="section-heading__description">
                这里只是代填最常见字段，真正的运行 contract 仍然来自 YAML 配置和 CLI 命令。
              </p>
            </div>
          </div>

          <div className="editorial-stack">
            <div className="surface-note surface-note--dense">
              <strong>experimentName</strong>
              <p>{QUICKSTART_TRAINING_PREFILL.experimentName}</p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>backbone / epochs / batchSize</strong>
              <p>
                {QUICKSTART_TRAINING_PREFILL.backbone} /{" "}
                {QUICKSTART_TRAINING_PREFILL.epochs} /{" "}
                {QUICKSTART_TRAINING_PREFILL.batchSize}
              </p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>numClasses / learningRate</strong>
              <p>
                {QUICKSTART_TRAINING_PREFILL.numClasses} /{" "}
                {QUICKSTART_TRAINING_PREFILL.learningRate}
              </p>
            </div>
            <Button
              icon={<ExperimentOutlined />}
              onClick={() =>
                navigate(`/training?source=guided-start&${TRAINING_QUERY}`)
              }
            >
              带默认参数继续
            </Button>
          </div>
        </Card>

        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Result contract</div>
              <h2 className="section-heading__title">跑完后应该看到这些文件</h2>
              <p className="section-heading__description">
                如果这些路径里的文件完整落盘，说明这次 quickstart 已经交付了可读的最小结果集。
              </p>
            </div>
            <Tag color="success">Expected artifacts</Tag>
          </div>

          <div className="editorial-stack">
            {QUICKSTART_RESULT_PATHS.map((resultPath) => (
              <pre key={resultPath} className="command-block">
                {resultPath}
              </pre>
            ))}
          </div>
        </Card>
      </div>
    </PageScaffold>
  );
}
