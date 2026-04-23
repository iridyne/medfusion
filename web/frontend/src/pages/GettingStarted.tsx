import { useNavigate } from "react-router-dom";
import { Button, Card, Space } from "antd";
import {
  ApartmentOutlined,
  ArrowRightOutlined,
  ControlOutlined,
  DatabaseOutlined,
  FileSearchOutlined,
  LinkOutlined,
  PlayCircleOutlined,
  RadarChartOutlined,
} from "@ant-design/icons";

import PageScaffold from "@/components/layout/PageScaffold";
import { QUICKSTART_TRAINING_PREFILL } from "@/config/quickstartRun";
import {
  START_COMFYUI_DEFAULT_ADAPTER,
} from "@/config/startExperience";
import { buildTrainingPrefillQuery } from "@/utils/trainingPrefill";

const PRIMARY_LAUNCH_STEPS = [
  {
    id: "datasets",
    title: "数据检查",
    description: "登记本机数据目录并确认可用性。",
    route: "/datasets",
    buttonLabel: "打开数据检查",
    icon: <DatabaseOutlined />,
  },
  {
    id: "preprocessing",
    title: "预处理预检",
    description: "先做训练前检查，再进入配置。",
    route: "/preprocessing",
    buttonLabel: "进入预处理",
    icon: <ArrowRightOutlined />,
  },
  {
    id: "config",
    title: "问题向导",
    description: "生成可运行配置并导向训练。",
    route: "/config",
    buttonLabel: "开始配置",
    icon: <ControlOutlined />,
  },
  {
    id: "training",
    title: "训练执行",
    description: "启动任务并查看进度。",
    route: "/training",
    buttonLabel: "进入训练",
    icon: <PlayCircleOutlined />,
  },
  {
    id: "models",
    title: "结果后台",
    description: "训练完成后查看结果产物。",
    route: "/models",
    buttonLabel: "查看结果",
    icon: <FileSearchOutlined />,
  },
  {
    id: "evaluation",
    title: "独立评估",
    description: "对已有 checkpoint 单独评估。",
    route: "/evaluation",
    buttonLabel: "打开评估",
    icon: <RadarChartOutlined />,
  },
] as const;

const MODEL_ENTRY_POINTS = [
  {
    title: "官方模型库",
    description: "从官方模板起步。",
    route: "/config/model",
    icon: <RadarChartOutlined />,
    buttonLabel: "进入官方模型库",
  },
  {
    title: "本机自定义模型",
    description: "基于官方单元组合本地模板。",
    route: "/config/model/custom",
    icon: <ApartmentOutlined />,
    buttonLabel: "进入自定义模型",
  },
  {
    title: "ComfyUI 适配入口",
    description: "需要适配桥接时进入。",
    route: "/config/comfyui",
    icon: <LinkOutlined />,
    buttonLabel: "打开 ComfyUI",
  },
] as const;

export default function GettingStarted() {
  const navigate = useNavigate();
  const trainingPrefillQuery = buildTrainingPrefillQuery(QUICKSTART_TRAINING_PREFILL);

  return (
    <PageScaffold
      eyebrow="Mainline launchpad"
      title="开始一次研究运行"
      description="先走主线，再按需进入模型模块和评估。"
      actions={
        <>
          <Button type="primary" icon={<DatabaseOutlined />} onClick={() => navigate("/datasets")}>
            从数据检查开始
          </Button>
          <Button icon={<ControlOutlined />} onClick={() => navigate("/config")}>
            直接进入问题向导
          </Button>
        </>
      }
    >
      <div className="surface-callout" style={{ marginBottom: 16 }}>
        <div className="surface-callout__copy">
          <strong>默认主线：数据检查 → 预处理预检 → 配置 → 训练</strong>
          <span>流程跑通后再进入结果和评估。</span>
        </div>
        <Button type="primary" icon={<ArrowRightOutlined />} onClick={() => navigate("/datasets")}>
          开始主线
        </Button>
      </div>

      <div className="start-surface__grid">
        <Card className="surface-card surface-card--accent">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Primary lane</div>
              <h2 className="section-heading__title">推荐主线</h2>
              <p className="section-heading__description">按顺序执行即可。</p>
            </div>
          </div>

          <div className="start-lane">
            {PRIMARY_LAUNCH_STEPS.map((step, index) => (
              <div key={step.id} className="start-lane__step">
                <div className="start-lane__step-index">0{index + 1}</div>
                <div className="start-lane__step-body">
                  <div className="start-lane__step-title">
                    <span className="start-lane__step-icon">{step.icon}</span>
                    <strong>{step.title}</strong>
                  </div>
                  <p>{step.description}</p>
                  <Button
                    type={index === 0 ? "primary" : "default"}
                    icon={<ArrowRightOutlined />}
                    onClick={() => navigate(step.route)}
                  >
                    {step.buttonLabel}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card className="surface-card">
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Model building</div>
                <h2 className="section-heading__title">模型搭建入口</h2>
                <p className="section-heading__description">按需进入对应入口。</p>
              </div>
            </div>

            <div className="start-utility-list">
              {MODEL_ENTRY_POINTS.map((item) => (
                <div key={item.route} className="surface-note surface-note--dense">
                  <Space align="start" size={12}>
                    <span className="start-utility-list__icon">{item.icon}</span>
                    <div className="start-utility-list__body">
                      <strong>{item.title}</strong>
                      <p>{item.description}</p>
                      <Button size="small" onClick={() => navigate(item.route)}>
                        {item.buttonLabel}
                      </Button>
                    </div>
                  </Space>
                </div>
              ))}
            </div>
          </Card>
        </Space>
      </div>

      <Card className="surface-card" style={{ marginTop: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Quick actions</div>
            <h2 className="section-heading__title">常用跳转</h2>
            <p className="section-heading__description">
              这里保留少量高频动作，避免用户在首页来回找入口。
            </p>
          </div>
        </div>

        <div className="cta-grid">
          <div className="cta-card">
            <div className="cta-card__meta">
              <strong>带推荐参数进入训练</strong>
              <p>快速验证当前运行链路。</p>
            </div>
            <Button
              icon={<PlayCircleOutlined />}
              onClick={() => navigate(`/training?source=guided-start&${trainingPrefillQuery}`)}
            >
              启动推荐训练
            </Button>
          </div>

          <div className="cta-card">
            <div className="cta-card__meta">
              <strong>直接查看结果后台</strong>
              <p>训练后直接查看结果与产物。</p>
            </div>
            <Button icon={<FileSearchOutlined />} onClick={() => navigate("/models")}>
              打开结果后台
            </Button>
          </div>

          <div className="cta-card">
            <div className="cta-card__meta">
              <strong>ComfyUI 辅助入口</strong>
              <p>{START_COMFYUI_DEFAULT_ADAPTER[0] || "仅在需要桥接时使用。"}</p>
            </div>
            <Button icon={<LinkOutlined />} onClick={() => navigate("/config/comfyui")}>
              打开 ComfyUI
            </Button>
          </div>
        </div>
      </Card>
    </PageScaffold>
  );
}
