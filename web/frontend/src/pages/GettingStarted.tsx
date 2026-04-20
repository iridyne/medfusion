import { useNavigate } from "react-router-dom";
import { Alert, Button, Card, Space, Tag } from "antd";
import {
  ArrowRightOutlined,
  ControlOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  LinkOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";

import PageScaffold from "@/components/layout/PageScaffold";
import { PRIMARY_ENTRY_COMMAND } from "@/config/navigation";
import {
  START_COMFYUI_OPTIONAL_MODULE,
  START_COMPONENTS,
  START_MODE_POSITIONING,
  START_PRIMARY_FLOW,
  START_RECOMMENDED_WORKFLOW,
} from "@/config/startExperience";

export default function GettingStarted() {
  const navigate = useNavigate();

  return (
    <PageScaffold
      eyebrow="Formal release entry"
      title="先理解正式版组件，再开始定义你的模型问题"
      description="正式版默认入口先介绍组件职责，再把你带到问题向导与模型搭建主链。GUI 是用户入口，runtime / CLI 仍然是执行真源；节点式编辑保留为高级模式，不直接替代默认首页。"
      chips={[
        { label: "GUI-first", tone: "amber" },
        { label: "Runtime-backed", tone: "blue" },
        { label: "Default before advanced", tone: "teal" },
      ]}
      actions={
        <>
          <Button
            type="primary"
            size="large"
            icon={<ControlOutlined />}
            onClick={() => navigate("/config")}
          >
            开始问题向导
          </Button>
          <Button
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() => navigate("/quickstart-run")}
          >
            查看第一次运行链路
          </Button>
          <Button
            size="large"
            icon={<ArrowRightOutlined />}
            onClick={() => navigate("/workbench")}
          >
            打开工作台总览
          </Button>
          <Button
            size="large"
            icon={<LinkOutlined />}
            onClick={() => navigate("/config/comfyui")}
          >
            打开 ComfyUI 入口
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Primary entry</span>
          <div className="hero-aside-panel__value">默认先走组件介绍与问题定义</div>
          <div className="hero-aside-panel__copy">
            目标不是先理解仓库结构，而是先知道有哪些组件、哪条路是主链、以及下一步该去哪里搭模型。
          </div>
          <pre className="command-block">{PRIMARY_ENTRY_COMMAND}</pre>
          <div className="surface-note">
            推荐下一步：
            <strong> {START_PRIMARY_FLOW.join(" -> ")}</strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Default mode",
          value: "question -> skeleton -> parameter edit",
          hint: START_MODE_POSITIONING.defaultMode,
          tone: "amber",
        },
        {
          label: "Execution source",
          value: "runtime / CLI",
          hint: "GUI explains and assembles, runtime executes and reproduces",
          tone: "blue",
        },
        {
          label: "Advanced mode",
          value: "node editing",
          hint: START_MODE_POSITIONING.advancedMode,
          tone: "blue",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="唯一主线：配置 -> 训练 -> 结果"
        description="正式版只有一条主线。当前在配置阶段默认使用 ComfyUI 适配配置，训练与结果仍由 MedFusion 主链执行。"
      />
      <Alert
        type="success"
        showIcon
        style={{ marginBottom: 16 }}
        message="ComfyUI 已有主页入口，不需要手动输入地址"
        description={
          <Space>
            <span>可以直接从首页进入 ComfyUI 集成页，并把回流导入参数一键带到结果后台。</span>
            <Button size="small" icon={<LinkOutlined />} onClick={() => navigate("/config/comfyui")}>
              进入 ComfyUI 集成
            </Button>
          </Space>
        }
      />

      <div className="split-grid">
        <Card className="surface-card surface-card--accent">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Product shell</div>
              <h2 className="section-heading__title">正式版组件与集成入口</h2>
              <p className="section-heading__description">
                入口页先把当前真正可用的组件讲清楚，避免用户误把实验页和历史页面当成默认主链。
              </p>
            </div>
            <Tag color="processing">{START_COMPONENTS.length} components</Tag>
          </div>

          <div className="workbench-flow">
            {START_COMPONENTS.map((item, index) => (
              <div key={item.key} className="flow-step">
                <strong>
                  {index + 1}. {item.title}
                </strong>
                <p>{item.description}</p>
                <Button
                  size="small"
                  icon={<ArrowRightOutlined />}
                  onClick={() => navigate(item.route)}
                >
                  打开组件
                </Button>
              </div>
            ))}
          </div>
        </Card>

        <Card className="surface-card surface-card--editorial">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Primary flow</div>
              <h2 className="section-heading__title">正式版默认怎么走</h2>
              <p className="section-heading__description">
                默认路径先降低认知负担，再把用户带入真实训练与结果回流；高级节点式编辑暂时不抢默认入口。
              </p>
            </div>
          </div>

          <div className="editorial-stack">
            <div className="surface-note surface-note--dense">
              <strong>1. 先看组件与能力</strong>
              <p>明确问题向导、训练执行、结果后台和工作台各自负责什么，不再从空表单或空画布直接开始。</p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>2. 先回答问题，再收敛骨架</strong>
              <p>从问题定义出发，映射到当前 runtime 已支持的模板和参数编辑层，而不是先手写 YAML 字段。</p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>3. 训练和结果继续复用主链</strong>
              <p>训练执行、artifact 构建和结果回流仍然保持可预测、可审计、可回放的 CLI / runtime 语义。</p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>4. 节点式编辑保留为高级模式</strong>
              <p>当前阶段优先默认模式，节点图承担后续高级结构编辑，不直接取代向导式入口。</p>
            </div>

            <Button
              icon={<ControlOutlined />}
              onClick={() => navigate("/config")}
            >
              进入问题向导
            </Button>
            <Button
              icon={<FileSearchOutlined />}
              onClick={() => navigate("/models")}
            >
              查看结果后台
            </Button>
            <Button
              icon={<ExperimentOutlined />}
              onClick={() => navigate("/quickstart-run")}
            >
              再看 quickstart 演示链路
            </Button>
          </div>
        </Card>
      </div>

      <Card className="surface-card" style={{ marginTop: 16 }} title="唯一主线（推荐）">
        <div className="editorial-stack">
          {START_RECOMMENDED_WORKFLOW.map((step) => (
            <div key={step} className="surface-note surface-note--dense">
              {step}
            </div>
          ))}
          <Button
            type="primary"
            icon={<ControlOutlined />}
            onClick={() => navigate("/config")}
          >
            按唯一主线开始
          </Button>
          <div className="surface-note surface-note--dense">
            <strong>可选模块：ComfyUI 适配（不改变主线）</strong>
            {START_COMFYUI_OPTIONAL_MODULE.map((step) => (
              <p key={step} style={{ marginBottom: 0 }}>
                {step}
              </p>
            ))}
            <Button
              size="small"
              icon={<LinkOutlined />}
              onClick={() => navigate("/config/comfyui")}
            >
              打开 ComfyUI 适配模块
            </Button>
          </div>
        </div>
      </Card>
    </PageScaffold>
  );
}
