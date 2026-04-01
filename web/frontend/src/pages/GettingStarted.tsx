import { useNavigate } from "react-router-dom";
import { Alert, Button, Card, Tag } from "antd";
import {
  ArrowRightOutlined,
  ExperimentOutlined,
  FileSearchOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";

import PageScaffold from "@/components/layout/PageScaffold";
import { PRIMARY_ENTRY_COMMAND } from "@/config/navigation";
import {
  START_CHECKS,
  START_NEXT_STEPS,
} from "@/config/startExperience";

export default function GettingStarted() {
  const navigate = useNavigate();

  return (
    <PageScaffold
      eyebrow="Guided first run"
      title="先跑通一次，再开始扩展 MedFusion"
      description="第一次进入时，先沿着一条推荐 quickstart 走通公开数据、训练和结果构建。Web 负责把路径讲清楚，真实执行仍然回到同一条 CLI 主链。"
      chips={[
        { label: "Beginner-first", tone: "amber" },
        { label: "CLI-backed", tone: "blue" },
        { label: "Artifact-aware", tone: "teal" },
      ]}
      actions={
        <>
          <Button
            type="primary"
            size="large"
            icon={<PlayCircleOutlined />}
            onClick={() => navigate("/quickstart-run")}
          >
            运行推荐 quickstart
          </Button>
          <Button
            size="large"
            icon={<ArrowRightOutlined />}
            onClick={() => navigate("/workbench")}
          >
            跳到工作台总览
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Primary entry</span>
          <div className="hero-aside-panel__value">默认从统一入口进入引导页</div>
          <div className="hero-aside-panel__copy">
            第一次成功的目标不是看懂所有页面，而是完成一次真实的主链运行。
          </div>
          <pre className="command-block">{PRIMARY_ENTRY_COMMAND}</pre>
          <div className="surface-note">
            推荐下一步：
            <strong> {START_NEXT_STEPS.join(" -> ")}</strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Recommended path",
          value: "public dataset quickstart",
          hint: "prepare -> validate-config -> train -> build-results",
          tone: "amber",
        },
        {
          label: "Research path",
          value: "YAML + medfusion",
          hint: "For reproduction and mature experiment assembly",
          tone: "blue",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="第一次成功标准"
        description="你应该至少得到 checkpoint、metrics.json、validation.json、summary.json 和 report.md，并知道这些文件写到了哪里。"
      />

      <div className="split-grid">
        <Card className="surface-card surface-card--accent">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Readiness checks</div>
              <h2 className="section-heading__title">四项最小检查</h2>
              <p className="section-heading__description">
                先确认入口、依赖、静态资源和输出路径，再开始第一次运行。
              </p>
            </div>
            <Tag color="processing">{START_CHECKS.length} checks</Tag>
          </div>

          <div className="workbench-flow">
            {START_CHECKS.map((item, index) => (
              <div key={item.key} className="flow-step">
                <strong>
                  {index + 1}. {item.title}
                </strong>
                <p>{item.description}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card className="surface-card surface-card--editorial">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">What happens next</div>
              <h2 className="section-heading__title">推荐第一次运行链路</h2>
              <p className="section-heading__description">
                这不是另一套 Web runtime，而是对现有主链的引导和解释层。
              </p>
            </div>
          </div>

          <div className="editorial-stack">
            <div className="surface-note surface-note--dense">
              <strong>1. 准备数据</strong>
              <p>使用公开数据 profile，减少私有数据准备带来的变量。</p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>2. 校验配置并训练</strong>
              <p>继续使用真实 YAML 和真实 CLI 主链，不发明另一套执行语义。</p>
            </div>
            <div className="surface-note surface-note--dense">
              <strong>3. 构建结果并解释输出</strong>
              <p>跑完后回到 Web 读 summary、validation、report 和关键 artifacts。</p>
            </div>

            <Button
              icon={<ExperimentOutlined />}
              onClick={() => navigate("/quickstart-run")}
            >
              打开 quickstart 链路
            </Button>
            <Button
              icon={<FileSearchOutlined />}
              onClick={() => navigate("/models")}
            >
              直接查看结果入口
            </Button>
          </div>
        </Card>
      </div>
    </PageScaffold>
  );
}
