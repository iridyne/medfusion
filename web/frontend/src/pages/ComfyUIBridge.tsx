import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, Button, Card, Input, Space, Tag, Typography, message } from "antd";
import {
  ArrowLeftOutlined,
  CopyOutlined,
  LinkOutlined,
  ReloadOutlined,
} from "@ant-design/icons";

import { getComfyUIHealth, type ComfyUIHealthResponse } from "@/api/comfyui";
import PageScaffold from "@/components/layout/PageScaffold";

const { Paragraph, Text } = Typography;

const STORAGE_KEY = "medfusion.comfyui.base_url";
const DEFAULT_COMFYUI_BASE_URL = "http://127.0.0.1:8188";

export default function ComfyUIBridge() {
  const navigate = useNavigate();
  const [baseUrl, setBaseUrl] = useState<string>(() => {
    if (typeof window === "undefined") {
      return DEFAULT_COMFYUI_BASE_URL;
    }
    return localStorage.getItem(STORAGE_KEY) || DEFAULT_COMFYUI_BASE_URL;
  });
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<ComfyUIHealthResponse | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);

  const probeTag = useMemo(() => {
    if (!health) {
      return { color: "default", text: "未检查" };
    }
    if (health.probe.reachable) {
      return { color: "success", text: "已连通" };
    }
    return { color: "error", text: "未连通" };
  }, [health]);

  const checkConnection = async (targetBaseUrl?: string) => {
    const nextBaseUrl = (targetBaseUrl || baseUrl).trim();
    if (!nextBaseUrl) {
      message.error("请输入 ComfyUI 地址");
      return;
    }

    setLoading(true);
    setRequestError(null);
    try {
      const payload = await getComfyUIHealth(nextBaseUrl);
      setHealth(payload);
      if (typeof window !== "undefined") {
        localStorage.setItem(STORAGE_KEY, nextBaseUrl);
      }
      if (payload.probe.reachable) {
        message.success("ComfyUI 已连通");
      }
    } catch (error: any) {
      const detail = error?.response?.data?.detail;
      const detailMessage =
        typeof detail?.message === "string"
          ? detail.message
          : "ComfyUI 检查失败，请确认后端服务可用";
      setRequestError(detailMessage);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void checkConnection(baseUrl);
  }, []);

  const handleOpenComfyUI = () => {
    const target = (health?.open_url || baseUrl).trim();
    if (!target) {
      message.error("当前没有可打开的 ComfyUI 地址");
      return;
    }
    window.open(target, "_blank", "noopener,noreferrer");
  };

  const handleCopyCommand = async () => {
    if (!health?.recommended_start_command) {
      return;
    }
    await navigator.clipboard.writeText(health.recommended_start_command);
    message.success("启动命令已复制");
  };

  return (
    <PageScaffold
      eyebrow="ComfyUI bridge"
      title="ComfyUI 上线入口（预览）"
      description="这里提供 ComfyUI 连通性检查、快速打开和最小回流指导。ComfyUI 负责工作流画布与生成交互，MedFusion 继续负责训练与结果合同。"
      chips={[
        { label: "Bridge preview", tone: "amber" },
        { label: "Connectivity check", tone: "blue" },
        { label: "Mainline handoff", tone: "teal" },
      ]}
      actions={
        <>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate("/config")}>
            返回问题向导
          </Button>
          <Button
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={() => void checkConnection()}
          >
            重新检查
          </Button>
          <Button
            type="primary"
            icon={<LinkOutlined />}
            onClick={handleOpenComfyUI}
          >
            打开 ComfyUI
          </Button>
        </>
      }
      metrics={[
        {
          label: "Connection",
          value: probeTag.text,
          hint: health?.probe.message || "先完成一次健康检查",
          tone: health?.probe.reachable ? "teal" : "rose",
        },
        {
          label: "Base URL",
          value: health?.base_url || baseUrl,
          hint: "可改为你的私有 ComfyUI 地址",
          tone: "blue",
        },
        {
          label: "Latency",
          value:
            typeof health?.probe.latency_ms === "number"
              ? `${health.probe.latency_ms} ms`
              : "-",
          hint: health?.probe.probe_url || "/system_stats",
          tone: "amber",
        },
      ]}
    >
      <div className="split-grid">
        <Card className="surface-card" title="连接配置">
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <Text type="secondary">
              默认地址是本机 ComfyUI：`http://127.0.0.1:8188`。你也可以换成局域网或服务器地址。
            </Text>
            <Input
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
              placeholder="http://127.0.0.1:8188"
            />
            <Space>
              <Button
                type="primary"
                loading={loading}
                onClick={() => void checkConnection()}
              >
                检查连通性
              </Button>
              <Tag color={probeTag.color}>{probeTag.text}</Tag>
            </Space>

            {requestError ? (
              <Alert type="error" showIcon message={requestError} />
            ) : null}

            {health ? (
              <Alert
                type={health.probe.reachable ? "success" : "warning"}
                showIcon
                message={health.probe.message}
                description={`Probe: ${health.probe.probe_url}`}
              />
            ) : null}
          </Space>
        </Card>

        <Card className="surface-card" title="最小上线链路">
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <div className="surface-note surface-note--dense">
              <strong>1. 启动 ComfyUI 服务</strong>
              <Paragraph style={{ marginBottom: 0 }}>
                在 ComfyUI 环境里执行启动命令，确认端口可访问。
              </Paragraph>
              <pre className="command-block">
                {health?.recommended_start_command ||
                  "python main.py --listen 127.0.0.1 --port 8188"}
              </pre>
              <Button
                size="small"
                icon={<CopyOutlined />}
                onClick={() => void handleCopyCommand()}
              >
                复制启动命令
              </Button>
            </div>

            <div className="surface-note surface-note--dense">
              <strong>2. 在 ComfyUI 完成画布流程</strong>
              <Paragraph style={{ marginBottom: 0 }}>
                先用 ComfyUI 完成你需要的生成或预处理图，产出数据与中间结果。
              </Paragraph>
            </div>

            <div className="surface-note surface-note--dense">
              <strong>3. 回到 MedFusion 主链训练和回流</strong>
              <Paragraph style={{ marginBottom: 0 }}>
                ComfyUI 不替代 MedFusion 训练执行层。完成前置后，继续用 Run Wizard / CLI 做可复现训练。
              </Paragraph>
              <pre className="command-block">
                uv run medfusion train --config &lt;your_config.yaml&gt;
              </pre>
            </div>
          </Space>
        </Card>
      </div>

      {health?.handoff_hint ? (
        <Alert
          style={{ marginTop: 16 }}
          type="info"
          showIcon
          message="主链边界"
          description={health.handoff_hint}
        />
      ) : null}
    </PageScaffold>
  );
}
