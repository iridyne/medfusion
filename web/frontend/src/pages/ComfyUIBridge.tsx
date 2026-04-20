import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Input,
  Row,
  Select,
  Skeleton,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import {
  AppstoreOutlined,
  ArrowLeftOutlined,
  CopyOutlined,
  ImportOutlined,
  LinkOutlined,
  ReloadOutlined,
} from "@ant-design/icons";

import {
  getComfyUIAdapterProfiles,
  getComfyUIHealth,
  type ComfyUIAdapterProfile,
  type ComfyUIHealthResponse,
} from "@/api/comfyui";
import PageScaffold from "@/components/layout/PageScaffold";

const { Paragraph, Text } = Typography;

const STORAGE_KEY = "medfusion.comfyui.base_url";
const DEFAULT_COMFYUI_BASE_URL = "http://127.0.0.1:8188";

interface ModelImportPrefill {
  config_path: string;
  checkpoint_path: string;
  output_dir?: string;
  split: "train" | "val" | "test";
  attention_samples: number;
  importance_sample_limit: number;
  name?: string;
  description?: string;
  tags?: string;
}

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
  const [adapterLoading, setAdapterLoading] = useState(false);
  const [adapterProfiles, setAdapterProfiles] = useState<ComfyUIAdapterProfile[]>([]);
  const [selectedAdapterProfileId, setSelectedAdapterProfileId] = useState<string>("");
  const [importPrefill, setImportPrefill] = useState<ModelImportPrefill>({
    config_path: "configs/starter/quickstart.yaml",
    checkpoint_path: "outputs/quickstart/checkpoints/best.pth",
    output_dir: "outputs/quickstart",
    split: "test",
    attention_samples: 4,
    importance_sample_limit: 128,
    name: "comfyui-handoff-run",
    description: "ComfyUI 预处理后回流到 MedFusion 结果后台",
    tags: "comfyui, handoff",
  });

  const probeTag = useMemo(() => {
    if (!health) {
      return { color: "default", text: "未检查" };
    }
    if (health.probe.reachable) {
      return { color: "success", text: "已连通" };
    }
    return { color: "error", text: "未连通" };
  }, [health]);

  const selectedAdapterProfile = useMemo(
    () =>
      adapterProfiles.find((item) => item.id === selectedAdapterProfileId) || null,
    [adapterProfiles, selectedAdapterProfileId],
  );

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

  const loadAdapterProfiles = async () => {
    setAdapterLoading(true);
    try {
      const payload = await getComfyUIAdapterProfiles();
      setAdapterProfiles(payload.profiles);
      if (!payload.profiles.length) {
        return;
      }
      const initial = payload.profiles[0];
      setSelectedAdapterProfileId((current) => current || initial.id);
      setImportPrefill((current) => ({
        ...current,
        ...initial.default_import_prefill,
      }));
    } catch (error) {
      console.error("Failed to load ComfyUI adapter profiles:", error);
      message.error("加载 ComfyUI 适配档案失败");
    } finally {
      setAdapterLoading(false);
    }
  };

  useEffect(() => {
    void loadAdapterProfiles();
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

  const handleOpenImportPrefill = () => {
    if (!importPrefill.config_path.trim() || !importPrefill.checkpoint_path.trim()) {
      message.error("请先填写配置路径和权重路径");
      return;
    }

    navigate("/models?action=import", {
      state: {
        importSource: "ComfyUI Bridge",
        importPrefill: {
          ...importPrefill,
          config_path: importPrefill.config_path.trim(),
          checkpoint_path: importPrefill.checkpoint_path.trim(),
          output_dir: importPrefill.output_dir?.trim() || undefined,
          name: importPrefill.name?.trim() || undefined,
          description: importPrefill.description?.trim() || undefined,
          tags: importPrefill.tags?.trim() || undefined,
        },
      },
    });
  };

  const handleSelectAdapterProfile = (profileId: string) => {
    setSelectedAdapterProfileId(profileId);
    const profile = adapterProfiles.find((item) => item.id === profileId);
    if (!profile) {
      return;
    }
    setImportPrefill((current) => ({
      ...current,
      ...profile.default_import_prefill,
    }));
  };

  const handleOpenAdapterCanvas = () => {
    if (!selectedAdapterProfile) {
      message.warning("请先选择一个适配档案");
      return;
    }
    navigate(selectedAdapterProfile.target_canvas_route);
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
        {
          label: "Import handoff",
          value: "Ready",
          hint: "预填后直接跳结果后台导入",
          tone: "teal",
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

      <Card className="surface-card" title="MedFusion 组件适配档案">
        {adapterLoading ? (
          <Skeleton active paragraph={{ rows: 3 }} />
        ) : (
          <Space direction="vertical" size={12} style={{ width: "100%" }}>
            <Alert
              type="info"
              showIcon
              message="ComfyUI 不是直接替代 MedFusion 训练层，而是接入适配档案后协同工作"
              description="先选适配档案，再进入对应的 MedFusion 组件骨架画布，这样 ComfyUI 流程和我们主链组件语义是一致的。"
            />
            <Select
              value={selectedAdapterProfileId || undefined}
              onChange={handleSelectAdapterProfile}
              placeholder="选择一个适配档案"
              options={adapterProfiles.map((profile) => ({
                label: profile.label,
                value: profile.id,
              }))}
            />
            {selectedAdapterProfile ? (
              <div className="surface-note surface-note--dense">
                <strong>{selectedAdapterProfile.label}</strong>
                <Paragraph style={{ marginBottom: 0 }}>
                  {selectedAdapterProfile.description}
                </Paragraph>
                <Space wrap style={{ marginTop: 8 }}>
                  {selectedAdapterProfile.family_chain.map((family) => (
                    <Tag key={family.family}>{family.label}</Tag>
                  ))}
                </Space>
              </div>
            ) : null}
            <Space>
              <Button
                type="primary"
                icon={<AppstoreOutlined />}
                onClick={handleOpenAdapterCanvas}
                disabled={!selectedAdapterProfile}
              >
                打开适配化组件画布
              </Button>
              <Button
                onClick={() => {
                  if (!selectedAdapterProfile) {
                    return;
                  }
                  handleSelectAdapterProfile(selectedAdapterProfile.id);
                  message.success("已按适配档案更新回流预填参数");
                }}
                disabled={!selectedAdapterProfile}
              >
                同步档案到回流预填
              </Button>
            </Space>
          </Space>
        )}
      </Card>

      <Card className="surface-card" title="ComfyUI -> 结果后台预填回流">
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <Alert
            type="info"
            showIcon
            message="把回流参数先在这里填好，再一键进入结果后台导入"
            description="这一步不会直接执行导入；它会把参数带到 Model Library 的导入弹窗，减少重复手填。"
          />
          <Row gutter={12}>
            <Col xs={24} xl={12}>
              <Text type="secondary">配置路径</Text>
              <Input
                value={importPrefill.config_path}
                onChange={(event) =>
                  setImportPrefill((current) => ({
                    ...current,
                    config_path: event.target.value,
                  }))
                }
                placeholder="configs/starter/quickstart.yaml"
              />
            </Col>
            <Col xs={24} xl={12}>
              <Text type="secondary">权重路径</Text>
              <Input
                value={importPrefill.checkpoint_path}
                onChange={(event) =>
                  setImportPrefill((current) => ({
                    ...current,
                    checkpoint_path: event.target.value,
                  }))
                }
                placeholder="outputs/quickstart/checkpoints/best.pth"
              />
            </Col>
            <Col xs={24} xl={8}>
              <Text type="secondary">输出目录（可选）</Text>
              <Input
                value={importPrefill.output_dir}
                onChange={(event) =>
                  setImportPrefill((current) => ({
                    ...current,
                    output_dir: event.target.value,
                  }))
                }
                placeholder="outputs/quickstart"
              />
            </Col>
            <Col xs={24} xl={8}>
              <Text type="secondary">Validation Split</Text>
              <Select
                style={{ width: "100%" }}
                value={importPrefill.split}
                onChange={(value: "train" | "val" | "test") =>
                  setImportPrefill((current) => ({ ...current, split: value }))
                }
                options={[
                  { label: "test", value: "test" },
                  { label: "val", value: "val" },
                  { label: "train", value: "train" },
                ]}
              />
            </Col>
            <Col xs={24} xl={8}>
              <Text type="secondary">标签（可选）</Text>
              <Input
                value={importPrefill.tags}
                onChange={(event) =>
                  setImportPrefill((current) => ({
                    ...current,
                    tags: event.target.value,
                  }))
                }
                placeholder="comfyui, handoff"
              />
            </Col>
          </Row>
          <Space>
            <Button
              type="primary"
              icon={<ImportOutlined />}
              onClick={handleOpenImportPrefill}
            >
              回流到结果后台导入
            </Button>
            <Button onClick={() => navigate("/models")}>直接打开结果后台</Button>
          </Space>
        </Space>
      </Card>

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
