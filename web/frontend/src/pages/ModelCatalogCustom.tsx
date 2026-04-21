import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Row,
  Select,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ArrowLeftOutlined,
  ControlOutlined,
  DeleteOutlined,
  HistoryOutlined,
  InboxOutlined,
  SaveOutlined,
} from "@ant-design/icons";

import {
  deleteCustomModelEntry,
  getModelCatalog,
  getCustomModelHistory,
  getCustomModels,
  restoreCustomModelEntry,
  saveCustomModelEntry,
  undeleteCustomModelEntry,
  updateCustomModelRetentionPolicy,
  type ModelCatalogComponent,
  type ModelCatalogResponse,
  type ModelCatalogTemplate,
  type CustomModelEntry,
} from "@/api/models";
import { getUIPreferences } from "@/api/system";
import PageScaffold from "@/components/layout/PageScaffold";
import {
  composeCustomModelEntry,
} from "@/utils/customModelCatalog";

const { Paragraph, Text } = Typography;

const STATUS_COLORS: Record<string, string> = {
  compile_ready: "success",
  conditional: "warning",
  draft_only: "default",
  local_custom: "processing",
};

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

export default function ModelCatalogCustom() {
  const navigate = useNavigate();
  const [catalog, setCatalog] = useState<ModelCatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [customModels, setCustomModels] = useState<CustomModelEntry[]>([]);
  const [trashedModels, setTrashedModels] = useState<CustomModelEntry[]>([]);
  const [form] = Form.useForm<{
    name: string;
    description?: string;
    baseModelId: string;
  }>();
  const [baseModelId, setBaseModelId] = useState<string>("quickstart_multimodal");
  const [slotSelections, setSlotSelections] = useState<Record<string, string>>({});
  const [customRootDir, setCustomRootDir] = useState<string>("");
  const [customSchemaVersion, setCustomSchemaVersion] = useState<string>("");
  const [customFormatContract, setCustomFormatContract] = useState<string>("");
  const [historyBackend, setHistoryBackend] = useState<string>("");
  const [retentionScope, setRetentionScope] = useState<string>("");
  const [retentionFloorScope, setRetentionFloorScope] = useState<string>("");
  const [historyDisplayMode, setHistoryDisplayMode] = useState<"friendly" | "technical">(
    "friendly",
  );
  const [retentionPolicy, setRetentionPolicy] = useState<{
    mode: "count" | "time";
    max_count: number;
    max_age_days: number;
    min_count_per_model: number;
  }>({
    mode: "count",
    max_count: 40,
    max_age_days: 90,
    min_count_per_model: 3,
  });
  const [historyModalOpen, setHistoryModalOpen] = useState(false);
  const [selectedHistoryEntry, setSelectedHistoryEntry] = useState<CustomModelEntry | null>(null);
  const [historyItems, setHistoryItems] = useState<
    Array<{ commit: string; committed_at: string; subject: string }>
  >([]);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const payload = await getModelCatalog();
        setCatalog(payload);
        const customPayload = await getCustomModels();
        const uiPreferences = await getUIPreferences();
        setCustomModels(customPayload.items);
        setTrashedModels(customPayload.trash_items);
        setCustomRootDir(customPayload.root_dir);
        setCustomSchemaVersion(customPayload.schema_version);
        setCustomFormatContract(customPayload.format_contract);
        setHistoryBackend(customPayload.history_backend);
        setRetentionScope(customPayload.retention_scope);
        setRetentionFloorScope(customPayload.retention_floor_scope);
        setRetentionPolicy(customPayload.retention_policy);
        setHistoryDisplayMode(uiPreferences.preferences.history_display_mode);
      } catch (error) {
        console.error("Failed to load model catalog for custom page:", error);
        message.error("加载官方模型目录失败");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, []);

  const officialModels = useMemo(
    () => (catalog?.models || []).filter((item) => item.source === "official"),
    [catalog],
  );
  const officialUnits = useMemo(
    () => (catalog?.units || []).filter((item) => item.source === "official"),
    [catalog],
  );

  const baseModel = useMemo(
    () => officialModels.find((item) => item.id === baseModelId) || null,
    [officialModels, baseModelId],
  );

  useEffect(() => {
    if (!baseModel) {
      return;
    }
    setSlotSelections(baseModel.unit_map || {});
    form.setFieldsValue({ baseModelId: baseModel.id });
  }, [baseModel, form]);

  const slotOptions = useMemo(() => {
    if (!baseModel?.editable_slots?.length) {
      return {};
    }
    return Object.fromEntries(
      baseModel.editable_slots.map((slot) => [
        slot,
        officialUnits.filter(
          (unit) => unit.family === slot && unit.status !== "draft_only",
        ),
      ]),
    ) as Record<string, ModelCatalogComponent[]>;
  }, [baseModel, officialUnits]);

  const selectedUnits = useMemo(
    () =>
      officialUnits.filter((unit) => Object.values(slotSelections).includes(unit.id)),
    [officialUnits, slotSelections],
  );

  const derivedRequirements = useMemo(
    () =>
      uniqueStrings([
        ...(baseModel?.data_requirements || []),
        ...selectedUnits.flatMap((unit) => unit.data_requirements),
      ]),
    [baseModel, selectedUnits],
  );

  const derivedCompute = useMemo(() => {
    if (!baseModel) {
      return null;
    }
    const profiles = [baseModel.compute_profile, ...selectedUnits.map((unit) => unit.compute_profile)];
    const order: Record<string, number> = { light: 1, medium: 2, heavy: 3 };
    const heaviest = profiles.reduce((current, candidate) =>
      (order[candidate.tier] || 0) > (order[current.tier] || 0) ? candidate : current,
    );
    return {
      tier: heaviest.tier,
      gpu_vram_hint: heaviest.gpu_vram_hint,
      notes: uniqueStrings(profiles.map((item) => item.notes)).join(" / "),
    };
  }, [baseModel, selectedUnits]);

  const handleSave = async () => {
    if (!baseModel) {
      message.error("请选择一个官方基础模型");
      return;
    }

    const values = await form.validateFields();
    const entry = composeCustomModelEntry({
      name: values.name,
      description: values.description,
      baseModel,
      unitSelections: slotSelections,
      allUnits: officialUnits,
    });
    await saveCustomModelEntry(entry);
    const customPayload = await getCustomModels();
    setCustomModels(customPayload.items);
    setTrashedModels(customPayload.trash_items);
    setCustomRootDir(customPayload.root_dir);
    setCustomSchemaVersion(customPayload.schema_version);
    setCustomFormatContract(customPayload.format_contract);
    setHistoryBackend(customPayload.history_backend);
    setRetentionScope(customPayload.retention_scope);
    setRetentionFloorScope(customPayload.retention_floor_scope);
    message.success("自定义模型已保存到本地文件模型来源");
    form.setFieldsValue({ name: "", description: "" });
  };

  const useCustomModel = (entry: CustomModelEntry) => {
    navigate("/config", {
      state: {
        modelPrefill: entry.wizard_prefill,
        source: "model-catalog-custom",
      },
    });
  };

  const saveRetentionPolicy = async (nextPolicy: {
    mode: "count" | "time";
    max_count: number;
    max_age_days: number;
    min_count_per_model: number;
  }) => {
    const response = await updateCustomModelRetentionPolicy(nextPolicy);
    setRetentionPolicy(response.policy);
    message.success("历史保留策略已更新");
  };

  const openHistory = async (entry: CustomModelEntry) => {
    const payload = await getCustomModelHistory(entry.id);
    setSelectedHistoryEntry(entry);
    setHistoryItems(payload.items);
    setHistoryModalOpen(true);
  };

  return (
    <PageScaffold
      eyebrow="Custom model source"
      title="用户自定义模型来源：基于官方最小组织单元，组合出新的本地模型模板"
      description="当前自定义来源不允许用户上传底层神经网络零件，而是基于官方可连接的最小黑盒，在允许替换的槽位里组合出新的本地模板。它先服务个人工作流，不进入服务端共享中心。"
      chips={[
        { label: "Official units only", tone: "amber" },
        { label: "Local custom models", tone: "teal" },
        { label: "No low-level parts", tone: "blue" },
      ]}
      actions={
        <>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate("/config/model")}>
            回到官方模型数据库
          </Button>
          <Button icon={<ControlOutlined />} onClick={() => navigate("/settings")}>
            打开显示设置
          </Button>
          <Button icon={<ControlOutlined />} onClick={() => navigate("/config")}>
            回到 Run Wizard
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Custom source policy</span>
          <div className="hero-aside-panel__value">允许自建，但只在官方单元上组合</div>
          <div className="hero-aside-panel__copy">
            当前阶段自定义模型来源是本地个人层能力，不开放服务端共享，不允许上传 pooling / FC 这类底层零件，也不允许绕过官方最小组织单元。
          </div>
          <div className="surface-note">
            当前本地模板：
            <strong> {customModels.length} 个</strong>
          </div>
          <div className="surface-note">
            当前回收站：
            <strong> {trashedModels.length} 个</strong>
          </div>
          {customRootDir ? (
            <div className="surface-note">
              存储目录：
              <strong> {customRootDir}</strong>
            </div>
          ) : null}
          {customSchemaVersion ? (
            <div className="surface-note">
              {historyDisplayMode === "technical" ? (
                <>
                  状态文件版本：
                  <strong> {customSchemaVersion}</strong>
                </>
              ) : (
                <>
                  版本记录格式：
                  <strong> 已启用</strong>
                </>
              )}
            </div>
          ) : null}
          {historyBackend ? (
            <div className="surface-note">
              {historyDisplayMode === "technical" ? (
                <>
                  历史后端：
                  <strong> {historyBackend}</strong>
                </>
              ) : (
                <>
                  版本记录：
                  <strong> 已启用</strong>
                </>
              )}
            </div>
          ) : null}
          {retentionScope ? (
            <div className="surface-note">
              保留策略范围：
              <strong> {retentionScope === "global_store" ? "整个本机自定义模型库" : retentionScope}</strong>
            </div>
          ) : null}
          {retentionFloorScope ? (
            <div className="surface-note">
              保底范围：
              <strong> {retentionFloorScope === "active_models_only" ? "仅当前存在模型" : retentionFloorScope}</strong>
            </div>
          ) : null}
        </div>
      }
      metrics={[
        {
          label: "Official base models",
          value: officialModels.length,
          hint: "作为自定义模板的起点",
          tone: "blue",
        },
        {
          label: "Composable slots",
          value: baseModel?.editable_slots?.length || 0,
          hint: "当前基础模型允许替换的槽位数",
          tone: "amber",
        },
        {
          label: "Official units",
          value: officialUnits.length,
          hint: "可选最小组织单元",
          tone: "teal",
        },
        {
          label: "Local custom models",
          value: customModels.length,
          hint: "当前本机文件已保存模板",
          tone: "rose",
        },
        {
          label: "Recycle bin",
          value: trashedModels.length,
          hint: "已移入回收站的模型",
          tone: "amber",
        },
      ]}
    >
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="当前自定义模型来源的边界"
          description={`允许自定义，但不是自由 builder。你只能从官方基础模型起步，在允许替换的槽位里选择官方最小组织单元，生成一个本地自定义模板。当前文件格式按 ${customFormatContract || "internal_state_file"} 处理。`}
        />

      <Card className="surface-card" style={{ marginBottom: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Retention</div>
            <h2 className="section-heading__title">本地历史保留策略</h2>
            <p className="section-heading__description">
              默认按条数保留，最多 40 条；如果后续需要，也可以切到按时间保留。
            </p>
          </div>
          <Tag color="processing">{historyBackend || "filesystem"}</Tag>
        </div>

        <Space wrap align="end">
          <div>
            <Text type="secondary">模式</Text>
            <Select
              style={{ width: 160, display: "block", marginTop: 8 }}
              value={retentionPolicy.mode}
              options={[
                { label: "按条数", value: "count" },
                { label: "按时间", value: "time" },
              ]}
              onChange={(value) => setRetentionPolicy((prev) => ({ ...prev, mode: value }))}
            />
          </div>
          {retentionPolicy.mode === "count" ? (
            <div>
              <Text type="secondary">最大保留条数</Text>
              <InputNumber
                style={{ display: "block", marginTop: 8 }}
                min={1}
                max={40}
                value={retentionPolicy.max_count}
                onChange={(value) =>
                  setRetentionPolicy((prev) => ({
                    ...prev,
                    max_count: Number(value ?? 40),
                  }))
                }
              />
            </div>
          ) : (
            <div>
              <Text type="secondary">最大保留天数</Text>
              <InputNumber
                style={{ display: "block", marginTop: 8 }}
                min={1}
                max={3650}
                value={retentionPolicy.max_age_days}
                onChange={(value) =>
                  setRetentionPolicy((prev) => ({
                    ...prev,
                    max_age_days: Number(value ?? 90),
                  }))
                }
              />
            </div>
          )}
          <div>
            <Text type="secondary">每模型至少保留</Text>
            <InputNumber
              style={{ display: "block", marginTop: 8 }}
              min={1}
              max={40}
              value={retentionPolicy.min_count_per_model}
              onChange={(value) =>
                setRetentionPolicy((prev) => ({
                  ...prev,
                  min_count_per_model: Number(value ?? 3),
                }))
              }
            />
          </div>
          <Button
            type="primary"
            onClick={() => void saveRetentionPolicy(retentionPolicy)}
          >
            保存策略
          </Button>
        </Space>
        <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
          当前策略作用于整个本机自定义模型库，而不是单个模型单独配置；在按条数模式下，会尽量保证当前仍存在的模型至少保留最近若干条历史，回收站里的模型不参与这个保底。
        </Paragraph>
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Card className="surface-card">
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Compose</div>
                <h2 className="section-heading__title">创建一个新的本地自定义模型</h2>
                <p className="section-heading__description">
                  先选官方基础模型，再在允许替换的槽位里调整 backbone / fusion / head 等最小组织单元。
                </p>
              </div>
              <Tag color="processing">Local only</Tag>
            </div>

            {loading || !baseModel ? (
              <Empty description="正在加载官方基础模型" />
            ) : (
              <Space direction="vertical" size={16} style={{ width: "100%" }}>
                <Form form={form} layout="vertical" initialValues={{ baseModelId }}>
                  <Form.Item
                    label="基础模型"
                    name="baseModelId"
                    rules={[{ required: true, message: "请选择基础模型" }]}
                  >
                    <Select
                      value={baseModelId}
                      options={officialModels.map((item) => ({
                        label: `${item.label} · ${item.status}`,
                        value: item.id,
                      }))}
                      onChange={(value) => setBaseModelId(value)}
                    />
                  </Form.Item>
                  <Form.Item
                    label="自定义模型名称"
                    name="name"
                    rules={[{ required: true, message: "请输入模型名称" }]}
                  >
                    <Input placeholder="例如：My EfficientNet Gated Baseline" />
                  </Form.Item>
                  <Form.Item label="说明" name="description">
                    <Input.TextArea
                      rows={3}
                      placeholder="记录这个自定义模板和官方基础模型相比改了什么"
                    />
                  </Form.Item>
                </Form>

                {baseModel.editable_slots?.map((slot) => (
                  <Card key={slot} size="small">
                    <Space direction="vertical" size={8} style={{ width: "100%" }}>
                      <Text strong>{slot}</Text>
                      <Select
                        value={slotSelections[slot]}
                        options={(slotOptions[slot] || []).map((item) => ({
                          label: `${item.label} · ${item.status}`,
                          value: item.id,
                        }))}
                        onChange={(value) =>
                          setSlotSelections((prev) => ({ ...prev, [slot]: value }))
                        }
                      />
                    </Space>
                  </Card>
                ))}

                <Button type="primary" icon={<SaveOutlined />} onClick={() => void handleSave()}>
                  保存为本地自定义模型
                </Button>
              </Space>
            )}
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card className="surface-card">
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Preview</div>
                <h2 className="section-heading__title">当前组合预览</h2>
                <p className="section-heading__description">
                  这里展示最终组合后的数据要求、算力估计和组成单元，避免自定义模型变成不可解释的大黑盒。
                </p>
              </div>
            </div>

            {baseModel ? (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <div className="surface-note surface-note--dense">
                  <strong>基础模型</strong>
                  <p>{baseModel.label}</p>
                </div>
                <div className="surface-note surface-note--dense">
                  <strong>当前单元链</strong>
                  <p>{Object.values(slotSelections).join(" -> ") || "-"}</p>
                </div>
                <div className="surface-note surface-note--dense">
                  <strong>数据要求</strong>
                  <p>{derivedRequirements.join(" / ") || "-"}</p>
                </div>
                {derivedCompute ? (
                  <div className="surface-note surface-note--dense">
                    <strong>算力需求</strong>
                    <p>
                      {derivedCompute.tier} · {derivedCompute.gpu_vram_hint} ·{" "}
                      {derivedCompute.notes}
                    </p>
                  </div>
                ) : null}
              </Space>
            ) : (
              <Empty description="请选择基础模型" />
            )}
          </Card>
        </Col>
      </Row>

      <Card className="surface-card" style={{ marginTop: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Saved local models</div>
            <h2 className="section-heading__title">已保存的本地自定义模型</h2>
            <p className="section-heading__description">
              这些模型当前保存在客户主机本地文件系统，适合个人工作流验证；后续如果要共享，再单独进入服务端来源与审核流程。
              {customRootDir ? ` 当前文件目录：${customRootDir}` : ""}
            </p>
          </div>
        </div>

        {customModels.length ? (
          <Row gutter={[16, 16]}>
            {customModels.map((entry) => (
              <Col xs={24} xl={12} key={entry.id}>
                <Card size="small">
                  <Space direction="vertical" size={10} style={{ width: "100%" }}>
                    <Space wrap>
                      <Text strong>{entry.label}</Text>
                      <Tag color={STATUS_COLORS[entry.status] || "default"}>{entry.status}</Tag>
                    </Space>
                    <Paragraph style={{ marginBottom: 0 }}>
                      {entry.description || "无补充说明"}
                    </Paragraph>
                    <div className="surface-note surface-note--dense">
                      <strong>基于</strong>
                      <p>{entry.based_on_model_id}</p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>单元链</strong>
                      <p>{entry.component_ids.join(" -> ")}</p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>算力需求</strong>
                      <p>
                        {entry.compute_profile.tier} · {entry.compute_profile.gpu_vram_hint}
                      </p>
                    </div>
                    <Space wrap>
                      <Button
                        type="primary"
                        icon={<ControlOutlined />}
                        onClick={() => useCustomModel(entry)}
                      >
                        带入 Run Wizard
                      </Button>
                      <Button
                        icon={<HistoryOutlined />}
                        onClick={() => void openHistory(entry)}
                      >
                        查看历史
                      </Button>
                      <Button
                        danger
                        icon={<DeleteOutlined />}
                        onClick={async () => {
                          await deleteCustomModelEntry(entry.id);
                          const customPayload = await getCustomModels();
                          setCustomModels(customPayload.items);
                          setTrashedModels(customPayload.trash_items);
                          setCustomRootDir(customPayload.root_dir);
                          setCustomSchemaVersion(customPayload.schema_version);
                          setCustomFormatContract(customPayload.format_contract);
                          setHistoryBackend(customPayload.history_backend);
                          setRetentionScope(customPayload.retention_scope);
                          setRetentionFloorScope(customPayload.retention_floor_scope);
                          setRetentionPolicy(customPayload.retention_policy);
                          message.success("已删除本地自定义模型文件");
                        }}
                      >
                        删除
                      </Button>
                    </Space>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Empty description="还没有本地自定义模型" />
        )}
      </Card>

      <Card className="surface-card" style={{ marginTop: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Recycle bin</div>
            <h2 className="section-heading__title">回收站</h2>
            <p className="section-heading__description">
              删除不会立刻永久清除，而是先移入回收站。当前不提供“彻底清除”按钮，只允许恢复；旧记录会由全局保留策略自动处理。
            </p>
          </div>
        </div>

        {trashedModels.length ? (
          <Row gutter={[16, 16]}>
            {trashedModels.map((entry) => (
              <Col xs={24} xl={12} key={entry.id}>
                <Card size="small">
                  <Space direction="vertical" size={10} style={{ width: "100%" }}>
                    <Space wrap>
                      <Text strong>{entry.label}</Text>
                      <Tag color="warning">in recycle bin</Tag>
                    </Space>
                    <Paragraph style={{ marginBottom: 0 }}>
                      {entry.description || "无补充说明"}
                    </Paragraph>
                    <div className="surface-note surface-note--dense">
                      <strong>基于</strong>
                      <p>{entry.based_on_model_id}</p>
                    </div>
                    <Space wrap>
                      <Button
                        type="primary"
                        icon={<InboxOutlined />}
                        onClick={async () => {
                          await undeleteCustomModelEntry(entry.id);
                          const customPayload = await getCustomModels();
                          setCustomModels(customPayload.items);
                          setTrashedModels(customPayload.trash_items);
                          setCustomRootDir(customPayload.root_dir);
                          setCustomSchemaVersion(customPayload.schema_version);
                          setCustomFormatContract(customPayload.format_contract);
                          setHistoryBackend(customPayload.history_backend);
                          setRetentionScope(customPayload.retention_scope);
                          setRetentionFloorScope(customPayload.retention_floor_scope);
                          setRetentionPolicy(customPayload.retention_policy);
                          message.success("已从回收站恢复到当前模型列表");
                        }}
                      >
                        恢复到当前列表
                      </Button>
                    </Space>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Empty description="回收站为空" />
        )}
      </Card>

      <Modal
        title={selectedHistoryEntry ? `${selectedHistoryEntry.label} · 版本历史` : "版本历史"}
        open={historyModalOpen}
        onCancel={() => setHistoryModalOpen(false)}
        footer={null}
        width={760}
      >
        {historyItems.length ? (
          <List
            dataSource={historyItems}
            renderItem={(item) => (
              <List.Item
                actions={[
                      <Button
                        key="restore"
                        size="small"
                        onClick={() => {
                          if (!selectedHistoryEntry) {
                            return;
                          }
                          Modal.confirm({
                            title: "恢复历史版本",
                            content:
                              "恢复会直接覆盖当前这个自定义模型，并保留一条新的 git 历史记录。若当前版本仍需保留，建议先复制出一个新模型。",
                            okText: "覆盖恢复",
                            cancelText: "取消",
                            onOk: async () => {
                              await restoreCustomModelEntry(selectedHistoryEntry.id, item.commit);
                              const customPayload = await getCustomModels();
                              setCustomModels(customPayload.items);
                              setCustomRootDir(customPayload.root_dir);
                              setCustomSchemaVersion(customPayload.schema_version);
                              setCustomFormatContract(customPayload.format_contract);
                              setHistoryBackend(customPayload.history_backend);
                              setRetentionScope(customPayload.retention_scope);
                              setRetentionFloorScope(customPayload.retention_floor_scope);
                              const historyPayload = await getCustomModelHistory(selectedHistoryEntry.id);
                              setHistoryItems(historyPayload.items);
                              message.success("已覆盖当前模型并恢复到所选版本");
                            },
                          });
                        }}
                      >
                        恢复到此版本
                      </Button>,
                ]}
              >
                <List.Item.Meta
                  title={item.subject}
                  description={
                    <Space direction="vertical" size={2}>
                      <Text type="secondary">{item.committed_at}</Text>
                      {historyDisplayMode === "technical" ? (
                        <Text code>{item.commit.slice(0, 12)}</Text>
                      ) : null}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        ) : (
          <Empty description="当前还没有历史记录" />
        )}
      </Modal>
    </PageScaffold>
  );
}
