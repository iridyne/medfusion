import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Tag,
  Typography,
  message,
} from "antd";
import {
  CopyOutlined,
  ExperimentOutlined,
  LinkOutlined,
  RadarChartOutlined,
} from "@ant-design/icons";

import {
  runIndependentEvaluation,
  type EvaluationRequest,
  type EvaluationResponse,
} from "@/api/evaluation";
import PageScaffold from "@/components/layout/PageScaffold";

const { Paragraph, Text } = Typography;

interface EvaluationPrefillState {
  configPath?: string;
  checkpointPath?: string;
  outputDir?: string;
  name?: string;
  description?: string;
  source?: string;
}

interface EvaluationFormValues {
  config_path: string;
  checkpoint_path: string;
  output_dir?: string;
  split: "train" | "val" | "test" | "all";
  attention_samples: number;
  enable_survival: boolean;
  survival_time_column?: string;
  survival_event_column?: string;
  enable_importance: boolean;
  importance_sample_limit: number;
  import_to_model_library: boolean;
  name?: string;
  description?: string;
  tags?: string;
}

function formatPercent(value?: number | null): string {
  return value === undefined || value === null ? "-" : `${(value * 100).toFixed(2)}%`;
}

function formatMetric(value?: number | null): string {
  return value === undefined || value === null ? "-" : value.toFixed(4);
}

export default function EvaluationWorkbench() {
  const navigate = useNavigate();
  const location = useLocation();
  const [form] = Form.useForm<EvaluationFormValues>();
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<EvaluationResponse | null>(null);
  const [prefillSource, setPrefillSource] = useState<string | null>(null);

  useEffect(() => {
    const state = location.state as EvaluationPrefillState | undefined;
    if (!state?.configPath || !state?.checkpointPath) {
      return;
    }
    form.setFieldsValue({
      config_path: state.configPath,
      checkpoint_path: state.checkpointPath,
      output_dir: state.outputDir,
      name: state.name,
      description: state.description,
    });
    setPrefillSource(state.source || "model-library");
  }, [form, location.state]);

  const currentValues = Form.useWatch([], form) as Partial<EvaluationFormValues> | undefined;
  const evaluateCommand = useMemo(() => {
    const configPath = currentValues?.config_path || "<config>";
    const checkpointPath = currentValues?.checkpoint_path || "<checkpoint>";
    const split = currentValues?.split || "test";
    return `uv run medfusion evaluate --config ${configPath} --checkpoint ${checkpointPath} --split ${split}`;
  }, [currentValues]);

  const buildResultsCommand = useMemo(() => {
    const configPath = currentValues?.config_path || "<config>";
    const checkpointPath = currentValues?.checkpoint_path || "<checkpoint>";
    return `uv run medfusion build-results --config ${configPath} --checkpoint ${checkpointPath}`;
  }, [currentValues]);

  const copyText = async (value: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(value);
      message.success(successMessage);
    } catch (error) {
      console.error("Clipboard write failed:", error);
      message.error("复制失败，请检查浏览器权限");
    }
  };

  const handleSubmit = async (values: EvaluationFormValues) => {
    setRunning(true);
    try {
      const payload: EvaluationRequest = {
        ...values,
        output_dir: values.output_dir?.trim() || undefined,
        survival_time_column: values.survival_time_column?.trim() || undefined,
        survival_event_column: values.survival_event_column?.trim() || undefined,
        name: values.name?.trim() || undefined,
        description: values.description?.trim() || undefined,
        tags: (values.tags || "")
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
      };
      const response = await runIndependentEvaluation(payload);
      setResult(response);
      message.success(
        response.model_library_import.imported
          ? "独立评估完成，结果已导入结果后台"
          : "独立评估完成",
      );
    } catch (error: any) {
      console.error("Independent evaluation failed:", error);
      const detail = error?.response?.data?.detail;
      message.error(typeof detail === "string" ? detail : "独立评估失败");
    } finally {
      setRunning(false);
    }
  };

  return (
    <PageScaffold
      eyebrow="Independent evaluation"
      title="独立评估模块：用现有 config + checkpoint 直接生成评估产物"
      description="这条入口不要求先重跑训练。它直接复用现有结果构建链，对已有 checkpoint 做独立评估，并可选把结果导入结果后台。当前阶段先只支持单次、单 checkpoint 评估。"
      chips={[
        { label: "Post-run module", tone: "teal" },
        { label: "Checkpoint-first", tone: "amber" },
        { label: "Optional model-library import", tone: "blue" },
      ]}
      actions={
        <>
          <Button icon={<ExperimentOutlined />} onClick={() => navigate("/training")}>
            回到训练监控
          </Button>
          <Button icon={<RadarChartOutlined />} onClick={() => navigate("/models")}>
            打开结果后台
          </Button>
          <Button icon={<CopyOutlined />} onClick={() => void copyText(evaluateCommand, "评估命令已复制")}>
            复制评估命令
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Evaluation posture</span>
          <div className="hero-aside-panel__value">单次单 checkpoint 评估</div>
          <div className="hero-aside-panel__copy">
            这不是新的训练入口，而是训练后的独立评估模块。它适合补跑 `validation / summary / report`，也适合对已有 checkpoint 做一次新的结果构建。多模型对比、批量评估和评估模板中心后续再做。
          </div>
          {prefillSource ? (
            <div className="surface-note">
              当前预填来源：
              <strong> {prefillSource}</strong>
            </div>
          ) : null}
          {result ? (
            <div className="surface-note">
              当前结果目录：
              <strong> {result.output_dir}</strong>
            </div>
          ) : null}
        </div>
      }
      metrics={[
        {
          label: "Mode",
          value: result?.mode === "evaluate_and_import" ? "Evaluate + import" : "Evaluate only",
          hint: "无需重跑训练",
          tone: "teal",
        },
        {
          label: "Split",
          value: result?.summary.split || currentValues?.split || "test",
          hint: "当前评估目标数据切分",
          tone: "amber",
        },
        {
          label: "Imported",
          value: result?.model_library_import.imported ? "Yes" : "No",
          hint: "是否已进入结果后台",
          tone: result?.model_library_import.imported ? "blue" : "rose",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="评估模块不是新的训练 runtime"
        description="这里直接复用现有 build-results 链路。区别只是：它把“checkpoint + config 的独立评估”单独抬成了一个用户可见模块，不再只能通过 CLI 或结果导入隐式触发。当前先只支持单次、单 checkpoint 评估。"
      />
      {prefillSource ? (
        <Alert
          type="success"
          showIcon
          style={{ marginBottom: 16 }}
          message="已收到上游页面预填"
          description={`当前表单来自 ${prefillSource}，你可以直接补评估参数后执行。`}
        />
      ) : null}

      <div className="split-grid">
        <Card className="surface-card">
          <Form<EvaluationFormValues>
            form={form}
            layout="vertical"
            initialValues={{
              split: "test",
              attention_samples: 4,
              enable_survival: true,
              enable_importance: true,
              importance_sample_limit: 128,
              import_to_model_library: true,
              config_path: "configs/starter/quickstart.yaml",
            }}
            onFinish={(values) => void handleSubmit(values)}
          >
            <Row gutter={16}>
              <Col span={24}>
                <Form.Item
                  label="配置文件路径"
                  name="config_path"
                  rules={[{ required: true, message: "请输入训练配置 YAML 路径" }]}
                >
                  <Input placeholder="例如：configs/starter/quickstart.yaml" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item
                  label="模型权重路径"
                  name="checkpoint_path"
                  rules={[{ required: true, message: "请输入 checkpoint 路径" }]}
                >
                  <Input placeholder="例如：outputs/quickstart/checkpoints/best.pth" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="评估输出目录（可选）" name="output_dir">
                  <Input placeholder="例如：outputs/quickstart-eval" />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="评估 split" name="split">
                  <Select
                    options={[
                      { label: "test", value: "test" },
                      { label: "val", value: "val" },
                      { label: "train", value: "train" },
                      { label: "all", value: "all" },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="Attention 样本数" name="attention_samples">
                  <InputNumber min={0} max={16} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="Importance 样本数" name="importance_sample_limit">
                  <InputNumber min={0} max={512} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="启用 survival 分析" name="enable_survival" valuePropName="checked">
                  <Switch />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="启用变量重要性" name="enable_importance" valuePropName="checked">
                  <Switch />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="导入结果后台" name="import_to_model_library" valuePropName="checked">
                  <Switch />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="Survival 时间列（可选）" name="survival_time_column">
                  <Input placeholder="例如：survival_time" />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="Survival 事件列（可选）" name="survival_event_column">
                  <Input placeholder="例如：event" />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="导入名称（可选）" name="name">
                  <Input placeholder="例如：quickstart-eval-model" />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="标签（可选）" name="tags">
                  <Input placeholder="多个标签用英文逗号分隔" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="说明（可选）" name="description">
                  <Input.TextArea rows={3} placeholder="记录这次独立评估的目的或来源" />
                </Form.Item>
              </Col>
            </Row>

            <Space wrap>
              <Button type="primary" htmlType="submit" loading={running}>
                {running ? "评估中..." : "开始独立评估"}
              </Button>
              <Button onClick={() => void copyText(evaluateCommand, "评估命令已复制")}>
                复制评估命令
              </Button>
              <Button onClick={() => void copyText(buildResultsCommand, "build-results 命令已复制")}>
                复制 build-results 命令
              </Button>
            </Space>
          </Form>
        </Card>

        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card className="surface-card" title="推荐 CLI">
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <div>
                <Text strong>独立评估</Text>
                <pre className="command-block">{evaluateCommand}</pre>
              </div>
              <div>
                <Text strong>结果构建底层命令</Text>
                <pre className="command-block">{buildResultsCommand}</pre>
              </div>
              <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                Web 入口适合客户主机上的日常操作；CLI 仍然保留为批量自动化和可复现路径。
              </Paragraph>
            </Space>
          </Card>

          {result ? (
            <Card className="surface-card" title="评估摘要">
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Alert
                  type={result.model_library_import.imported ? "success" : "info"}
                  showIcon
                  message={result.next_step}
                />
                <Row gutter={[16, 16]}>
                  <Col xs={24} sm={8}>
                    <Statistic title="样本数" value={result.summary.sample_count ?? "-"} />
                  </Col>
                  <Col xs={24} sm={8}>
                    <Statistic title="Accuracy" value={formatPercent(result.summary.accuracy)} />
                  </Col>
                  <Col xs={24} sm={8}>
                    <Statistic title="ROC AUC" value={formatMetric(result.summary.auc)} />
                  </Col>
                </Row>
                <Descriptions column={1} size="small" labelStyle={{ width: 140 }}>
                  <Descriptions.Item label="实验名">
                    {result.summary.experiment_name}
                  </Descriptions.Item>
                  <Descriptions.Item label="输出目录">
                    <div style={{ wordBreak: "break-all" }}>{result.output_dir}</div>
                  </Descriptions.Item>
                  <Descriptions.Item label="导入结果后台">
                    {result.model_library_import.imported ? (
                      <Space wrap>
                        <Tag color="success">已导入</Tag>
                        <Text>{result.model_library_import.model_name}</Text>
                        <Button
                          size="small"
                          icon={<LinkOutlined />}
                          onClick={() =>
                            navigate(`/models?modelId=${result.model_library_import.model_id}`)
                          }
                        >
                          打开结果详情
                        </Button>
                      </Space>
                    ) : (
                      <Tag>未导入</Tag>
                    )}
                  </Descriptions.Item>
                </Descriptions>
              </Space>
            </Card>
          ) : (
            <Card className="surface-card" title="模块边界">
              <Space direction="vertical" size={10} style={{ width: "100%" }}>
                <div className="surface-note surface-note--dense">
                  <strong>它解决什么问题</strong>
                  <p>给已有 checkpoint 单独补跑 validation / summary / report，而不是强制重新训练。</p>
                </div>
                <div className="surface-note surface-note--dense">
                  <strong>它不做什么</strong>
                  <p>不替代训练主链，也不重新定义模型结构；只做 post-run 评估与可选结果归档。</p>
                </div>
              </Space>
            </Card>
          )}
        </Space>
      </div>

      {result ? (
        <Card className="surface-card" style={{ marginTop: 16 }} title="产物路径">
          <Space direction="vertical" size={10} style={{ width: "100%" }}>
            {Object.entries(result.artifact_paths).map(([key, path]) => (
              <div className="surface-note surface-note--dense" key={key}>
                <strong>{key}</strong>
                <p style={{ wordBreak: "break-all" }}>{path}</p>
              </div>
            ))}
          </Space>
        </Card>
      ) : null}
    </PageScaffold>
  );
}
