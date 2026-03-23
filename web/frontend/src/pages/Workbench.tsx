import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Progress,
  Row,
  Space,
  Statistic,
  Tag,
  Typography,
} from "antd";
import {
  ArrowRightOutlined,
  ControlOutlined,
  ExperimentOutlined,
  ImportOutlined,
  DatabaseOutlined,
  PlayCircleOutlined,
  RocketOutlined,
} from "@ant-design/icons";

import { getDatasetStatistics } from "@/api/datasets";
import { getModels, getModelStatistics } from "@/api/models";
import trainingApi from "@/api/training";

const { Paragraph, Text, Title } = Typography;

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

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size={20} style={{ width: "100%" }}>
        <Card
          bordered={false}
          style={{
            background:
              "linear-gradient(135deg, rgba(10,72,107,0.96) 0%, rgba(21,122,110,0.92) 55%, rgba(218,165,32,0.86) 100%)",
            color: "#fff",
            overflow: "hidden",
          }}
          bodyStyle={{ padding: 28 }}
        >
          <Row gutter={[24, 24]} align="middle">
            <Col xs={24} xl={16}>
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Tag
                  color="gold"
                  style={{
                    width: "fit-content",
                    color: "#4d3b00",
                    fontWeight: 700,
                    border: "none",
                  }}
                >
                  统一入口
                </Tag>
                <Title level={2} style={{ color: "#fff", margin: 0 }}>
                  MedFusion 工作台
                </Title>
                <Paragraph style={{ color: "rgba(255,255,255,0.88)", fontSize: 16, margin: 0 }}>
                  现在推荐从 `medfusion start` 进入。这里把演示型 MVP、真实训练结果导入和数据准备收口到一个起点，
                  避免用户一上来就在 CLI、YAML、Web 三条链之间迷路。
                </Paragraph>
                <Space wrap size={[12, 12]}>
                  <Button
                    type="primary"
                    size="large"
                    icon={<PlayCircleOutlined />}
                    onClick={() => navigate("/training?action=start")}
                  >
                    快速演示训练
                  </Button>
                  <Button
                    size="large"
                    icon={<ControlOutlined />}
                    onClick={() => navigate("/config")}
                  >
                    训练配置向导
                  </Button>
                  <Button
                    size="large"
                    icon={<ImportOutlined />}
                    onClick={() => navigate("/models?action=import")}
                  >
                    导入真实结果
                  </Button>
                  <Button
                    size="large"
                    icon={<DatabaseOutlined />}
                    onClick={() => navigate("/datasets")}
                  >
                    准备数据
                  </Button>
                </Space>
              </Space>
            </Col>

            <Col xs={24} xl={8}>
              <Card
                size="small"
                style={{
                  background: "rgba(255,255,255,0.14)",
                  borderColor: "rgba(255,255,255,0.18)",
                }}
              >
                <Space direction="vertical" size={10} style={{ width: "100%" }}>
                  <Text style={{ color: "rgba(255,255,255,0.88)" }}>推荐起步路径</Text>
                  <Text code style={{ width: "fit-content" }}>
                    uv run medfusion start
                  </Text>
                  <Text style={{ color: "rgba(255,255,255,0.88)" }}>
                    最近结果: {latestModelName}
                  </Text>
                  <Progress
                    percent={runningRatio}
                    strokeColor="#faad14"
                    trailColor="rgba(255,255,255,0.16)"
                    format={() => `${stats.runningJobs}/${stats.totalJobs || 0} 运行中`}
                  />
                </Space>
              </Card>
            </Col>
          </Row>
        </Card>

        <Alert
          type="info"
          showIcon
          message="当前主线已经收敛"
          description="工作台负责发现和引导，CLI 负责执行和复现，结果页负责展示 artifact。后面要统一的是配置语义，不是简单把 CLI 删掉。"
        />

        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} xl={6}>
            <Card loading={loading}>
              <Statistic title="数据集总数" value={stats.totalDatasets} prefix={<DatabaseOutlined />} />
            </Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card loading={loading}>
              <Statistic title="样本总数" value={stats.totalSamples} />
            </Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card loading={loading}>
              <Statistic title="模型记录" value={stats.totalModels} prefix={<RocketOutlined />} />
            </Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card loading={loading}>
              <Statistic
                title="平均准确率"
                value={stats.avgAccuracy * 100}
                precision={2}
                suffix="%"
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]}>
          <Col xs={24} md={12} xl={6}>
            <Card
              title="1. 演示型 MVP"
              extra={<Tag color="processing">对外展示</Tag>}
            >
              <Paragraph>
                适合你现在的推广节奏。直接启动一条演示训练，快速生成训练曲线、ROC、attention 热力图和结果页。
              </Paragraph>
              <Space direction="vertical" size={10} style={{ width: "100%" }}>
                <Text code>进入 Training Monitor 并直接打开“启动演示训练”</Text>
                <Button
                  type="primary"
                  block
                  icon={<ExperimentOutlined />}
                  onClick={() => navigate("/training?action=start")}
                >
                  进入演示训练
                </Button>
              </Space>
            </Card>
          </Col>

          <Col xs={24} md={12} xl={6}>
            <Card
              title="2. 真实训练向导"
              extra={<Tag color="blue">真实 schema</Tag>}
            >
              <Paragraph>
                用表单替代手填 YAML，直接生成和 CLI 主链一致的训练配置。拖拽式模型搭建后面也会复用这层 RunSpec。
              </Paragraph>
              <Space direction="vertical" size={10} style={{ width: "100%" }}>
                <Text code>先在向导里产出真实 config，再执行 train</Text>
                <Button
                  block
                  icon={<ControlOutlined />}
                  onClick={() => navigate("/config")}
                >
                  打开训练向导
                </Button>
              </Space>
            </Card>
          </Col>

          <Col xs={24} md={12} xl={6}>
            <Card
              title="3. 真实结果导入"
              extra={<Tag color="success">真实 artifact</Tag>}
            >
              <Paragraph>
                如果你已经通过 CLI 跑出了 `checkpoint`，这里直接导入到模型库，不需要再手动整理结果文件。
              </Paragraph>
              <Space direction="vertical" size={10} style={{ width: "100%" }}>
                <Text code>填写 config 路径和 checkpoint 路径即可</Text>
                <Button
                  block
                  icon={<ImportOutlined />}
                  onClick={() => navigate("/models?action=import")}
                >
                  打开导入弹窗
                </Button>
              </Space>
            </Card>
          </Col>

          <Col xs={24} md={12} xl={6}>
            <Card
              title="4. 数据准备"
              extra={<Tag color="warning">前置步骤</Tag>}
            >
              <Paragraph>
                先登记本地数据目录或准备公开验证数据集，再进入训练和结果展示。后面会继续把公开数据集接入成一键验证入口。
              </Paragraph>
              <Space direction="vertical" size={10} style={{ width: "100%" }}>
                <Text code>先准备 dataset，再开始真实训练</Text>
                <Button
                  block
                  icon={<ArrowRightOutlined />}
                  onClick={() => navigate("/datasets")}
                >
                  去数据管理
                </Button>
              </Space>
            </Card>
          </Col>
        </Row>

        <Card title="统一入口说明">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Space direction="vertical" size={8} style={{ width: "100%" }}>
                <Text strong>对用户讲的入口</Text>
                <Text code>uv run medfusion start</Text>
                <Text type="secondary">
                  启动 Web 工作台，适合演示、导入真实结果和后续做配置向导。
                </Text>
              </Space>
            </Col>
            <Col xs={24} lg={12}>
              <Space direction="vertical" size={8} style={{ width: "100%" }}>
                <Text strong>对工程侧保留的执行层</Text>
                <Text code>uv run medfusion train --config ...</Text>
                <Text type="secondary">
                  CLI 继续作为可复现、可自动化、适合远程 GPU 和 CI 的执行层存在。
                </Text>
              </Space>
            </Col>
          </Row>
        </Card>
      </Space>
    </div>
  );
}
