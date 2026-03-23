import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Row,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import { ArrowRightOutlined, ProfileOutlined } from "@ant-design/icons";

import { getProjectTemplates, type ProjectTaskType, type ProjectTemplate } from "@/api/projects";

const { Paragraph, Text, Title } = Typography;

const TASK_LABELS: Record<ProjectTaskType, string> = {
  binary_classification: "二分类预测",
  cox_survival: "Cox 生存分析",
  multimodal_research: "多模态研究",
};

export default function TemplateCenter() {
  const navigate = useNavigate();
  const [templates, setTemplates] = useState<ProjectTemplate[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadTemplates = async () => {
      setLoading(true);
      try {
        setTemplates(await getProjectTemplates());
      } catch (error) {
        console.error("Failed to load project templates:", error);
        message.error("加载模板库失败");
      } finally {
        setLoading(false);
      }
    };

    void loadTemplates();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size={20} style={{ width: "100%" }}>
        <Card bordered={false}>
          <Space direction="vertical" size={8} style={{ width: "100%" }}>
            <Title level={2} style={{ margin: 0 }}>
              模板中心
            </Title>
            <Paragraph style={{ margin: 0 }}>
              模板中心负责把开源主链的复杂配置收敛成几个固定场景，作为 Local Pro v1 的医生模式入口。
            </Paragraph>
          </Space>
        </Card>

        <Alert
          type="info"
          showIcon
          message="第一期只做三个固定模板"
          description="模板的目标是减少配置自由度，提升本地专业版的可用性和可交付性，而不是覆盖所有研究变体。"
        />

        <Row gutter={[16, 16]}>
          {templates.map((template) => (
            <Col xs={24} xl={8} key={template.id}>
              <Card
                loading={loading}
                title={
                  <Space>
                    <ProfileOutlined />
                    <span>{template.name}</span>
                  </Space>
                }
                extra={<Tag color="blue">{TASK_LABELS[template.task_type]}</Tag>}
              >
                <Space direction="vertical" size={12} style={{ width: "100%" }}>
                  <Paragraph style={{ margin: 0 }}>{template.description}</Paragraph>

                  <div>
                    <Text strong>推荐配置</Text>
                    <div style={{ marginTop: 8 }}>
                      <Tag>{template.recommended_backbone || "-"}</Tag>
                      <Tag color="purple">{template.recommended_fusion || "-"}</Tag>
                    </div>
                  </div>

                  <div>
                    <Text strong>必填字段</Text>
                    <div style={{ marginTop: 8 }}>
                      {template.required_fields.map((field) => (
                        <Tag key={field}>{field}</Tag>
                      ))}
                    </div>
                  </div>

                  <div>
                    <Text strong>默认输出物</Text>
                    <div style={{ marginTop: 8 }}>
                      {template.expected_outputs.map((item) => (
                        <Tag color="green" key={item}>
                          {item}
                        </Tag>
                      ))}
                    </div>
                  </div>

                  {template.warnings.length ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="使用提醒"
                      description={
                        <ul style={{ paddingLeft: 18, marginBottom: 0 }}>
                          {template.warnings.map((warning) => (
                            <li key={warning}>{warning}</li>
                          ))}
                        </ul>
                      }
                    />
                  ) : null}

                  <Button
                    type="primary"
                    icon={<ArrowRightOutlined />}
                    onClick={() =>
                      navigate(`/projects?action=create&template=${template.id}`)
                    }
                  >
                    用这个模板创建项目
                  </Button>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </Space>
    </div>
  );
}
