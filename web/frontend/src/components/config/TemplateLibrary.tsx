import { Card, Row, Col, Button, Space, Typography, Tag, Tooltip } from 'antd';
import { ThunderboltOutlined, ExperimentOutlined, RocketOutlined, SafetyOutlined } from '@ant-design/icons';

const { Text, Paragraph } = Typography;

interface Template {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  tags: string[];
  config: any;
  recommended: boolean;
}

interface TemplateLibraryProps {
  onLoad: (config: any) => void;
}

const templates: Template[] = [
  {
    id: 'quick_start',
    name: '快速开始',
    description: 'ResNet50 + Concatenate + Mean，适合快速实验和基线模型',
    icon: <ThunderboltOutlined style={{ fontSize: '24px', color: '#1890ff' }} />,
    tags: ['推荐', '快速', '基线'],
    recommended: true,
    config: {
      backbone: {
        type: 'resnet50',
        pretrained: true,
        freeze_layers: 0,
      },
      fusion: {
        strategy: 'concatenate',
      },
      aggregator: {
        type: 'mean',
      },
      hyperparameters: {
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 100,
        optimizer: 'adam',
        weight_decay: 0.0001,
      },
      training: {
        mixed_precision: true,
        gradient_accumulation_steps: 1,
        early_stopping_patience: 10,
      },
    },
  },
  {
    id: 'high_accuracy',
    name: '高精度模型',
    description: 'EfficientNetV2-S + Gated Fusion + Attention，追求最高精度',
    icon: <RocketOutlined style={{ fontSize: '24px', color: '#52c41a' }} />,
    tags: ['高精度', '推荐'],
    recommended: true,
    config: {
      backbone: {
        type: 'efficientnet_v2_s',
        pretrained: true,
        freeze_layers: 0,
      },
      fusion: {
        strategy: 'gated',
        hidden_dim: 512,
      },
      aggregator: {
        type: 'attention',
        params: {
          hidden_dim: 256,
        },
      },
      hyperparameters: {
        learning_rate: 0.0005,
        batch_size: 16,
        epochs: 150,
        optimizer: 'adamw',
        scheduler: 'cosine',
        weight_decay: 0.0001,
      },
      training: {
        mixed_precision: true,
        gradient_accumulation_steps: 2,
        early_stopping_patience: 15,
      },
    },
  },
  {
    id: 'lightweight',
    name: '轻量级模型',
    description: 'MobileNetV3 + Concatenate + Mean，适合资源受限场景',
    icon: <SafetyOutlined style={{ fontSize: '24px', color: '#faad14' }} />,
    tags: ['轻量', '快速'],
    recommended: false,
    config: {
      backbone: {
        type: 'mobilenet_v3_large',
        pretrained: true,
        freeze_layers: 0,
      },
      fusion: {
        strategy: 'concatenate',
      },
      aggregator: {
        type: 'mean',
      },
      hyperparameters: {
        learning_rate: 0.001,
        batch_size: 64,
        epochs: 100,
        optimizer: 'adam',
        weight_decay: 0.0001,
      },
      training: {
        mixed_precision: true,
        gradient_accumulation_steps: 1,
        early_stopping_patience: 10,
      },
    },
  },
  {
    id: 'advanced',
    name: '高级配置',
    description: 'Swin Transformer + Cross Attention + Cross-View，适合复杂场景',
    icon: <ExperimentOutlined style={{ fontSize: '24px', color: '#722ed1' }} />,
    tags: ['高级', 'Transformer'],
    recommended: false,
    config: {
      backbone: {
        type: 'swin_t',
        pretrained: true,
        freeze_layers: 0,
      },
      fusion: {
        strategy: 'cross_attention',
        hidden_dim: 512,
        num_heads: 8,
      },
      aggregator: {
        type: 'cross_view',
        params: {
          hidden_dim: 256,
          num_heads: 4,
        },
      },
      hyperparameters: {
        learning_rate: 0.0001,
        batch_size: 16,
        epochs: 200,
        optimizer: 'adamw',
        scheduler: 'cosine',
        weight_decay: 0.0001,
      },
      training: {
        mixed_precision: true,
        gradient_accumulation_steps: 4,
        early_stopping_patience: 20,
      },
    },
  },
];

export default function TemplateLibrary({ onLoad }: TemplateLibraryProps) {
  return (
    <Row gutter={[16, 16]}>
      {templates.map((template) => (
        <Col key={template.id} xs={24} sm={12} md={12} lg={6}>
          <Card
            hoverable
            style={{
              height: '100%',
              border: template.recommended ? '2px solid #1890ff' : '1px solid #d9d9d9',
            }}
          >
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                {template.icon}
              </div>

              <div>
                <Space>
                  <Text strong>{template.name}</Text>
                  {template.recommended && (
                    <Tag color="blue">推荐</Tag>
                  )}
                </Space>
              </div>

              <Paragraph
                style={{
                  margin: 0,
                  fontSize: '12px',
                  color: '#666',
                  minHeight: '40px',
                }}
              >
                {template.description}
              </Paragraph>

              <Space size={[0, 8]} wrap>
                {template.tags.map((tag) => (
                  <Tag key={tag} color={tag === '推荐' ? 'blue' : 'default'}>
                    {tag}
                  </Tag>
                ))}
              </Space>

              <Button
                type={template.recommended ? 'primary' : 'default'}
                block
                onClick={() => onLoad(template.config)}
              >
                使用此模板
              </Button>
            </Space>
          </Card>
        </Col>
      ))}
    </Row>
  );
}
