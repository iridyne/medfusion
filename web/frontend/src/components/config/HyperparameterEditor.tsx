import { Card, Row, Col, InputNumber, Select, Switch, Space, Typography, Tooltip, Divider } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;
const { Option } = Select;

interface HyperparameterEditorProps {
  hyperparameters: {
    learning_rate: number;
    batch_size: number;
    epochs: number;
    optimizer: string;
    scheduler?: string;
    weight_decay: number;
  };
  training: {
    mixed_precision: boolean;
    gradient_accumulation_steps: number;
    early_stopping_patience: number;
  };
  onChange: (hyperparameters: any, training: any) => void;
}

export default function HyperparameterEditor({ hyperparameters, training, onChange }: HyperparameterEditorProps) {
  const handleHyperparameterChange = (key: string, value: any) => {
    onChange(
      { ...hyperparameters, [key]: value },
      training
    );
  };

  const handleTrainingChange = (key: string, value: any) => {
    onChange(
      hyperparameters,
      { ...training, [key]: value }
    );
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {/* 基础超参数 */}
      <Card title="基础超参数" size="small">
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>学习率</Text>
                <Tooltip title="初始学习率，影响训练速度和收敛性">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <InputNumber
                min={0.00001}
                max={0.1}
                step={0.0001}
                value={hyperparameters.learning_rate}
                onChange={(val) => handleHyperparameterChange('learning_rate', val)}
                style={{ width: '200px' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                推荐: 0.001 (Adam) 或 0.01 (SGD)
              </Text>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>Batch Size</Text>
                <Tooltip title="每批次样本数量，影响训练速度和内存占用">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <InputNumber
                min={1}
                max={256}
                step={1}
                value={hyperparameters.batch_size}
                onChange={(val) => handleHyperparameterChange('batch_size', val)}
                style={{ width: '200px' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                推荐: 16, 32, 64
              </Text>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>训练轮数</Text>
                <Tooltip title="完整遍历数据集的次数">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <InputNumber
                min={1}
                max={1000}
                step={10}
                value={hyperparameters.epochs}
                onChange={(val) => handleHyperparameterChange('epochs', val)}
                style={{ width: '200px' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                推荐: 100
              </Text>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>优化器</Text>
                <Tooltip title="梯度下降优化算法">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <Select
                value={hyperparameters.optimizer}
                onChange={(val) => handleHyperparameterChange('optimizer', val)}
                style={{ width: '200px' }}
              >
                <Option value="adam">Adam (推荐)</Option>
                <Option value="adamw">AdamW</Option>
                <Option value="sgd">SGD</Option>
                <Option value="rmsprop">RMSprop</Option>
              </Select>
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                Adam 适合大多数场景
              </Text>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>学习率调度器</Text>
                <Tooltip title="动态调整学习率的策略">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <Select
                value={hyperparameters.scheduler || 'none'}
                onChange={(val) => handleHyperparameterChange('scheduler', val === 'none' ? undefined : val)}
                style={{ width: '200px' }}
              >
                <Option value="none">不使用</Option>
                <Option value="cosine">Cosine Annealing</Option>
                <Option value="step">Step LR</Option>
                <Option value="plateau">Reduce on Plateau</Option>
              </Select>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>权重衰减</Text>
                <Tooltip title="L2 正则化系数，防止过拟合">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <InputNumber
                min={0}
                max={0.01}
                step={0.0001}
                value={hyperparameters.weight_decay}
                onChange={(val) => handleHyperparameterChange('weight_decay', val)}
                style={{ width: '200px' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                推荐: 0.0001
              </Text>
            </Col>
          </Row>
        </Space>
      </Card>

      <Divider />

      {/* 训练配置 */}
      <Card title="训练配置" size="small">
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>混合精度训练</Text>
                <Tooltip title="使用 FP16 加速训练，减少显存占用">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <Switch
                checked={training.mixed_precision}
                onChange={(val) => handleTrainingChange('mixed_precision', val)}
                checkedChildren="开启"
                unCheckedChildren="关闭"
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                推荐开启（需要 GPU 支持）
              </Text>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>梯度累积步数</Text>
                <Tooltip title="累积多个 batch 的梯度后再更新，等效于增大 batch size">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <InputNumber
                min={1}
                max={32}
                step={1}
                value={training.gradient_accumulation_steps}
                onChange={(val) => handleTrainingChange('gradient_accumulation_steps', val)}
                style={{ width: '200px' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                等效 batch size: {hyperparameters.batch_size * training.gradient_accumulation_steps}
              </Text>
            </Col>
          </Row>

          <Row align="middle">
            <Col span={8}>
              <Space>
                <Text>早停耐心值</Text>
                <Tooltip title="验证集性能不提升时，等待的轮数">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={16}>
              <InputNumber
                min={1}
                max={50}
                step={1}
                value={training.early_stopping_patience}
                onChange={(val) => handleTrainingChange('early_stopping_patience', val)}
                style={{ width: '200px' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                推荐: 10
              </Text>
            </Col>
          </Row>
        </Space>
      </Card>

      {/* 配置总结 */}
      <Card size="small" style={{ backgroundColor: '#f0f5ff' }}>
        <Space direction="vertical" size="small">
          <Text strong>配置总结</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            • 优化器: {hyperparameters.optimizer.toUpperCase()} | 学习率: {hyperparameters.learning_rate}
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            • Batch Size: {hyperparameters.batch_size} × {training.gradient_accumulation_steps} = {hyperparameters.batch_size * training.gradient_accumulation_steps} (等效)
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            • 训练轮数: {hyperparameters.epochs} | 早停: {training.early_stopping_patience} epochs
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            • 混合精度: {training.mixed_precision ? '✅ 开启' : '❌ 关闭'}
          </Text>
        </Space>
      </Card>
    </Space>
  );
}
