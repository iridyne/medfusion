import { Card, Radio, Space, InputNumber, Row, Col, Typography, Tag, Tooltip, Alert } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

const { Text, Paragraph } = Typography;

interface FusionSelectorProps {
  value: {
    strategy: string;
    hidden_dim?: number;
    num_heads?: number;
  };
  onChange: (value: any) => void;
}

interface FusionStrategy {
  name: string;
  key: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  recommended: boolean;
  params: string[];
  useCase: string;
}

const fusionStrategies: FusionStrategy[] = [
  {
    name: 'Concatenate',
    key: 'concatenate',
    description: '简单拼接多模态特征，计算效率高',
    complexity: 'low',
    recommended: true,
    params: [],
    useCase: '适合快速实验和基线模型',
  },
  {
    name: 'Gated Fusion',
    key: 'gated',
    description: '使用门控机制动态调整模态权重',
    complexity: 'medium',
    recommended: true,
    params: ['hidden_dim'],
    useCase: '适合模态重要性不均衡的场景',
  },
  {
    name: 'Attention Fusion',
    key: 'attention',
    description: '使用自注意力机制融合多模态特征',
    complexity: 'medium',
    recommended: false,
    params: ['hidden_dim', 'num_heads'],
    useCase: '适合需要捕捉模态间复杂关系的场景',
  },
  {
    name: 'Cross Attention',
    key: 'cross_attention',
    description: '跨模态注意力，建模模态间的交互',
    complexity: 'high',
    recommended: false,
    params: ['hidden_dim', 'num_heads'],
    useCase: '适合模态间存在强交互的场景',
  },
  {
    name: 'Bilinear Pooling',
    key: 'bilinear',
    description: '双线性池化，捕捉二阶特征交互',
    complexity: 'high',
    recommended: false,
    params: ['hidden_dim'],
    useCase: '适合细粒度特征融合',
  },
];

export default function FusionSelector({ value, onChange }: FusionSelectorProps) {
  const selectedStrategy = fusionStrategies.find(s => s.key === value.strategy);

  const handleStrategyChange = (strategyKey: string) => {
    const strategy = fusionStrategies.find(s => s.key === strategyKey);
    const newValue: any = { strategy: strategyKey };

    // 根据策略设置默认参数
    if (strategy?.params.includes('hidden_dim')) {
      newValue.hidden_dim = value.hidden_dim || 256;
    }
    if (strategy?.params.includes('num_heads')) {
      newValue.num_heads = value.num_heads || 8;
    }

    onChange(newValue);
  };

  const handleHiddenDimChange = (dim: number | null) => {
    if (dim) {
      onChange({ ...value, hidden_dim: dim });
    }
  };

  const handleNumHeadsChange = (heads: number | null) => {
    if (heads) {
      onChange({ ...value, num_heads: heads });
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return 'green';
      case 'medium': return 'orange';
      case 'high': return 'red';
      default: return 'default';
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {/* 策略选择 */}
      <Radio.Group
        value={value.strategy}
        onChange={(e) => handleStrategyChange(e.target.value)}
        style={{ width: '100%' }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          {fusionStrategies.map((strategy) => (
            <Card
              key={strategy.key}
              size="small"
              hoverable
              style={{
                border: value.strategy === strategy.key ? '2px solid #1890ff' : '1px solid #d9d9d9',
              }}
            >
              <Radio value={strategy.key} style={{ width: '100%' }}>
                <Row gutter={16} align="middle">
                  <Col span={6}>
                    <Space>
                      <Text strong>{strategy.name}</Text>
                      {strategy.recommended && (
                        <Tag color="blue">推荐</Tag>
                      )}
                    </Space>
                  </Col>
                  <Col span={10}>
                    <Paragraph style={{ margin: 0, fontSize: '12px', color: '#666' }}>
                      {strategy.description}
                    </Paragraph>
                    <Text type="secondary" style={{ fontSize: '11px' }}>
                      {strategy.useCase}
                    </Text>
                  </Col>
                  <Col span={8}>
                    <Space size="small">
                      <Tooltip title="计算复杂度">
                        <Tag color={getComplexityColor(strategy.complexity)}>
                          {strategy.complexity === 'low' && '低复杂度'}
                          {strategy.complexity === 'medium' && '中复杂度'}
                          {strategy.complexity === 'high' && '高复杂度'}
                        </Tag>
                      </Tooltip>
                      {strategy.params.length > 0 && (
                        <Tooltip title="需要配置参数">
                          <Tag>需配置</Tag>
                        </Tooltip>
                      )}
                    </Space>
                  </Col>
                </Row>
              </Radio>
            </Card>
          ))}
        </Space>
      </Radio.Group>

      {/* 参数配置 */}
      {selectedStrategy && selectedStrategy.params.length > 0 && (
        <Card title="Fusion 参数配置" size="small">
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {selectedStrategy.params.includes('hidden_dim') && (
              <Row align="middle">
                <Col span={8}>
                  <Space>
                    <Text>隐藏层维度</Text>
                    <Tooltip title="融合层的隐藏维度，影响模型容量">
                      <InfoCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                </Col>
                <Col span={16}>
                  <InputNumber
                    min={64}
                    max={2048}
                    step={64}
                    value={value.hidden_dim || 256}
                    onChange={handleHiddenDimChange}
                    style={{ width: '200px' }}
                  />
                  <Text type="secondary" style={{ marginLeft: '12px' }}>
                    推荐: 256 或 512
                  </Text>
                </Col>
              </Row>
            )}

            {selectedStrategy.params.includes('num_heads') && (
              <Row align="middle">
                <Col span={8}>
                  <Space>
                    <Text>注意力头数</Text>
                    <Tooltip title="多头注意力的头数，必须能整除隐藏维度">
                      <InfoCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                </Col>
                <Col span={16}>
                  <InputNumber
                    min={1}
                    max={16}
                    value={value.num_heads || 8}
                    onChange={handleNumHeadsChange}
                    style={{ width: '200px' }}
                  />
                  <Text type="secondary" style={{ marginLeft: '12px' }}>
                    推荐: 4 或 8
                  </Text>
                </Col>
              </Row>
            )}

            {/* 参数验证提示 */}
            {value.hidden_dim && value.num_heads && value.hidden_dim % value.num_heads !== 0 && (
              <Alert
                message="参数警告"
                description="隐藏层维度必须能被注意力头数整除"
                type="warning"
                showIcon
              />
            )}
          </Space>
        </Card>
      )}

      {/* 策略说明 */}
      <Card size="small" style={{ backgroundColor: '#f0f5ff' }}>
        <Space direction="vertical" size="small">
          <Space>
            <InfoCircleOutlined style={{ color: '#1890ff' }} />
            <Text strong>当前策略: {selectedStrategy?.name}</Text>
          </Space>
          <Paragraph style={{ margin: 0, fontSize: '12px', color: '#666' }}>
            {selectedStrategy?.description}
          </Paragraph>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            💡 {selectedStrategy?.useCase}
          </Text>
        </Space>
      </Card>
    </Space>
  );
}
