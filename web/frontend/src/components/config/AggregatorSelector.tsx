import { Card, Radio, Space, InputNumber, Row, Col, Typography, Tag, Tooltip, Alert } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

const { Text, Paragraph } = Typography;

interface AggregatorSelectorProps {
  value: {
    type: string;
    params?: Record<string, any>;
  };
  onChange: (value: any) => void;
}

interface AggregatorInfo {
  name: string;
  key: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  recommended: boolean;
  params: string[];
  useCase: string;
  learnable: boolean;
}

const aggregators: AggregatorInfo[] = [
  {
    name: 'Mean Pooling',
    key: 'mean',
    description: '对多视图特征取平均，简单高效',
    complexity: 'low',
    recommended: true,
    params: [],
    useCase: '适合视图重要性相近的场景',
    learnable: false,
  },
  {
    name: 'Max Pooling',
    key: 'max',
    description: '取多视图特征的最大值，保留显著特征',
    complexity: 'low',
    recommended: false,
    params: [],
    useCase: '适合需要突出关键特征的场景',
    learnable: false,
  },
  {
    name: 'Attention Aggregator',
    key: 'attention',
    description: '使用注意力机制自动学习视图权重',
    complexity: 'medium',
    recommended: true,
    params: ['hidden_dim'],
    useCase: '适合视图重要性不均的场景',
    learnable: true,
  },
  {
    name: 'Cross-View Attention',
    key: 'cross_view',
    description: '跨视图注意力，建模视图间的交互',
    complexity: 'high',
    recommended: false,
    params: ['hidden_dim', 'num_heads'],
    useCase: '适合视图间存在复杂关系的场景',
    learnable: true,
  },
  {
    name: 'Learned Weight',
    key: 'learned_weight',
    description: '为每个视图学习独立的权重参数',
    complexity: 'low',
    recommended: false,
    params: [],
    useCase: '适合视图数量固定的场景',
    learnable: true,
  },
];

export default function AggregatorSelector({ value, onChange }: AggregatorSelectorProps) {
  const selectedAggregator = aggregators.find(a => a.key === value.type);

  const handleAggregatorChange = (aggregatorKey: string) => {
    const aggregator = aggregators.find(a => a.key === aggregatorKey);
    const newValue: any = { type: aggregatorKey };

    // 根据聚合器设置默认参数
    if (aggregator?.params.includes('hidden_dim')) {
      newValue.params = { ...value.params, hidden_dim: value.params?.hidden_dim || 256 };
    }
    if (aggregator?.params.includes('num_heads')) {
      newValue.params = { ...value.params, num_heads: value.params?.num_heads || 4 };
    }

    onChange(newValue);
  };

  const handleParamChange = (paramName: string, paramValue: number | null) => {
    if (paramValue) {
      onChange({
        ...value,
        params: {
          ...value.params,
          [paramName]: paramValue,
        },
      });
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
      {/* 聚合器选择 */}
      <Radio.Group
        value={value.type}
        onChange={(e) => handleAggregatorChange(e.target.value)}
        style={{ width: '100%' }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          {aggregators.map((aggregator) => (
            <Card
              key={aggregator.key}
              size="small"
              hoverable
              style={{
                border: value.type === aggregator.key ? '2px solid #1890ff' : '1px solid #d9d9d9',
              }}
            >
              <Radio value={aggregator.key} style={{ width: '100%' }}>
                <Row gutter={16} align="middle">
                  <Col span={6}>
                    <Space>
                      <Text strong>{aggregator.name}</Text>
                      {aggregator.recommended && (
                        <Tag color="blue">推荐</Tag>
                      )}
                    </Space>
                  </Col>
                  <Col span={10}>
                    <Paragraph style={{ margin: 0, fontSize: '12px', color: '#666' }}>
                      {aggregator.description}
                    </Paragraph>
                    <Text type="secondary" style={{ fontSize: '11px' }}>
                      {aggregator.useCase}
                    </Text>
                  </Col>
                  <Col span={8}>
                    <Space size="small">
                      <Tooltip title="计算复杂度">
                        <Tag color={getComplexityColor(aggregator.complexity)}>
                          {aggregator.complexity === 'low' && '低复杂度'}
                          {aggregator.complexity === 'medium' && '中复杂度'}
                          {aggregator.complexity === 'high' && '高复杂度'}
                        </Tag>
                      </Tooltip>
                      {aggregator.learnable && (
                        <Tooltip title="包含可学习参数">
                          <Tag color="purple">可学习</Tag>
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
      {selectedAggregator && selectedAggregator.params.length > 0 && (
        <Card title="Aggregator 参数配置" size="small">
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {selectedAggregator.params.includes('hidden_dim') && (
              <Row align="middle">
                <Col span={8}>
                  <Space>
                    <Text>隐藏层维度</Text>
                    <Tooltip title="注意力层的隐藏维度">
                      <InfoCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                </Col>
                <Col span={16}>
                  <InputNumber
                    min={64}
                    max={1024}
                    step={64}
                    value={value.params?.hidden_dim || 256}
                    onChange={(val) => handleParamChange('hidden_dim', val)}
                    style={{ width: '200px' }}
                  />
                  <Text type="secondary" style={{ marginLeft: '12px' }}>
                    推荐: 256
                  </Text>
                </Col>
              </Row>
            )}

            {selectedAggregator.params.includes('num_heads') && (
              <Row align="middle">
                <Col span={8}>
                  <Space>
                    <Text>注意力头数</Text>
                    <Tooltip title="多头注意力的头数">
                      <InfoCircleOutlined style={{ color: '#1890ff' }} />
                    </Tooltip>
                  </Space>
                </Col>
                <Col span={16}>
                  <InputNumber
                    min={1}
                    max={8}
                    value={value.params?.num_heads || 4}
                    onChange={(val) => handleParamChange('num_heads', val)}
                    style={{ width: '200px' }}
                  />
                  <Text type="secondary" style={{ marginLeft: '12px' }}>
                    推荐: 4
                  </Text>
                </Col>
              </Row>
            )}

            {/* 参数验证 */}
            {value.params?.hidden_dim && value.params?.num_heads &&
             value.params.hidden_dim % value.params.num_heads !== 0 && (
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

      {/* 聚合器说明 */}
      <Card size="small" style={{ backgroundColor: '#f0f5ff' }}>
        <Space direction="vertical" size="small">
          <Space>
            <InfoCircleOutlined style={{ color: '#1890ff' }} />
            <Text strong>当前聚合器: {selectedAggregator?.name}</Text>
            {selectedAggregator?.learnable && (
              <Tag color="purple">可学习</Tag>
            )}
          </Space>
          <Paragraph style={{ margin: 0, fontSize: '12px', color: '#666' }}>
            {selectedAggregator?.description}
          </Paragraph>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            💡 {selectedAggregator?.useCase}
          </Text>
        </Space>
      </Card>
    </Space>
  );
}
