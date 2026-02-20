import { useState } from 'react';
import { Card, Tabs, Radio, Space, Switch, Slider, Input, Tag, Tooltip, Row, Col, Typography } from 'antd';
import { SearchOutlined, InfoCircleOutlined } from '@ant-design/icons';

const { TabPane } = Tabs;
const { Search } = Input;
const { Text, Paragraph } = Typography;

interface BackboneInfo {
  name: string;
  params: string;
  flops: string;
  accuracy: string;
  description: string;
  recommended: boolean;
}

interface BackboneSelectorProps {
  value: {
    type: string;
    pretrained: boolean;
    freeze_layers: number;
  };
  onChange: (value: any) => void;
}

const backboneData: Record<string, BackboneInfo[]> = {
  'ResNet': [
    { name: 'resnet18', params: '11.7M', flops: '1.8G', accuracy: '69.8%', description: '轻量级，适合快速实验', recommended: false },
    { name: 'resnet34', params: '21.8M', flops: '3.7G', accuracy: '73.3%', description: '平衡性能和速度', recommended: false },
    { name: 'resnet50', params: '25.6M', flops: '4.1G', accuracy: '76.1%', description: '经典选择，广泛使用', recommended: true },
    { name: 'resnet101', params: '44.5M', flops: '7.8G', accuracy: '77.4%', description: '更深的网络，更高精度', recommended: false },
    { name: 'resnet152', params: '60.2M', flops: '11.6G', accuracy: '78.3%', description: '最深的 ResNet', recommended: false },
  ],
  'MobileNet': [
    { name: 'mobilenet_v2', params: '3.5M', flops: '0.3G', accuracy: '71.9%', description: '移动端优化，极致轻量', recommended: false },
    { name: 'mobilenet_v3_small', params: '2.5M', flops: '0.06G', accuracy: '67.7%', description: '最小模型，边缘设备', recommended: false },
    { name: 'mobilenet_v3_large', params: '5.4M', flops: '0.22G', accuracy: '75.0%', description: '移动端高精度', recommended: false },
  ],
  'EfficientNet': [
    { name: 'efficientnet_b0', params: '5.3M', flops: '0.4G', accuracy: '77.3%', description: '基础版本，效率优先', recommended: true },
    { name: 'efficientnet_b1', params: '7.8M', flops: '0.7G', accuracy: '79.1%', description: '轻微增强', recommended: false },
    { name: 'efficientnet_b2', params: '9.2M', flops: '1.0G', accuracy: '80.1%', description: '平衡选择', recommended: false },
    { name: 'efficientnet_b3', params: '12.0M', flops: '1.8G', accuracy: '81.6%', description: '中等规模', recommended: false },
    { name: 'efficientnet_b4', params: '19.0M', flops: '4.2G', accuracy: '82.9%', description: '高精度', recommended: false },
    { name: 'efficientnet_b5', params: '30.0M', flops: '9.9G', accuracy: '83.6%', description: '更高精度', recommended: false },
    { name: 'efficientnet_b6', params: '43.0M', flops: '19.0G', accuracy: '84.0%', description: '接近最优', recommended: false },
    { name: 'efficientnet_b7', params: '66.0M', flops: '37.0G', accuracy: '84.3%', description: '最高精度', recommended: false },
  ],
  'EfficientNetV2': [
    { name: 'efficientnet_v2_s', params: '21.5M', flops: '8.4G', accuracy: '84.2%', description: 'V2 小型版本', recommended: true },
    { name: 'efficientnet_v2_m', params: '54.1M', flops: '24.7G', accuracy: '85.1%', description: 'V2 中型版本', recommended: false },
    { name: 'efficientnet_v2_l', params: '118.5M', flops: '56.3G', accuracy: '85.7%', description: 'V2 大型版本', recommended: false },
  ],
  'ConvNeXt': [
    { name: 'convnext_tiny', params: '28.6M', flops: '4.5G', accuracy: '82.1%', description: '现代卷积网络，小型', recommended: true },
    { name: 'convnext_small', params: '50.2M', flops: '8.7G', accuracy: '83.1%', description: '现代卷积网络，中型', recommended: false },
    { name: 'convnext_base', params: '88.6M', flops: '15.4G', accuracy: '83.8%', description: '现代卷积网络，基础', recommended: false },
    { name: 'convnext_large', params: '197.8M', flops: '34.4G', accuracy: '84.3%', description: '现代卷积网络，大型', recommended: false },
  ],
  'Vision Transformer': [
    { name: 'vit_b_16', params: '86.6M', flops: '17.6G', accuracy: '81.1%', description: 'ViT Base，16x16 patch', recommended: true },
    { name: 'vit_b_32', params: '88.2M', flops: '4.4G', accuracy: '75.9%', description: 'ViT Base，32x32 patch', recommended: false },
    { name: 'vit_l_16', params: '304.3M', flops: '61.6G', accuracy: '82.6%', description: 'ViT Large，高精度', recommended: false },
    { name: 'vit_l_32', params: '306.5M', flops: '15.4G', accuracy: '76.9%', description: 'ViT Large，32x32 patch', recommended: false },
  ],
  'Swin Transformer': [
    { name: 'swin_t', params: '28.3M', flops: '4.5G', accuracy: '81.3%', description: 'Swin Tiny，层次化 ViT', recommended: true },
    { name: 'swin_s', params: '49.6M', flops: '8.7G', accuracy: '83.0%', description: 'Swin Small', recommended: false },
    { name: 'swin_b', params: '87.8M', flops: '15.4G', accuracy: '83.5%', description: 'Swin Base', recommended: false },
  ],
  'RegNet': [
    { name: 'regnet_y_400mf', params: '4.3M', flops: '0.4G', accuracy: '74.0%', description: '极轻量级', recommended: false },
    { name: 'regnet_y_800mf', params: '6.3M', flops: '0.8G', accuracy: '76.4%', description: '轻量级', recommended: false },
    { name: 'regnet_y_1_6gf', params: '11.2M', flops: '1.6G', accuracy: '78.0%', description: '中等规模', recommended: true },
    { name: 'regnet_y_3_2gf', params: '19.4M', flops: '3.2G', accuracy: '79.4%', description: '较大规模', recommended: false },
    { name: 'regnet_y_8gf', params: '39.2M', flops: '8.0G', accuracy: '80.0%', description: '大规模', recommended: false },
    { name: 'regnet_y_16gf', params: '83.6M', flops: '16.0G', accuracy: '80.4%', description: '超大规模', recommended: false },
    { name: 'regnet_y_32gf', params: '145.0M', flops: '32.0G', accuracy: '80.9%', description: '最大规模', recommended: false },
  ],
};

export default function BackboneSelector({ value, onChange }: BackboneSelectorProps) {
  const [searchText, setSearchText] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('ResNet');

  const handleBackboneChange = (backboneName: string) => {
    onChange({
      ...value,
      type: backboneName,
    });
  };

  const handlePretrainedChange = (checked: boolean) => {
    onChange({
      ...value,
      pretrained: checked,
    });
  };

  const handleFreezeLayersChange = (layers: number) => {
    onChange({
      ...value,
      freeze_layers: layers,
    });
  };

  const filterBackbones = (backbones: BackboneInfo[]) => {
    if (!searchText) return backbones;
    return backbones.filter(b =>
      b.name.toLowerCase().includes(searchText.toLowerCase()) ||
      b.description.toLowerCase().includes(searchText.toLowerCase())
    );
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {/* 搜索框 */}
      <Search
        placeholder="搜索 Backbone 模型..."
        prefix={<SearchOutlined />}
        value={searchText}
        onChange={(e) => setSearchText(e.target.value)}
        allowClear
      />

      {/* 分类标签页 */}
      <Tabs activeKey={selectedCategory} onChange={setSelectedCategory}>
        {Object.entries(backboneData).map(([category, backbones]) => (
          <TabPane tab={`${category} (${backbones.length})`} key={category}>
            <Radio.Group
              value={value.type}
              onChange={(e) => handleBackboneChange(e.target.value)}
              style={{ width: '100%' }}
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                {filterBackbones(backbones).map((backbone) => (
                  <Card
                    key={backbone.name}
                    size="small"
                    hoverable
                    style={{
                      border: value.type === backbone.name ? '2px solid #1890ff' : '1px solid #d9d9d9',
                    }}
                  >
                    <Radio value={backbone.name} style={{ width: '100%' }}>
                      <Row gutter={16} align="middle">
                        <Col span={6}>
                          <Space>
                            <Text strong>{backbone.name}</Text>
                            {backbone.recommended && (
                              <Tag color="blue">推荐</Tag>
                            )}
                          </Space>
                        </Col>
                        <Col span={12}>
                          <Paragraph style={{ margin: 0, fontSize: '12px', color: '#666' }}>
                            {backbone.description}
                          </Paragraph>
                        </Col>
                        <Col span={6}>
                          <Space size="small">
                            <Tooltip title="参数量">
                              <Tag>{backbone.params}</Tag>
                            </Tooltip>
                            <Tooltip title="计算量">
                              <Tag>{backbone.flops}</Tag>
                            </Tooltip>
                            <Tooltip title="ImageNet 准确率">
                              <Tag color="green">{backbone.accuracy}</Tag>
                            </Tooltip>
                          </Space>
                        </Col>
                      </Row>
                    </Radio>
                  </Card>
                ))}
              </Space>
            </Radio.Group>
          </TabPane>
        ))}
      </Tabs>

      {/* 配置选项 */}
      <Card title="Backbone 配置" size="small">
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          {/* 预训练权重 */}
          <Row align="middle">
            <Col span={6}>
              <Space>
                <Text>使用预训练权重</Text>
                <Tooltip title="使用 ImageNet 预训练权重可以加速收敛并提高性能">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={18}>
              <Switch
                checked={value.pretrained}
                onChange={handlePretrainedChange}
                checkedChildren="开启"
                unCheckedChildren="关闭"
              />
              {value.pretrained && (
                <Text type="secondary" style={{ marginLeft: '12px' }}>
                  将加载 ImageNet 预训练权重
                </Text>
              )}
            </Col>
          </Row>

          {/* 冻结层数 */}
          <Row align="middle">
            <Col span={6}>
              <Space>
                <Text>冻结前 N 层</Text>
                <Tooltip title="冻结前面的层可以防止过拟合，适合小数据集">
                  <InfoCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
              </Space>
            </Col>
            <Col span={18}>
              <Slider
                min={0}
                max={10}
                value={value.freeze_layers}
                onChange={handleFreezeLayersChange}
                marks={{
                  0: '不冻结',
                  5: '5层',
                  10: '10层',
                }}
                style={{ width: '80%' }}
              />
              <Text type="secondary" style={{ marginLeft: '12px' }}>
                当前: {value.freeze_layers} 层
              </Text>
            </Col>
          </Row>
        </Space>
      </Card>

      {/* 选择总结 */}
      <Card size="small" style={{ backgroundColor: '#f0f5ff' }}>
        <Space>
          <InfoCircleOutlined style={{ color: '#1890ff' }} />
          <Text>
            已选择: <Text strong>{value.type}</Text>
            {value.pretrained && <Tag color="blue" style={{ marginLeft: '8px' }}>预训练</Tag>}
            {value.freeze_layers > 0 && <Tag color="orange">冻结 {value.freeze_layers} 层</Tag>}
          </Text>
        </Space>
      </Card>
    </Space>
  );
}
