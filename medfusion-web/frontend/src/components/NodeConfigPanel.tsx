import { Drawer, Form, Input, InputNumber, Select, Button, Space } from 'antd'
import { Node } from 'reactflow'

interface NodeConfigPanelProps {
  node: Node | null
  open: boolean
  onClose: () => void
  onSave: (nodeId: string, data: any) => void
}

const backboneOptions = [
  'resnet18', 'resnet34', 'resnet50', 'resnet101',
  'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
  'vit_b16', 'vit_b32',
  'swin_tiny', 'swin_small',
  'convnext_tiny', 'convnext_small',
]

export default function NodeConfigPanel({
  node,
  open,
  onClose,
  onSave,
}: NodeConfigPanelProps) {
  const [form] = Form.useForm()

  const handleSave = () => {
    form.validateFields().then((values) => {
      if (node) {
        onSave(node.id, values)
        onClose()
      }
    })
  }

  const renderFormFields = () => {
    if (!node) return null

    switch (node.type) {
      case 'dataLoader':
        return (
          <>
            <Form.Item
              label="数据路径"
              name="dataPath"
              rules={[{ required: true, message: '请输入数据路径' }]}
            >
              <Input placeholder="/path/to/data" />
            </Form.Item>
            <Form.Item label="批次大小" name="batchSize" initialValue={32}>
              <InputNumber min={1} max={512} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="工作进程数" name="numWorkers" initialValue={4}>
              <InputNumber min={0} max={16} style={{ width: '100%' }} />
            </Form.Item>
          </>
        )

      case 'model':
        return (
          <>
            <Form.Item
              label="Backbone"
              name="backbone"
              initialValue="resnet18"
              rules={[{ required: true }]}
            >
              <Select>
                {backboneOptions.map((opt) => (
                  <Select.Option key={opt} value={opt}>
                    {opt}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item
              label="类别数"
              name="numClasses"
              initialValue={2}
              rules={[{ required: true }]}
            >
              <InputNumber min={2} max={1000} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="预训练" name="pretrained" initialValue={true}>
              <Select>
                <Select.Option value={true}>是</Select.Option>
                <Select.Option value={false}>否</Select.Option>
              </Select>
            </Form.Item>
          </>
        )

      case 'training':
        return (
          <>
            <Form.Item
              label="训练轮数"
              name="epochs"
              initialValue={50}
              rules={[{ required: true }]}
            >
              <InputNumber min={1} max={1000} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item
              label="学习率"
              name="learningRate"
              initialValue={0.001}
              rules={[{ required: true }]}
            >
              <InputNumber
                min={0.00001}
                max={1}
                step={0.0001}
                style={{ width: '100%' }}
              />
            </Form.Item>
            <Form.Item label="优化器" name="optimizer" initialValue="adam">
              <Select>
                <Select.Option value="adam">Adam</Select.Option>
                <Select.Option value="sgd">SGD</Select.Option>
                <Select.Option value="adamw">AdamW</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="混合精度" name="useAmp" initialValue={true}>
              <Select>
                <Select.Option value={true}>启用</Select.Option>
                <Select.Option value={false}>禁用</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              label="梯度检查点"
              name="gradientCheckpointing"
              initialValue={false}
            >
              <Select>
                <Select.Option value={true}>启用</Select.Option>
                <Select.Option value={false}>禁用</Select.Option>
              </Select>
            </Form.Item>
          </>
        )

      case 'evaluation':
        return (
          <>
            <Form.Item
              label="评估指标"
              name="metrics"
              initialValue={['accuracy', 'loss']}
            >
              <Select mode="multiple">
                <Select.Option value="accuracy">准确率</Select.Option>
                <Select.Option value="loss">损失</Select.Option>
                <Select.Option value="precision">精确率</Select.Option>
                <Select.Option value="recall">召回率</Select.Option>
                <Select.Option value="f1">F1分数</Select.Option>
                <Select.Option value="auc">AUC</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="保存结果" name="saveResults" initialValue={true}>
              <Select>
                <Select.Option value={true}>是</Select.Option>
                <Select.Option value={false}>否</Select.Option>
              </Select>
            </Form.Item>
          </>
        )

      default:
        return null
    }
  }

  return (
    <Drawer
      title={`配置节点: ${node?.data?.label || ''}`}
      placement="right"
      width={400}
      open={open}
      onClose={onClose}
      extra={
        <Space>
          <Button onClick={onClose}>取消</Button>
          <Button type="primary" onClick={handleSave}>
            保存
          </Button>
        </Space>
      }
    >
      <Form form={form} layout="vertical" initialValues={node?.data}>
        {renderFormFields()}
      </Form>
    </Drawer>
  )
}
