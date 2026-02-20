import {
  Drawer,
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Space,
  Switch,
} from "antd";
import { Node } from "reactflow";
import { useState, useEffect } from "react";

interface NodeConfigPanelProps {
  node: Node | null;
  open: boolean;
  onClose: () => void;
  onSave: (nodeId: string, data: any) => void;
}

const backboneOptions = [
  "resnet18",
  "resnet34",
  "resnet50",
  "resnet101",
  "efficientnet_b0",
  "efficientnet_b1",
  "efficientnet_b2",
  "vit_b16",
  "vit_b32",
  "swin_tiny",
  "swin_small",
  "convnext_tiny",
  "convnext_small",
];

const fusionOptions = [
  { value: "concatenate", label: "Concatenate（拼接）", complexity: "低" },
  { value: "gated", label: "Gated Fusion（门控融合）", complexity: "中" },
  {
    value: "attention",
    label: "Attention Fusion（注意力融合）",
    complexity: "高",
  },
  {
    value: "cross_attention",
    label: "Cross Attention（跨模态注意力）",
    complexity: "高",
  },
  {
    value: "bilinear",
    label: "Bilinear Pooling（双线性池化）",
    complexity: "高",
  },
];

const aggregatorOptions = [
  { value: "mean", label: "Mean Pooling（平均池化）", learnable: false },
  { value: "max", label: "Max Pooling（最大池化）", learnable: false },
  {
    value: "attention",
    label: "Attention Aggregator（注意力聚合）",
    learnable: true,
  },
  {
    value: "cross_view",
    label: "Cross-View Aggregator（跨视图聚合）",
    learnable: true,
  },
  {
    value: "learned_weight",
    label: "Learned Weight（可学习权重）",
    learnable: true,
  },
];

export default function NodeConfigPanel({
  node,
  open,
  onClose,
  onSave,
}: NodeConfigPanelProps) {
  const [form] = Form.useForm();

  const handleSave = () => {
    form.validateFields().then((values) => {
      if (node) {
        onSave(node.id, values);
        onClose();
      }
    });
  };

  // 模拟数据集列表（后续可改为 API 调用）
  const [datasets] = useState([
    { id: "1", name: "Chest X-Ray Dataset", samples: 5000, type: "image" },
    { id: "2", name: "Clinical Records", samples: 10000, type: "tabular" },
    {
      id: "3",
      name: "Multimodal Cancer Dataset",
      samples: 3000,
      type: "multimodal",
    },
  ]);

  const renderFormFields = () => {
    if (!node) return null;

    switch (node.type) {
      case "dataLoader":
        return (
          <>
            <Form.Item
              label="选择数据集"
              name="datasetId"
              rules={[{ required: true, message: "请选择数据集" }]}
            >
              <Select
                placeholder="选择已上传的数据集"
                onChange={(value) => {
                  const dataset = datasets.find((d) => d.id === value);
                  if (dataset) {
                    form.setFieldsValue({ datasetName: dataset.name });
                  }
                }}
              >
                {datasets.map((ds) => (
                  <Select.Option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.samples} 样本, {ds.type})
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item name="datasetName" hidden>
              <Input />
            </Form.Item>
            <Form.Item
              label="数据划分"
              name="split"
              initialValue="train"
              rules={[{ required: true }]}
            >
              <Select>
                <Select.Option value="train">训练集 (Train)</Select.Option>
                <Select.Option value="val">验证集 (Validation)</Select.Option>
                <Select.Option value="test">测试集 (Test)</Select.Option>
                <Select.Option value="all">全部数据 (All)</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="批次大小" name="batchSize" initialValue={32}>
              <InputNumber min={1} max={512} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item
              label="打乱数据"
              name="shuffle"
              initialValue={true}
              valuePropName="checked"
            >
              <Switch checkedChildren="是" unCheckedChildren="否" />
            </Form.Item>
            <Form.Item label="随机种子" name="seed" initialValue={42}>
              <InputNumber min={0} max={99999} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item label="工作进程数" name="numWorkers" initialValue={4}>
              <InputNumber min={0} max={16} style={{ width: "100%" }} />
            </Form.Item>
          </>
        );

      case "model":
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
              <InputNumber min={2} max={1000} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item label="预训练" name="pretrained" initialValue={true}>
              <Select>
                <Select.Option value={true}>是</Select.Option>
                <Select.Option value={false}>否</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item
              label="Fusion 策略"
              name="fusion"
              initialValue="concatenate"
            >
              <Select>
                {fusionOptions.map((opt) => (
                  <Select.Option key={opt.value} value={opt.value}>
                    {opt.label}{" "}
                    <span style={{ color: "#999" }}>({opt.complexity})</span>
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item
              label="Aggregator（多视图）"
              name="aggregator"
              initialValue="mean"
            >
              <Select>
                {aggregatorOptions.map((opt) => (
                  <Select.Option key={opt.value} value={opt.value}>
                    {opt.label}{" "}
                    {opt.learnable && (
                      <span style={{ color: "#52c41a" }}>✓ 可学习</span>
                    )}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item label="隐藏层维度" name="hiddenDim" initialValue={512}>
              <InputNumber
                min={64}
                max={2048}
                step={64}
                style={{ width: "100%" }}
              />
            </Form.Item>
          </>
        );

      case "training":
        return (
          <>
            <Form.Item
              label="训练轮数"
              name="epochs"
              initialValue={50}
              rules={[{ required: true }]}
            >
              <InputNumber min={1} max={1000} style={{ width: "100%" }} />
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
                style={{ width: "100%" }}
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
        );

      case "evaluation":
        return (
          <>
            <Form.Item
              label="评估指标"
              name="metrics"
              initialValue={["accuracy", "loss"]}
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
        );

      default:
        return null;
    }
  };

  return (
    <Drawer
      title={`配置节点: ${node?.data?.label || ""}`}
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
  );
}
