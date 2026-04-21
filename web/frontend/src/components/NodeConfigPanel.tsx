import { useEffect } from "react";
import {
  Button,
  Divider,
  Drawer,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
} from "antd";
import { Node } from "reactflow";

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
  "efficientnet_b0",
  "efficientnet_b1",
  "vit_b16",
  "swin_tiny",
  "convnext_tiny",
];

const fusionOptions = [
  { value: "concatenate", label: "Concatenate（拼接）" },
  { value: "gated", label: "Gated Fusion（门控融合）" },
  { value: "attention", label: "Attention Fusion（注意力融合）" },
];

const workflowDatasetPresets = [
  {
    value: "repo-mock",
    label: "仓库 mock 数据（推荐）",
    fields: {
      datasetName: "repo-mock",
      dataPath: "data/mock",
      csvPath: "data/mock/metadata.csv",
      imageDir: "data/mock",
      imagePathColumn: "image_path",
      targetColumn: "diagnosis",
      patientIdColumn: "",
      numericalFeatures: "age",
      categoricalFeatures: "gender",
      numClasses: 2,
      imageSize: 64,
      batchSize: 8,
      numWorkers: 4,
    },
  },
];

export default function NodeConfigPanel({
  node,
  open,
  onClose,
  onSave,
}: NodeConfigPanelProps) {
  const [form] = Form.useForm();

  useEffect(() => {
    if (!node) {
      form.resetFields();
      return;
    }

    form.setFieldsValue(node.data);
  }, [form, node]);

  const handleSave = () => {
    form.validateFields().then((values) => {
      if (!node) {
        return;
      }
      onSave(node.id, values);
      onClose();
    });
  };

  const renderFormFields = () => {
    if (!node) {
      return null;
    }

    switch (node.type) {
      case "dataLoader":
        return (
          <>
            <Form.Item label="快速模板" name="datasetPreset">
              <Select
                allowClear
                placeholder="先选一个可运行模板，或手工填写下方字段"
                onChange={(value) => {
                  const preset = workflowDatasetPresets.find((item) => item.value === value);
                  if (preset) {
                    form.setFieldsValue(preset.fields);
                  }
                }}
              >
                {workflowDatasetPresets.map((preset) => (
                  <Select.Option key={preset.value} value={preset.value}>
                    {preset.label}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Divider orientation="left" plain>
              数据源
            </Divider>

            <Form.Item label="数据集名称" name="datasetName" initialValue="workflow-dataset">
              <Input placeholder="例如：repo-mock / my-private-dataset" />
            </Form.Item>
            <Form.Item label="datasetId（可选）" name="datasetId">
              <Input placeholder="如果来自 DatasetManager，可填写已有 dataset id" />
            </Form.Item>
            <Form.Item
              label="dataPath"
              name="dataPath"
              rules={[{ required: true, message: "请填写数据目录或数据描述文件路径" }]}
            >
              <Input placeholder="例如：data/mock 或 C:\\datasets\\study-a" />
            </Form.Item>
            <Form.Item label="csvPath（可选）" name="csvPath">
              <Input placeholder="例如：data/mock/metadata.csv" />
            </Form.Item>
            <Form.Item label="imageDir（可选）" name="imageDir">
              <Input placeholder="例如：data/mock" />
            </Form.Item>
            <Form.Item
              label="imagePathColumn"
              name="imagePathColumn"
              initialValue="image_path"
            >
              <Input placeholder="例如：image_path" />
            </Form.Item>
            <Form.Item label="targetColumn" name="targetColumn" initialValue="diagnosis">
              <Input placeholder="例如：diagnosis / label" />
            </Form.Item>
            <Form.Item label="patientIdColumn（可选）" name="patientIdColumn">
              <Input placeholder="例如：patient_id" />
            </Form.Item>
            <Form.Item label="数值特征（可选）" name="numericalFeatures">
              <Input placeholder="逗号分隔，例如：age,bmi" />
            </Form.Item>
            <Form.Item label="类别特征（可选）" name="categoricalFeatures">
              <Input placeholder="逗号分隔，例如：gender,site" />
            </Form.Item>
            <Form.Item label="类别数（可选）" name="numClasses" initialValue={2}>
              <InputNumber min={2} max={1000} style={{ width: "100%" }} />
            </Form.Item>

            <Divider orientation="left" plain>
              加载参数
            </Divider>

            <Form.Item label="批次大小" name="batchSize" initialValue={8}>
              <InputNumber min={1} max={512} style={{ width: "100%" }} />
            </Form.Item>
            <Form.Item label="图像大小" name="imageSize" initialValue={64}>
              <InputNumber min={32} max={1024} step={32} style={{ width: "100%" }} />
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
                {backboneOptions.map((option) => (
                  <Select.Option key={option} value={option}>
                    {option}
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
            <Form.Item label="Fusion 策略" name="fusion" initialValue="concatenate">
              <Select>
                {fusionOptions.map((option) => (
                  <Select.Option key={option.value} value={option.value}>
                    {option.label}
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>
          </>
        );

      case "training":
        return (
          <>
            <Form.Item label="实验名称（可选）" name="experimentName">
              <Input placeholder="留空则回退到工作流名称" />
            </Form.Item>
            <Form.Item
              label="训练轮数"
              name="epochs"
              initialValue={3}
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
              <InputNumber min={0.00001} max={1} step={0.0001} style={{ width: "100%" }} />
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
                <Select.Option value="f1">F1 分数</Select.Option>
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
      width={420}
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
      <Form form={form} layout="vertical">
        {renderFormFields()}
      </Form>
    </Drawer>
  );
}
