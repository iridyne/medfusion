import React, { useState } from "react";
import {
  Alert,
  Button,
  Divider,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Typography,
  message,
} from "antd";
import {
  DatabaseOutlined,
  FileImageOutlined,
  FileTextOutlined,
  FolderOpenOutlined,
} from "@ant-design/icons";

import { analyzeDataset, createDataset } from "@/api/datasets";

const { TextArea } = Input;
const { Paragraph, Text } = Typography;

interface DatasetUploaderProps {
  onSuccess: () => void;
  onCancel: () => void;
}

interface DatasetRegisterValues {
  name: string;
  description?: string;
  type: "image" | "tabular" | "multimodal";
  dataPath: string;
  numSamples?: number;
  numClasses?: number;
  tags?: string[];
  createdBy?: string;
}

const DatasetUploader: React.FC<DatasetUploaderProps> = ({
  onSuccess,
  onCancel,
}) => {
  const [form] = Form.useForm<DatasetRegisterValues>();
  const [submitting, setSubmitting] = useState(false);

  const handleRegister = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);

      const created = await createDataset({
        name: values.name,
        description: values.description,
        data_path: values.dataPath,
        dataset_type: values.type,
        num_samples: values.numSamples,
        num_classes: values.numClasses,
        tags: values.tags,
        created_by: values.createdBy,
      });

      try {
        await analyzeDataset(created.id);
      } catch (error) {
        console.warn("Failed to analyze dataset after registration:", error);
      }

      message.success("数据集已登记");
      form.resetFields();
      onSuccess();
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }

      console.error("Failed to register dataset:", error);
      message.error(error?.response?.data?.detail || "登记数据集失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <Alert
        type="info"
        showIcon
        message="当前版本采用本地目录登记"
        description="当前版本不在浏览器里直接上传大文件，而是登记一份本地数据目录，保证数据管理、训练与结果链路的可复现性。"
        style={{ marginBottom: 16 }}
      />

      <Paragraph type="secondary" style={{ marginBottom: 16 }}>
        推荐填写真实存在的本地路径，例如
        <Text code style={{ marginLeft: 8 }}>
          /data/chest-xray
        </Text>
        或
        <Text code style={{ marginLeft: 8 }}>
          D:\\datasets\\breast-ultrasound
        </Text>
        。
      </Paragraph>

      <Form
        form={form}
        layout="vertical"
        initialValues={{ type: "image" }}
      >
        <Form.Item
          label="数据集名称"
          name="name"
          rules={[
            { required: true, message: "请输入数据集名称" },
            { min: 2, message: "名称至少 2 个字符" },
            { max: 100, message: "名称最多 100 个字符" },
          ]}
        >
          <Input
            prefix={<DatabaseOutlined />}
            placeholder="例如：Chest X-Ray Dataset"
          />
        </Form.Item>

        <Form.Item
          label="数据集类型"
          name="type"
          rules={[{ required: true, message: "请选择数据集类型" }]}
        >
          <Select placeholder="选择数据集类型">
            <Select.Option value="image">
              <Space>
                <FileImageOutlined />
                图像数据集
              </Space>
            </Select.Option>
            <Select.Option value="tabular">
              <Space>
                <FileTextOutlined />
                表格数据集
              </Space>
            </Select.Option>
            <Select.Option value="multimodal">
              <Space>
                <DatabaseOutlined />
                多模态数据集
              </Space>
            </Select.Option>
          </Select>
        </Form.Item>

        <Form.Item
          label="本地目录路径"
          name="dataPath"
          rules={[
            { required: true, message: "请输入本地目录路径" },
            { min: 3, message: "路径看起来过短，请确认" },
          ]}
        >
          <Input
            prefix={<FolderOpenOutlined />}
            placeholder="例如：/data/chest-xray"
          />
        </Form.Item>

        <Form.Item label="描述" name="description">
          <TextArea
            rows={3}
            placeholder="简要描述数据内容、来源和适用场景"
            maxLength={500}
            showCount
          />
        </Form.Item>

        <Divider />

        <Space
          size={16}
          style={{ display: "flex", marginBottom: 8 }}
          align="start"
        >
          <Form.Item label="样本数" name="numSamples" style={{ minWidth: 180 }}>
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              placeholder="可选"
            />
          </Form.Item>

          <Form.Item label="类别数" name="numClasses" style={{ minWidth: 180 }}>
            <InputNumber
              min={0}
              style={{ width: "100%" }}
              placeholder="可选"
            />
          </Form.Item>

          <Form.Item label="创建人" name="createdBy" style={{ flex: 1 }}>
            <Input placeholder="例如：lab-team" />
          </Form.Item>
        </Space>

        <Form.Item label="标签" name="tags">
          <Select
            mode="tags"
            placeholder="添加标签，回车确认"
            options={[
              { value: "医学影像", label: "医学影像" },
              { value: "分类", label: "分类" },
              { value: "快速验证", label: "快速验证" },
              { value: "多模态", label: "多模态" },
            ]}
          />
        </Form.Item>
      </Form>

      <Space style={{ width: "100%", justifyContent: "flex-end", marginTop: 24 }}>
        <Button onClick={onCancel} disabled={submitting}>
          取消
        </Button>
        <Button type="primary" onClick={handleRegister} loading={submitting}>
          登记数据集
        </Button>
      </Space>
    </div>
  );
};

export default DatasetUploader;
