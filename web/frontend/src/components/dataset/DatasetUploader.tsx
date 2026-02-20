import React, { useState } from "react";
import {
  Upload,
  Form,
  Input,
  Select,
  Button,
  Space,
  message,
  Progress,
  List,
  Tag,
  Alert,
  Divider,
} from "antd";
import {
  InboxOutlined,
  FileImageOutlined,
  FileTextOutlined,
  FileZipOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
} from "@ant-design/icons";
import type { UploadFile, UploadProps } from "antd/es/upload/interface";

const { Dragger } = Upload;
const { TextArea } = Input;

interface DatasetUploaderProps {
  onSuccess: () => void;
  onCancel: () => void;
}

interface DatasetMetadata {
  name: string;
  description?: string;
  type: "image" | "tabular" | "multimodal";
  tags?: string[];
}

const DatasetUploader: React.FC<DatasetUploaderProps> = ({ onSuccess, onCancel }) => {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // 文件类型配置
  const acceptedFileTypes = {
    image: [".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz"],
    tabular: [".csv", ".xlsx", ".json"],
    archive: [".zip", ".tar", ".tar.gz"],
  };

  // 获取文件图标
  const getFileIcon = (fileName: string) => {
    const ext = fileName.toLowerCase().split(".").pop();
    if (["jpg", "jpeg", "png", "dcm", "nii"].includes(ext || "")) {
      return <FileImageOutlined style={{ color: "#1890ff" }} />;
    } else if (["csv", "xlsx", "json"].includes(ext || "")) {
      return <FileTextOutlined style={{ color: "#52c41a" }} />;
    } else if (["zip", "tar", "gz"].includes(ext || "")) {
      return <FileZipOutlined style={{ color: "#faad14" }} />;
    }
    return <FileTextOutlined />;
  };

  // 验证文件类型
  const validateFileType = (file: File, datasetType: string): boolean => {
    const fileName = file.name.toLowerCase();
    const ext = fileName.split(".").pop() || "";

    if (datasetType === "image") {
      return acceptedFileTypes.image.some((type) => fileName.endsWith(type));
    } else if (datasetType === "tabular") {
      return acceptedFileTypes.tabular.some((type) => fileName.endsWith(type));
    } else if (datasetType === "multimodal") {
      return (
        acceptedFileTypes.image.some((type) => fileName.endsWith(type)) ||
        acceptedFileTypes.tabular.some((type) => fileName.endsWith(type)) ||
        acceptedFileTypes.archive.some((type) => fileName.endsWith(type))
      );
    }
    return false;
  };

  // 上传配置
  const uploadProps: UploadProps = {
    name: "file",
    multiple: true,
    fileList,
    beforeUpload: (file) => {
      const datasetType = form.getFieldValue("type");
      if (!datasetType) {
        message.error("请先选择数据集类型");
        return Upload.LIST_IGNORE;
      }

      if (!validateFileType(file, datasetType)) {
        message.error(`文件 ${file.name} 类型不符合要求`);
        return Upload.LIST_IGNORE;
      }

      // 检查文件大小（限制 5GB）
      const maxSize = 5 * 1024 * 1024 * 1024;
      if (file.size > maxSize) {
        message.error(`文件 ${file.name} 超过 5GB 限制`);
        return Upload.LIST_IGNORE;
      }

      setFileList((prev) => [...prev, file as UploadFile]);
      return false; // 阻止自动上传
    },
    onRemove: (file) => {
      setFileList((prev) => prev.filter((f) => f.uid !== file.uid));
    },
    showUploadList: false,
  };

  // 处理上传
  const handleUpload = async () => {
    try {
      const values = await form.validateFields();

      if (fileList.length === 0) {
        message.error("请至少上传一个文件");
        return;
      }

      setUploading(true);
      setUploadProgress(0);

      // 创建 FormData
      const formData = new FormData();
      formData.append("name", values.name);
      formData.append("type", values.type);
      if (values.description) {
        formData.append("description", values.description);
      }
      if (values.tags && values.tags.length > 0) {
        formData.append("tags", JSON.stringify(values.tags));
      }

      // 添加文件
      fileList.forEach((file) => {
        if (file.originFileObj) {
          formData.append("files", file.originFileObj);
        }
      });

      // TODO: 调用后端 API
      // const response = await fetch("/api/datasets/upload", {
      //   method: "POST",
      //   body: formData,
      //   onUploadProgress: (progressEvent) => {
      //     const percentCompleted = Math.round(
      //       (progressEvent.loaded * 100) / progressEvent.total
      //     );
      //     setUploadProgress(percentCompleted);
      //   },
      // });

      // 模拟上传进度
      for (let i = 0; i <= 100; i += 10) {
        await new Promise((resolve) => setTimeout(resolve, 200));
        setUploadProgress(i);
      }

      message.success("数据集上传成功");
      onSuccess();
    } catch (error) {
      message.error("上传失败");
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  // 移除文件
  const handleRemoveFile = (file: UploadFile) => {
    setFileList((prev) => prev.filter((f) => f.uid !== file.uid));
  };

  // 格式化文件大小
  const formatSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // 获取接受的文件类型提示
  const getAcceptedTypesHint = (type: string): string => {
    if (type === "image") {
      return "支持: JPG, PNG, DICOM (.dcm), NIfTI (.nii, .nii.gz)";
    } else if (type === "tabular") {
      return "支持: CSV, Excel (.xlsx), JSON";
    } else if (type === "multimodal") {
      return "支持: 图像文件、表格文件、ZIP 压缩包";
    }
    return "";
  };

  const datasetType = Form.useWatch("type", form);

  return (
    <div>
      <Form
        form={form}
        layout="vertical"
        initialValues={{ type: "image" }}
      >
        {/* 基本信息 */}
        <Form.Item
          label="数据集名称"
          name="name"
          rules={[
            { required: true, message: "请输入数据集名称" },
            { min: 3, message: "名称至少 3 个字符" },
            { max: 100, message: "名称最多 100 个字符" },
          ]}
        >
          <Input placeholder="例如: Chest X-Ray Dataset" />
        </Form.Item>

        <Form.Item
          label="数据集类型"
          name="type"
          rules={[{ required: true, message: "请选择数据集类型" }]}
        >
          <Select
            placeholder="选择数据集类型"
            onChange={() => setFileList([])} // 切换类型时清空文件列表
          >
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
                <FileZipOutlined />
                多模态数据集
              </Space>
            </Select.Option>
          </Select>
        </Form.Item>

        <Form.Item label="描述" name="description">
          <TextArea
            rows={3}
            placeholder="简要描述数据集的内容、来源和用途"
            maxLength={500}
            showCount
          />
        </Form.Item>

        <Form.Item label="标签" name="tags">
          <Select
            mode="tags"
            placeholder="添加标签（按回车添加）"
            style={{ width: "100%" }}
          >
            <Select.Option value="医学影像">医学影像</Select.Option>
            <Select.Option value="分类">分类</Select.Option>
            <Select.Option value="检测">检测</Select.Option>
            <Select.Option value="分割">分割</Select.Option>
            <Select.Option value="多模态">多模态</Select.Option>
          </Select>
        </Form.Item>

        <Divider />

        {/* 文件上传区域 */}
        <Form.Item label="上传文件">
          {datasetType && (
            <Alert
              message={getAcceptedTypesHint(datasetType)}
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}
          <Dragger {...uploadProps} disabled={!datasetType || uploading}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
            <p className="ant-upload-hint">
              支持单个或批量上传。请先选择数据集类型。
            </p>
          </Dragger>
        </Form.Item>

        {/* 文件列表 */}
        {fileList.length > 0 && (
          <Form.Item label={`已选择文件 (${fileList.length})`}>
            <List
              size="small"
              bordered
              dataSource={fileList}
              style={{ maxHeight: 300, overflow: "auto" }}
              renderItem={(file) => (
                <List.Item
                  actions={[
                    <Button
                      type="text"
                      danger
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() => handleRemoveFile(file)}
                      disabled={uploading}
                    />,
                  ]}
                >
                  <List.Item.Meta
                    avatar={getFileIcon(file.name)}
                    title={file.name}
                    description={formatSize(file.size || 0)}
                  />
                </List.Item>
              )}
            />
          </Form.Item>
        )}

        {/* 上传进度 */}
        {uploading && (
          <Form.Item>
            <Progress percent={uploadProgress} status="active" />
          </Form.Item>
        )}
      </Form>

      {/* 操作按钮 */}
      <Space style={{ width: "100%", justifyContent: "flex-end", marginTop: 24 }}>
        <Button onClick={onCancel} disabled={uploading}>
          取消
        </Button>
        <Button
          type="primary"
          onClick={handleUpload}
          loading={uploading}
          disabled={fileList.length === 0}
        >
          {uploading ? "上传中..." : "开始上传"}
        </Button>
      </Space>
    </div>
  );
};

export default DatasetUploader;
