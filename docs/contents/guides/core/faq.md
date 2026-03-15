# FAQ 和故障排查指南

本指南提供 MedFusion 框架常见问题的解答和故障排查方法。

## 目录

- [常见问题 (FAQ)](#常见问题-faq)
- [安装问题](#安装问题)
- [配置问题](#配置问题)
- [训练问题](#训练问题)
- [数据加载问题](#数据加载问题)
- [GPU 和内存问题](#gpu-和内存问题)
- [模型问题](#模型问题)
- [性能问题](#性能问题)
- [Docker 问题](#docker-问题)
- [调试技巧](#调试技巧)

---

## 常见问题 (FAQ)

### Q1: MedFusion 支持哪些 Python 版本？

**A**: MedFusion 支持 Python 3.10, 3.11, 和 3.12。

```bash
# 检查 Python 版本
python --version

# 推荐使用 Python 3.11
python3.11 -m venv .venv
```

### Q2: 如何安装 MedFusion？

**A**: 使用 uv 或 pip 安装：

```bash
# 使用 uv (推荐)
uv pip install -e .

# 使用 pip
pip install -e .

# 开发模式（包含开发依赖）
uv pip install -e ".[dev]"
```

### Q3: 需要 GPU 吗？

**A**: 不是必需的，但强烈推荐。

- **训练**: 推荐使用 GPU（CUDA 11.0+）
- **推理**: CPU 也可以，但速度较慢
- **开发/测试**: CPU 足够

```bash
# 检查 GPU 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### Q4: 如何查看框架版本？

**A**: 

```python
import med_core
print(med_core.__version__)
```

或使用 CLI：

```bash
python -m med_core.cli --version
```

### Q5: 支持哪些医学影像模态？

**A**: MedFusion 支持多种模态：

- CT (Computed Tomography)
- MRI (Magnetic Resonance Imaging)
- X-Ray
- PET (Positron Emission Tomography)
- 病理图像
- 多模态融合

### Q6: 如何贡献代码？

**A**: 

1. Fork 仓库
2. 创建特性分支
3. 提交更改
4. 运行测试和检查
5. 创建 Pull Request

详见 [CONTRIBUTING.md](../CONTRIBUTING.md)

### Q7: 在哪里获取帮助？

**A**: 

- 📖 查看文档: `docs/`
- 🐛 报告问题: GitHub Issues
- 💬 讨论: GitHub Discussions
- 📧 联系: your-email@example.com

---

## 安装问题

### 问题 1: 安装依赖失败

**症状**:
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**原因**: PyTorch 版本不兼容或网络问题

**解决方案**:

```bash
# 方案 1: 使用清华镜像
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方案 2: 直接从 PyTorch 官网安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 方案 3: 使用 uv (更快)
uv pip install torch torchvision
```

### 问题 2: CUDA 版本不匹配

**症状**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因**: PyTorch CUDA 版本与系统 CUDA 不匹配

**解决方案**:

```bash
# 1. 检查系统 CUDA 版本
nvidia-smi

# 2. 安装匹配的 PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 问题 3: 权限错误

**症状**:
```
PermissionError: [Errno 13] Permission denied
```

**解决方案**:

```bash
# 使用用户安装
pip install --user -e .

# 或使用虚拟环境
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 配置问题

### 问题 1: 配置文件验证失败

**症状**:
```
[E001] model.backbone: Invalid backbone 'resnet999'
```

**原因**: 配置值无效

**解决方案**:

```bash
# 1. 查看可用选项
python -c "from med_core.backbones import AVAILABLE_BACKBONES; print(AVAILABLE_BACKBONES)"

# 2. 使用有效的配置
# 编辑 configs/your_config.yaml
model:
  backbone: resnet50  # 使用有效的 backbone
```

### 问题 2: 配置文件找不到

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'configs/my_config.yaml'
```

**解决方案**:

```bash
# 使用绝对路径
python -m med_core.cli train --config /absolute/path/to/config.yaml

# 或从正确的目录运行
cd /path/to/medfusion
python -m med_core.cli train --config configs/my_config.yaml
```

### 问题 3: 注意力监督配置错误

**症状**:
```
[E028] Attention supervision requires CBAM attention mechanism
```

**原因**: 注意力监督需要 CBAM

**解决方案**:

```yaml
# 方案 1: 启用 CBAM
model:
  vision:
    attention_type: cbam

training:
  use_attention_supervision: true

# 方案 2: 禁用注意力监督
training:
  use_attention_supervision: false
```

---

## 训练问题

### 问题 1: 训练立即崩溃

**症状**:
```
RuntimeError: CUDA out of memory
```

**原因**: GPU 内存不足

**解决方案**:

```yaml
# 1. 减小 batch size
training:
  batch_size: 8  # 从 32 减小

# 2. 使用梯度累积
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # 等效于 batch_size=32

# 3. 使用混合精度
training:
  use_amp: true

# 4. 减小图像尺寸
data:
  image_size: 224  # 从 512 减小
```

### 问题 2: Loss 变成 NaN

**症状**:
```
Epoch 1, Step 100: loss=nan
```

**原因**: 学习率过大、梯度爆炸、数据问题

**解决方案**:

```yaml
# 1. 降低学习率
training:
  optimizer:
    lr: 0.0001  # 从 0.001 降低

# 2. 使用梯度裁剪
training:
  max_grad_norm: 1.0

# 3. 检查数据
# 确保数据归一化正确
data:
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

# 4. 使用更稳定的优化器
training:
  optimizer:
    type: adamw
    weight_decay: 0.01
```

### 问题 3: 训练速度慢

**症状**: GPU 利用率低，训练缓慢

**原因**: 数据加载瓶颈

**解决方案**:

```yaml
# 1. 增加 data workers
data:
  num_workers: 8  # 根据 CPU 核心数调整
  pin_memory: true
  persistent_workers: true

# 2. 使用更快的数据格式
# 将数据转换为 LMDB 或 TFRecord

# 3. 预加载数据到内存
# 如果数据集较小

# 4. 使用 SSD 存储数据
```

### 问题 4: 检查点保存失败

**症状**:
```
OSError: [Errno 28] No space left on device
```

**原因**: 磁盘空间不足

**解决方案**:

```yaml
# 1. 只保存最佳模型
training:
  save_best_only: true

# 2. 限制保存的检查点数量
training:
  max_checkpoints: 3

# 3. 清理旧的输出
rm -rf outputs/old_experiment/

# 4. 使用更大的磁盘
# 或挂载外部存储
```

---

## 数据加载问题

### 问题 1: 数据集找不到

**症状**:
```
FileNotFoundError: Dataset not found at /path/to/data
```

**解决方案**:

```bash
# 1. 检查路径
ls /path/to/data

# 2. 使用绝对路径
# 编辑配置文件
data:
  data_dir: /absolute/path/to/data

# 3. 生成模拟数据（测试用）
python scripts/generate_mock_data.py
```

### 问题 2: 数据格式错误

**症状**:
```
ValueError: Expected CSV with columns: patient_id, image_path, label
```

**原因**: CSV 格式不正确

**解决方案**:

```python
# 检查 CSV 格式
import pandas as pd
df = pd.read_csv('data/metadata.csv')
print(df.columns)
print(df.head())

# 确保包含必需的列
# patient_id, image_path, label
```

### 问题 3: 图像加载失败

**症状**:
```
RuntimeError: Error loading image: /path/to/image.nii.gz
```

**原因**: 图像文件损坏或格式不支持

**解决方案**:

```python
# 1. 验证图像文件
import nibabel as nib
try:
    img = nib.load('image.nii.gz')
    print(f"Shape: {img.shape}")
except Exception as e:
    print(f"Error: {e}")

# 2. 检查文件权限
ls -l /path/to/image.nii.gz

# 3. 重新下载或转换图像
```

### 问题 4: 内存泄漏

**症状**: 内存使用持续增长

**原因**: 数据加载器未正确清理

**解决方案**:

```yaml
# 1. 使用 persistent_workers
data:
  persistent_workers: true

# 2. 减少 num_workers
data:
  num_workers: 4  # 从 16 减少

# 3. 禁用缓存（如果启用）
data:
  cache_data: false
```

---

## GPU 和内存问题

### 问题 1: CUDA out of memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**:

```python
# 1. 清理 GPU 缓存
import torch
torch.cuda.empty_cache()

# 2. 减小 batch size
# 见训练问题部分

# 3. 使用梯度检查点
model = create_model(use_checkpoint=True)

# 4. 监控内存使用
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 问题 2: 多 GPU 训练失败

**症状**:
```
RuntimeError: NCCL error
```

**解决方案**:

```bash
# 1. 检查 GPU 可见性
nvidia-smi

# 2. 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3. 使用正确的启动命令
torchrun --nproc_per_node=4 -m med_core.cli train --config config.yaml

# 4. 检查 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"
```

### 问题 3: GPU 利用率低

**症状**: GPU 使用率 < 50%

**原因**: 数据加载瓶颈或模型太小

**解决方案**:

```yaml
# 1. 增加 batch size
training:
  batch_size: 64  # 增大

# 2. 增加 data workers
data:
  num_workers: 8
  prefetch_factor: 2

# 3. 使用更大的模型
model:
  backbone: resnet101  # 从 resnet50 增大

# 4. 启用混合精度
training:
  use_amp: true
```

---

## 模型问题

### 问题 1: 模型加载失败

**症状**:
```
RuntimeError: Error loading checkpoint
```

**解决方案**:

```python
# 1. 检查检查点文件
import torch
checkpoint = torch.load('model.pth', map_location='cpu')
print(checkpoint.keys())

# 2. 使用兼容模式加载
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# 3. 检查模型架构是否匹配
```

### 问题 2: 预训练权重不兼容

**症状**:
```
RuntimeError: size mismatch for fc.weight
```

**原因**: 类别数不匹配

**解决方案**:

```yaml
# 方案 1: 不加载分类头
model:
  pretrained: true
  load_classifier: false

# 方案 2: 使用正确的类别数
model:
  num_classes: 1000  # 匹配预训练模型
```

### 问题 3: 模型输出形状错误

**症状**:
```
RuntimeError: Expected input shape (B, C, H, W), got (B, H, W, C)
```

**解决方案**:

```python
# 检查输入形状
print(f"Input shape: {x.shape}")

# 转换维度顺序
x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
```

---

## 性能问题

### 问题 1: 推理速度慢

**解决方案**:

```python
# 1. 使用 eval 模式
model.eval()

# 2. 禁用梯度计算
with torch.no_grad():
    output = model(input)

# 3. 使用 TorchScript
model_scripted = torch.jit.script(model)

# 4. 使用 ONNX
torch.onnx.export(model, dummy_input, "model.onnx")

# 5. 使用批处理
# 一次处理多个样本
```

### 问题 2: 训练不收敛

**症状**: Loss 不下降或震荡

**解决方案**:

```yaml
# 1. 调整学习率
training:
  optimizer:
    lr: 0.0001  # 降低
  
  scheduler:
    type: cosine
    warmup_epochs: 5

# 2. 增加训练轮数
training:
  epochs: 100  # 增加

# 3. 使用数据增强
data:
  augmentation:
    random_flip: true
    random_rotation: 15
    color_jitter: 0.2

# 4. 检查数据质量
# 确保标签正确
```

---

## Docker 问题

### 问题 1: Docker 构建失败

**症状**:
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**解决方案**:

```bash
# 1. 清理 Docker 缓存
docker builder prune -a

# 2. 无缓存构建
docker build --no-cache -t medfusion:latest .

# 3. 检查网络连接
# 使用镜像加速器

# 4. 增加构建超时
docker build --network=host -t medfusion:latest .
```

### 问题 2: 容器内 GPU 不可用

**症状**:
```
RuntimeError: CUDA not available
```

**解决方案**:

```bash
# 1. 安装 nvidia-docker
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# 2. 使用 --gpus 标志
docker run --gpus all medfusion:latest

# 3. 在 docker-compose.yml 中配置
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## 调试技巧

### 1. 启用详细日志

```python
from med_core.utils.logging import setup_logging

# 设置 DEBUG 级别
setup_logging(level="DEBUG")
```

### 2. 使用 Python 调试器

```python
# 在代码中设置断点
import pdb; pdb.set_trace()

# 或使用 ipdb
import ipdb; ipdb.set_trace()
```

### 3. 检查中间输出

```python
# 添加打印语句
print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}")
print(f"Min: {tensor.min()}, Max: {tensor.max()}")

# 使用 hook 检查梯度
def print_grad(grad):
    print(f"Gradient: {grad.norm()}")

tensor.register_hook(print_grad)
```

### 4. 可视化数据

```python
import matplotlib.pyplot as plt

# 可视化图像
plt.imshow(image)
plt.show()

# 可视化特征图
plt.imshow(feature_map[0, 0].cpu().detach().numpy())
plt.show()
```

### 5. 性能分析

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 6. 内存分析

```python
import torch

# 监控内存
torch.cuda.memory_summary()

# 检测内存泄漏
import gc
gc.collect()
torch.cuda.empty_cache()
```

---

## 获取帮助

如果以上方法都无法解决问题：

1. **查看日志**: 检查详细的错误日志
2. **搜索 Issues**: 在 GitHub Issues 中搜索类似问题
3. **创建 Issue**: 提供详细的错误信息和复现步骤
4. **社区讨论**: 在 GitHub Discussions 中提问
5. **联系维护者**: 发送邮件到 your-email@example.com

### 报告问题时请包含

- MedFusion 版本
- Python 版本
- PyTorch 版本
- CUDA 版本（如果使用 GPU）
- 操作系统
- 完整的错误堆栈
- 最小可复现示例
- 配置文件

---

**最后更新**: 2026-02-20
