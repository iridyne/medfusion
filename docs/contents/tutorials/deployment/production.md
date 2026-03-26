# 生产环境清单

**预计时间：15分钟**

本清单帮助你在将 MedFusion 模型部署到生产环境前进行全面检查。

## 模型验证

### 1. 功能测试

```python
import torch
import numpy as np

# 加载模型
model = torch.load("outputs/checkpoints/best.pth")
model.eval()

# 测试基本推理
test_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(test_input)
    assert output.shape == (1, 2), "输出形状错误"
    assert not torch.isnan(output).any(), "输出包含 NaN"
    assert not torch.isinf(output).any(), "输出包含 Inf"

print("✓ 基本功能测试通过")
```

### 2. 边界条件测试

```python
# 测试不同批次大小
for batch_size in [1, 8, 16, 32]:
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        assert output.shape[0] == batch_size

# 测试极端输入
test_cases = [
    torch.zeros(1, 3, 224, 224),  # 全零
    torch.ones(1, 3, 224, 224),   # 全一
    torch.randn(1, 3, 224, 224) * 1000,  # 大值
]

for x in test_cases:
    with torch.no_grad():
        output = model(x)
        assert not torch.isnan(output).any()

print("✓ 边界条件测试通过")
```

### 3. 性能基准测试

```python
import time

# 预热
for _ in range(10):
    with torch.no_grad():
        _ = model(test_input)

# 性能测试
times = []
for _ in range(100):
    start = time.time()
    with torch.no_grad():
        _ = model(test_input)
    times.append(time.time() - start)

avg_time = np.mean(times)
std_time = np.std(times)
p95_time = np.percentile(times, 95)

print(f"平均推理时间: {avg_time*1000:.2f}ms")
print(f"标准差: {std_time*1000:.2f}ms")
print(f"P95 延迟: {p95_time*1000:.2f}ms")

# 设置性能阈值
assert avg_time < 0.1, "平均推理时间超过 100ms"
assert p95_time < 0.15, "P95 延迟超过 150ms"

print("✓ 性能基准测试通过")
```

### 4. 准确率验证

```python
from sklearn.metrics import accuracy_score, roc_auc_score

# 在测试集上评估
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)

print(f"测试集准确率: {accuracy:.4f}")
print(f"测试集 AUC: {auc:.4f}")

# 设置最低准确率阈值
assert accuracy > 0.80, f"准确率 {accuracy:.4f} 低于阈值 0.80"
assert auc > 0.85, f"AUC {auc:.4f} 低于阈值 0.85"

print("✓ 准确率验证通过")
```

## 错误处理

### 1. 异常捕获

```python
def safe_predict(model, image):
    """
    安全的预测函数，包含完整的错误处理
    """
    try:
        # 输入验证
        if image is None:
            raise ValueError("输入图像为 None")

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"期望 torch.Tensor，得到 {type(image)}")

        if image.dim() != 4:
            raise ValueError(f"期望 4D 张量，得到 {image.dim()}D")

        # 推理
        model.eval()
        with torch.no_grad():
            output = model(image)

        # 输出验证
        if torch.isnan(output).any():
            raise RuntimeError("模型输出包含 NaN")

        if torch.isinf(output).any():
            raise RuntimeError("模型输出包含 Inf")

        return output

    except ValueError as e:
        print(f"输入验证错误: {e}")
        return None
    except RuntimeError as e:
        print(f"推理错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

# 测试错误处理
assert safe_predict(model, None) is None
assert safe_predict(model, "invalid") is None
assert safe_predict(model, torch.randn(3, 224, 224)) is None  # 缺少批次维度

print("✓ 错误处理测试通过")
```

### 2. 日志记录

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def predict_with_logging(model, image):
    """
    带日志记录的预测函数
    """
    logger.info("开始预测")

    try:
        output = safe_predict(model, image)

        if output is None:
            logger.error("预测失败")
            return None

        logger.info(f"预测成功，输出形状: {output.shape}")
        return output

    except Exception as e:
        logger.exception("预测过程中发生异常")
        return None

print("✓ 日志记录配置完成")
```

## 监控和告警

### 1. 性能监控

```python
import time
from collections import deque

class PerformanceMonitor:
    """
    性能监控器
    """
    def __init__(self, window_size=100):
        self.latencies = deque(maxlen=window_size)
        self.errors = 0
        self.total_requests = 0

    def record_request(self, latency, success=True):
        self.total_requests += 1
        self.latencies.append(latency)
        if not success:
            self.errors += 1

    def get_metrics(self):
        if not self.latencies:
            return {}

        return {
            'avg_latency': np.mean(self.latencies),
            'p95_latency': np.percentile(self.latencies, 95),
            'p99_latency': np.percentile(self.latencies, 99),
            'error_rate': self.errors / self.total_requests,
            'total_requests': self.total_requests
        }

    def check_health(self):
        metrics = self.get_metrics()

        # 健康检查阈值
        if metrics.get('avg_latency', 0) > 0.2:
            logger.warning(f"平均延迟过高: {metrics['avg_latency']:.3f}s")

        if metrics.get('error_rate', 0) > 0.05:
            logger.error(f"错误率过高: {metrics['error_rate']:.2%}")

        return metrics

# 使用监控器
monitor = PerformanceMonitor()

for _ in range(100):
    start = time.time()
    output = safe_predict(model, test_input)
    latency = time.time() - start
    monitor.record_request(latency, success=(output is not None))

metrics = monitor.check_health()
print(f"监控指标: {metrics}")

print("✓ 性能监控配置完成")
```

### 2. 健康检查端点

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    try:
        # 检查模型是否可用
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(test_input)

        # 检查性能指标
        metrics = monitor.get_metrics()

        if metrics.get('error_rate', 0) > 0.1:
            raise HTTPException(status_code=503, detail="错误率过高")

        return {
            "status": "healthy",
            "metrics": metrics
        }

    except Exception as e:
        logger.exception("健康检查失败")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/ready")
async def readiness_check():
    """
    就绪检查端点
    """
    # 检查模型是否加载
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    return {"status": "ready"}

print("✓ 健康检查端点配置完成")
```

## 安全检查

### 1. 输入验证

```python
def validate_input(image):
    """
    验证输入图像
    """
    # 类型检查
    if not isinstance(image, torch.Tensor):
        raise TypeError("输入必须是 torch.Tensor")

    # 形状检查
    if image.dim() != 4:
        raise ValueError(f"输入必须是 4D 张量，得到 {image.dim()}D")

    # 值范围检查
    if image.min() < -10 or image.max() > 10:
        raise ValueError("输入值超出合理范围")

    # 大小检查
    if image.shape[2] > 1024 or image.shape[3] > 1024:
        raise ValueError("输入图像尺寸过大")

    return True

print("✓ 输入验证配置完成")
```

### 2. 速率限制

```python
from collections import defaultdict
import time

class RateLimiter:
    """
    简单的速率限制器
    """
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    def is_allowed(self, client_id):
        now = time.time()
        # 清理过期请求
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window
        ]

        # 检查是否超过限制
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # 记录请求
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window=60)

print("✓ 速率限制配置完成")
```

## 部署清单

### 环境检查

- [ ] Python 版本 >= 3.11
- [ ] PyTorch 版本 >= 2.0
- [ ] CUDA 版本兼容（如使用 GPU）
- [ ] 所有依赖已安装
- [ ] 环境变量已配置

### 模型检查

- [ ] 模型文件完整
- [ ] 模型可以正常加载
- [ ] 推理结果正确
- [ ] 性能满足要求
- [ ] 内存占用合理

### 代码检查

- [ ] 所有测试通过
- [ ] 代码已经过 lint 检查
- [ ] 类型检查通过
- [ ] 无安全漏洞
- [ ] 日志记录完善

### 配置检查

- [ ] 生产配置文件就绪
- [ ] 敏感信息已移除
- [ ] 资源限制已设置
- [ ] 超时配置合理
- [ ] 重试策略已配置

### 监控检查

- [ ] 日志收集配置完成
- [ ] 性能监控已启用
- [ ] 告警规则已设置
- [ ] 健康检查端点可用
- [ ] 指标导出正常

### 安全检查

- [ ] 输入验证已实现
- [ ] 速率限制已启用
- [ ] 认证授权已配置
- [ ] HTTPS 已启用
- [ ] 敏感数据已加密

### 容灾检查

- [ ] 备份策略已制定
- [ ] 回滚方案已准备
- [ ] 故障转移已配置
- [ ] 数据恢复已测试
- [ ] 应急预案已制定

## 部署脚本

### 自动化部署

```bash
#!/bin/bash
# deploy.sh - 自动化部署脚本

set -e  # 遇到错误立即退出

echo "开始部署..."

# 1. 环境检查
echo "检查环境..."
python --version
pip list | grep torch

# 2. 运行测试
echo "运行测试..."
pytest tests/ -v

# 3. 构建 Docker 镜像
echo "构建 Docker 镜像..."
docker build -t medfusion:latest .

# 4. 运行健康检查
echo "运行健康检查..."
docker run --rm medfusion:latest python -c "import torch; print('OK')"

# 5. 部署
echo "部署到生产环境..."
docker-compose -f docker-compose.prod.yml up -d

# 6. 验证部署
echo "验证部署..."
sleep 10
curl -f http://localhost:8000/health || exit 1

echo "部署完成！"
```

### 回滚脚本

```bash
#!/bin/bash
# rollback.sh - 回滚脚本

set -e

echo "开始回滚..."

# 1. 停止当前版本
docker-compose -f docker-compose.prod.yml down

# 2. 恢复上一个版本
docker tag medfusion:previous medfusion:latest

# 3. 重新部署
docker-compose -f docker-compose.prod.yml up -d

# 4. 验证
sleep 10
curl -f http://localhost:8000/health || exit 1

echo "回滚完成！"
```

## 常见问题

### Q1: 如何设置性能阈值？

A: 根据业务需求设置：
- 医疗诊断：准确率 > 90%，延迟 < 500ms
- 辅助筛查：准确率 > 85%，延迟 < 200ms
- 研究用途：准确率 > 80%，延迟 < 1s

### Q2: 如何处理模型更新？

A: 使用蓝绿部署或金丝雀发布：
1. 部署新版本到独立环境
2. 逐步切换流量
3. 监控指标
4. 出现问题立即回滚

### Q3: 如何监控生产环境？

A: 三个层面：
1. 基础设施：CPU、内存、磁盘、网络
2. 应用层：延迟、吞吐量、错误率
3. 业务层：准确率、用户满意度

### Q4: 如何保证高可用？

A: 多种策略：
1. 负载均衡
2. 多实例部署
3. 自动重启
4. 健康检查
5. 故障转移

## 下一步

恭喜完成所有教程！接下来可以：

- 查看 [API 文档](../../api/) - 深入了解 API
- 阅读 [高级指南](../../guides/) - 学习高级特性
- 参与 [社区讨论](https://github.com/iridyne/medfusion/discussions) - 与其他用户交流
- 贡献代码 - 帮助改进 MedFusion

## 参考资源

- [Docker 部署指南](docker.md)
- [模型导出指南](model-export.md)
- [快速参考](../../guides/core/quick-reference.md)
- [性能基准测试](../../guides/advanced-features/performance-benchmarking.md)
