# 类型注解改进计划

**创建日期**: 2026-02-25
**当前状态**: 进行中
**目标**: 将类型注解覆盖率从 40% 提升到 80%+

---

## 当前状态

### 统计数据
- **总类型错误**: 678 个
- **已修复**: 11 个
- **剩余**: 667 个

### 已完成模块 ✅
- `med_core/web/config.py` (2 个错误)
- `med_core/exceptions.py` (1 个错误)
- `med_core/monitoring/metrics.py` (3 个错误)
- `med_core/registry/model_registry.py` (5 个错误)

### 待处理模块 ⏳

#### 高优先级 (核心模块)
1. **med_core/utils/tuning.py** - 约 20 个错误
   - 缺少返回类型注解
   - 缺少参数类型注解
   - 工作量: 2-3 小时

2. **med_core/utils/logging.py** - 约 5 个错误
   - 类型赋值不兼容
   - 工作量: 1 小时

3. **med_core/evaluation/metrics_calculator.py** - 约 3 个错误
   - 字典类型不匹配
   - 工作量: 1 小时

#### 中优先级 (Web 模块)
4. **med_core/web/** - 约 100 个错误
   - `api/datasets.py`
   - `api/training.py`
   - `services/training_service.py`
   - `workflow_engine.py`
   - `node_executors.py`
   - 工作量: 1-2 天

#### 低优先级 (其他模块)
5. **med_core/datasets/** - 约 200 个错误
6. **med_core/trainers/** - 约 150 个错误
7. **med_core/models/** - 约 100 个错误
8. **med_core/backbones/** - 约 89 个错误

---

## 改进策略

### 阶段 1: 快速修复 (1-2 天)
**目标**: 修复 100 个最简单的错误

**方法**:
1. 添加缺失的返回类型注解 (`-> None`, `-> dict`, 等)
2. 为 `__init__` 方法添加 `-> None`
3. 为简单函数添加类型提示

**预期结果**: 减少到 ~550 个错误

### 阶段 2: 核心模块 (2-3 天)
**目标**: 完成核心模块的类型注解

**模块**:
- `med_core/utils/`
- `med_core/evaluation/`
- `med_core/configs/`

**预期结果**: 减少到 ~400 个错误

### 阶段 3: 数据和训练模块 (3-5 天)
**目标**: 完成数据加载和训练相关模块

**模块**:
- `med_core/datasets/`
- `med_core/trainers/`

**预期结果**: 减少到 ~150 个错误

### 阶段 4: Web 和其他模块 (2-3 天)
**目标**: 完成 Web UI 和其他辅助模块

**模块**:
- `med_core/web/`
- `med_core/serving/`
- `med_core/monitoring/`

**预期结果**: 减少到 <50 个错误

### 阶段 5: 最终清理 (1-2 天)
**目标**: 修复剩余的复杂类型错误

**方法**:
1. 使用 `TypedDict` 定义复杂字典结构
2. 使用 `Protocol` 定义接口
3. 添加泛型类型参数

**预期结果**: <10 个错误

---

## 类型注解最佳实践

### 1. 函数签名
```python
# ❌ 错误
def process_data(data, config):
    return result

# ✅ 正确
def process_data(
    data: pd.DataFrame,
    config: dict[str, Any]
) -> dict[str, float]:
    return result
```

### 2. 类方法
```python
# ❌ 错误
class Model:
    def __init__(self, config):
        self.config = config

    def forward(self, x):
        return self.model(x)

# ✅ 正确
class Model:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

### 3. 复杂类型
```python
from typing import TypedDict, Protocol

# 使用 TypedDict 定义字典结构
class ConfigDict(TypedDict):
    learning_rate: float
    batch_size: int
    epochs: int

# 使用 Protocol 定义接口
class Trainable(Protocol):
    def train(self, data: DataLoader) -> None: ...
    def evaluate(self, data: DataLoader) -> dict[str, float]: ...
```

### 4. 可选类型
```python
from typing import Optional

# ❌ 旧式
def get_model(name: str) -> Optional[Model]:
    ...

# ✅ 新式 (Python 3.10+)
def get_model(name: str) -> Model | None:
    ...
```

---

## 工具和配置

### mypy 配置
```ini
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
strict_equality = true
ignore_missing_imports = true
```

### 运行检查
```bash
# 检查所有错误
uv run mypy med_core --ignore-missing-imports

# 检查特定模块
uv run mypy med_core/utils --ignore-missing-imports

# 生成报告
uv run mypy med_core --ignore-missing-imports --html-report mypy-report
```

---

## 进度跟踪

### 每日目标
- **第 1 天**: 修复 utils/ 和 evaluation/ (50-100 个错误)
- **第 2 天**: 修复 configs/ 和 registry/ (30-50 个错误)
- **第 3-5 天**: 修复 datasets/ 和 trainers/ (200-300 个错误)
- **第 6-8 天**: 修复 web/ 和其他模块 (150-200 个错误)
- **第 9-10 天**: 最终清理和验证 (<50 个错误)

### 里程碑
- [ ] **里程碑 1**: 减少到 500 个错误 (预计: 第 2 天)
- [ ] **里程碑 2**: 减少到 300 个错误 (预计: 第 5 天)
- [ ] **里程碑 3**: 减少到 100 个错误 (预计: 第 8 天)
- [ ] **里程碑 4**: 减少到 <50 个错误 (预计: 第 10 天)
- [ ] **里程碑 5**: 完成 (目标: <10 个错误)

---

## 注意事项

1. **不要过度使用 `Any`**: 尽量使用具体类型
2. **保持向后兼容**: 不要破坏现有 API
3. **测试覆盖**: 类型注解不能替代测试
4. **渐进式改进**: 优先修复核心模块
5. **文档同步**: 更新文档字符串以匹配类型签名

---

## 参考资源

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [PEP 585 - Type Hinting Generics](https://www.python.org/dev/peps/pep-0585/)

---

**最后更新**: 2026-02-25
**负责人**: 开发团队
**审查周期**: 每周
