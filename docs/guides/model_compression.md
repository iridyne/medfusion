# 模型压缩指南

## 概述

支持量化和剪枝两种压缩方法。

## 快速开始

### 量化
```python
from med_core.utils.compression import quantize_model

# 动态量化
quantized_model = quantize_model(model, method="dynamic")
```

### 剪枝
```python
from med_core.utils.compression import prune_model

# 非结构化剪枝
pruned_model = prune_model(model, amount=0.3)
```

### 完整压缩
```python
from med_core.utils.compression import compress_model

compressed_model = compress_model(model, quantize=True, prune=True)
```

## 参考
- `med_core/utils/compression.py`
- `examples/model_compression_demo.py`
