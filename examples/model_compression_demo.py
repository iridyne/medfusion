"""模型压缩示例"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

def create_model():
    return nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

def main():
    print("=" * 60)
    print("模型压缩演示")
    print("=" * 60)
    
    from med_core.utils.compression import quantize_model, prune_model, compress_model
    
    model = create_model()
    
    # 量化
    print("\n1. 动态量化:")
    quantized = quantize_model(model, method="dynamic")
    
    # 剪枝
    print("\n2. 非结构化剪枝:")
    pruned = prune_model(model, amount=0.3, method="unstructured")
    
    # 压缩（量化 + 剪枝）
    print("\n3. 完整压缩:")
    compressed = compress_model(model, quantize=True, prune=True, prune_amount=0.3)
    
    print("\n✓ 演示完成！")

if __name__ == "__main__":
    main()
