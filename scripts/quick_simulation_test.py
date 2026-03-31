"""
快速模拟测试脚本 - 使用合成数据测试 MedFusion 完整流程

目标：
1. 生成合成的医学影像数据（模拟肺炎检测）
2. 配置并训练模型
3. 生成评估报告
4. 记录整个过程的问题和耗时

测试场景：
- 数据：500 张合成 X 光图片 + 患者信息（年龄、性别）
- 任务：二分类（NORMAL vs PNEUMONIA）
- 模型：ResNet18 + 简单融合
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SIMULATION_OUTPUT_DIR = "artifacts/dev/simulation_test/run"
SIMULATION_RESULTS_PATH = Path("artifacts/dev/simulation_test/simulation_test_results.json")

# 记录测试结果
test_results = {
    "stage_times": {},
    "issues": [],
    "features_used": [],
    "features_not_used": [],
}


def log_stage(stage_name):
    """装饰器：记录每个阶段的耗时"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"开始阶段: {stage_name}")
            logger.info(f"{'=' * 60}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                test_results["stage_times"][stage_name] = elapsed
                logger.info(f"✅ {stage_name} 完成，耗时: {elapsed:.2f} 秒")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                test_results["stage_times"][stage_name] = elapsed
                test_results["issues"].append(
                    {
                        "stage": stage_name,
                        "error": str(e),
                        "time": elapsed,
                    }
                )
                logger.error(f"❌ {stage_name} 失败: {e}")
                raise

        return wrapper

    return decorator


@log_stage("阶段 1: 生成合成数据")
def generate_synthetic_data():
    """生成合成的医学影像数据"""
    logger.info("生成 500 张合成 X 光图片...")

    # 创建数据目录
    data_dir = Path("data/simulation_test")
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 生成图片和标签
    data_records = []

    for i in range(500):
        # 生成随机图片（模拟 X 光）
        if i % 2 == 0:
            # NORMAL: 较亮的图片
            img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            label = 0
            label_name = "NORMAL"
        else:
            # PNEUMONIA: 较暗的图片（模拟肺部阴影）
            img_array = np.random.randint(50, 120, (224, 224, 3), dtype=np.uint8)
            label = 1
            label_name = "PNEUMONIA"

        # 保存图片
        img_path = images_dir / f"patient_{i:04d}.png"
        img = Image.fromarray(img_array)
        img.save(img_path)

        # 生成患者信息
        data_records.append(
            {
                "patient_id": f"P{i:04d}",
                "image_path": str(img_path.relative_to(data_dir)),
                "age": np.random.randint(20, 80),
                "sex": np.random.choice(["M", "F"]),
                "diagnosis": label,
                "diagnosis_name": label_name,
            }
        )

    # 保存 CSV
    df = pd.DataFrame(data_records)
    csv_path = data_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"✓ 生成了 {len(df)} 条数据")
    logger.info(f"✓ 图片保存在: {images_dir}")
    logger.info(f"✓ CSV 保存在: {csv_path}")
    logger.info(f"✓ 类别分布: {df['diagnosis_name'].value_counts().to_dict()}")

    test_results["features_used"].append("数据生成")

    return data_dir, csv_path


@log_stage("阶段 2: 配置模型")
def configure_model():
    """创建配置文件"""
    logger.info("创建训练配置...")

    config_content = """# 模拟测试配置
project_name: "simulation-test"
experiment_name: "pneumonia_detection_test"
description: "快速测试 MedFusion 完整流程"

seed: 42
deterministic: true
device: "auto"

# 数据配置
data:
  data_root: "data/simulation_test"
  csv_path: "data/simulation_test/dataset.csv"
  image_dir: "data/simulation_test"

  image_path_column: "image_path"
  target_column: "diagnosis"
  patient_id_column: "patient_id"

  numerical_features:
    - "age"
  categorical_features:
    - "sex"

  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

  image_size: 224
  image_channels: 3
  batch_size: 16
  num_workers: 2
  pin_memory: true
  augmentation_strength: "light"

# 模型配置
model:
  num_classes: 2

  vision:
    backbone: "resnet18"
    pretrained: true
    freeze_backbone: false
    feature_dim: 128
    dropout: 0.3
    attention_type: "none"

  tabular:
    hidden_dims: [32, 32]
    output_dim: 16
    dropout: 0.2
    use_batch_norm: true

  fusion:
    fusion_type: "concatenate"
    hidden_dim: 64
    dropout: 0.3

# 训练配置（快速测试）
training:
  num_epochs: 3
  mixed_precision: true
  gradient_clip: 1.0

  use_progressive_training: false

  early_stopping: false

  save_top_k: 1
  save_last: true

  optimizer:
    optimizer: "adam"
    learning_rate: 1.0e-3
    weight_decay: 0.01

  scheduler:
    scheduler: "step"
    step_size: 2
    gamma: 0.1

# 日志配置
logging:
  output_dir: "artifacts/dev/simulation_test/run"
  use_tensorboard: false
  use_wandb: false
  log_every_n_steps: 5
  save_visualizations: true
"""

    config_path = Path("configs/testing/simulation_test.yaml")
    config_path.write_text(config_content)

    logger.info(f"✓ 配置文件保存在: {config_path}")
    logger.info("✓ 使用 ResNet18 + 简单拼接融合")
    logger.info("✓ 训练 3 个 epoch（快速测试）")

    test_results["features_used"].append("YAML 配置")

    return config_path


@log_stage("阶段 3: 训练模型")
def train_model(config_path):
    """使用 MedFusion 训练模型"""
    logger.info("开始训练模型...")
    logger.info("注意：这是快速测试，只训练 3 个 epoch")

    try:
        # 尝试导入 MedFusion
        from med_core.configs import ExperimentConfig
        from med_core.datasets import MedicalMultimodalDataset, create_dataloaders
        from med_core.models import create_multimodal_model
        from med_core.trainers import MultimodalTrainer

        logger.info("✓ MedFusion 模块导入成功")
        test_results["features_used"].append("MedFusion 核心模块")

        # 加载配置
        config = ExperimentConfig.from_yaml(config_path)
        logger.info("✓ 配置加载成功")

        # 创建数据集
        logger.info("创建数据集...")
        # 这里需要实际实现数据加载逻辑
        # 由于我们不确定具体 API，先记录这一步

        logger.info("⚠️  需要查看实际的数据加载 API")
        test_results["issues"].append(
            {
                "stage": "训练模型",
                "error": "需要确认数据加载 API",
                "severity": "medium",
            }
        )

        return None

    except ImportError as e:
        logger.error(f"导入失败: {e}")
        test_results["issues"].append(
            {
                "stage": "训练模型",
                "error": f"导入失败: {e}",
                "severity": "high",
            }
        )
        raise


@log_stage("阶段 4: 评估和报告")
def evaluate_and_report():
    """生成评估报告"""
    logger.info("生成评估报告...")

    # 这一步取决于训练是否成功
    logger.info("⚠️  等待训练完成后生成报告")

    return None


def print_test_summary():
    """打印测试总结"""
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)

    # 耗时统计
    logger.info("\n⏱️  各阶段耗时:")
    total_time = 0
    for stage, elapsed in test_results["stage_times"].items():
        logger.info(f"  {stage}: {elapsed:.2f} 秒")
        total_time += elapsed
    logger.info(f"  总计: {total_time:.2f} 秒")

    # 问题统计
    logger.info(f"\n⚠️  发现 {len(test_results['issues'])} 个问题:")
    for i, issue in enumerate(test_results["issues"], 1):
        logger.info(f"  {i}. [{issue.get('severity', 'unknown')}] {issue['stage']}")
        logger.info(f"     {issue['error']}")

    # 功能使用统计
    logger.info(f"\n✅ 使用的功能 ({len(test_results['features_used'])}):")
    for feature in test_results["features_used"]:
        logger.info(f"  - {feature}")

    logger.info("\n" + "=" * 60)


def main():
    """主测试流程"""
    logger.info("🚀 开始 MedFusion 模拟测试")
    logger.info("目标：验证完整流程是否能跑通\n")

    try:
        # 阶段 1: 生成数据
        data_dir, csv_path = generate_synthetic_data()

        # 阶段 2: 配置模型
        config_path = configure_model()

        # 阶段 3: 训练模型
        train_model(config_path)

        # 阶段 4: 评估和报告
        evaluate_and_report()

    except Exception as e:
        logger.error(f"\n❌ 测试中断: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 打印总结
        print_test_summary()

        # 保存测试结果
        import json

        results_path = SIMULATION_RESULTS_PATH
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"\n📄 测试结果已保存: {results_path}")


if __name__ == "__main__":
    main()
