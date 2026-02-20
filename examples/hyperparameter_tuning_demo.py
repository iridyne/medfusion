"""
超参数调优示例

演示如何使用 Optuna 进行超参数优化。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_model(params):
    """创建模型"""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(params.get("dropout", 0.2)),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(params.get("dropout", 0.2)),
        nn.Linear(64, 2),
    )


def train_and_evaluate(params):
    """训练和评估"""
    # 创建数据
    X_train = torch.randn(1000, 10)
    y_train = torch.randint(0, 2, (1000,))
    X_val = torch.randn(200, 10)
    y_val = torch.randint(0, 2, (200,))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # 创建模型
    model = create_model(params)

    # 优化器
    if params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    elif params["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    criterion = nn.CrossEntropyLoss()

    # 训练
    model.train()
    for _epoch in range(5):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    return accuracy


def demo_basic_tuning():
    """基本调优示例"""
    print("=" * 60)
    print("基本超参数调优")
    print("=" * 60)

    from med_core.utils.tuning import tune_hyperparameters

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
        return train_and_evaluate(params)

    best_params = tune_hyperparameters(
        objective,
        n_trials=20,
        direction="maximize",
        study_name="basic_tuning",
    )

    print(f"\n最佳参数: {best_params}")


def demo_search_space():
    """搜索空间示例"""
    print("\n" + "=" * 60)
    print("使用搜索空间")
    print("=" * 60)

    from med_core.utils.tuning import HyperparameterTuner, SearchSpace

    # 定义搜索空间
    space = SearchSpace()
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_int("batch_size", 16, 128, step=16)
    space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    space.add_float("weight_decay", 1e-6, 1e-2, log=True)
    space.add_float("dropout", 0.0, 0.5, step=0.1)

    def objective(trial):
        params = space.suggest(trial)
        return train_and_evaluate(params)

    tuner = HyperparameterTuner(objective, direction="maximize")
    best_params = tuner.optimize(n_trials=20)

    print(f"\n最佳参数: {best_params}")


def demo_model_tuner():
    """模型调优器示例"""
    print("\n" + "=" * 60)
    print("使用模型调优器")
    print("=" * 60)

    from med_core.utils.tuning import ModelTuner, SearchSpace

    # 定义搜索空间
    space = SearchSpace()
    space.add_float("lr", 1e-4, 1e-2, log=True)
    space.add_int("batch_size", 32, 64, step=16)
    space.add_float("dropout", 0.1, 0.3, step=0.1)

    def model_fn(params):
        return create_model(params)

    def train_fn(model, params):
        # 简化的训练
        pass

    def eval_fn(model):
        # 简化的评估
        return 0.85

    tuner = ModelTuner(model_fn, train_fn, eval_fn, search_space=space)
    best_params = tuner.tune(n_trials=10)

    print(f"\n最佳参数: {best_params}")


def main():
    print("\n" + "=" * 60)
    print("MedFusion 超参数调优演示")
    print("=" * 60)

    try:
        # 演示 1: 基本调优
        demo_basic_tuning()

        # 演示 2: 搜索空间
        demo_search_space()

        # 演示 3: 模型调优器
        demo_model_tuner()

        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)

    except ImportError as e:
        print(f"\n⚠ 错误: {e}")
        print("请安装 Optuna: pip install optuna")


if __name__ == "__main__":
    main()
