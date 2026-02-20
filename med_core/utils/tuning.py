"""
自动超参数调优

使用 Optuna 进行超参数优化。
"""

import logging
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    超参数调优器
    
    使用 Optuna 进行贝叶斯优化。
    
    Args:
        objective_fn: 目标函数
        direction: 优化方向 ("minimize" 或 "maximize")
        study_name: 研究名称
        storage: 存储位置
        
    Example:
        >>> def objective(trial):
        ...     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        ...     return train_and_evaluate(lr)
        >>> tuner = HyperparameterTuner(objective, direction="maximize")
        >>> best_params = tuner.optimize(n_trials=100)
    """
    
    def __init__(
        self,
        objective_fn: Callable,
        direction: str = "maximize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is not installed. "
                "Install it with: pip install optuna"
            )
        
        self.objective_fn = objective_fn
        self.direction = direction
        self.study_name = study_name or "hyperparameter_optimization"
        self.storage = storage
        
        # 创建研究
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True,
        )
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
    ) -> dict[str, Any]:
        """
        运行优化
        
        Args:
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行作业数
            show_progress_bar: 是否显示进度条
            
        Returns:
            最佳参数字典
        """
        logger.info(f"Starting hyperparameter optimization: {self.study_name}")
        logger.info(f"  Direction: {self.direction}")
        logger.info(f"  Trials: {n_trials}")
        logger.info(f"  Jobs: {n_jobs}")

        self.study.optimize(
            self.objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        logger.info(f"\n✓ Optimization completed!")
        logger.info(f"  Best value: {self.study.best_value:.4f}")
        logger.info(f"  Best params: {self.study.best_params}")

        return self.study.best_params
    
    def get_best_trial(self):
        """获取最佳试验"""
        return self.study.best_trial
    
    def get_trials_dataframe(self):
        """获取试验数据框"""
        return self.study.trials_dataframe()
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """绘制优化历史"""
        try:
            import optuna.visualization as vis
        except ImportError:
            logger.warning("Optuna visualization not available")
            return

        fig = vis.plot_optimization_history(self.study)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved optimization history to {save_path}")
        else:
            fig.show()

    def plot_param_importances(self, save_path: Optional[str] = None):
        """绘制参数重要性"""
        try:
            import optuna.visualization as vis
        except ImportError:
            logger.warning("Optuna visualization not available")
            return

        fig = vis.plot_param_importances(self.study)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved parameter importances to {save_path}")
        else:
            fig.show()

    def plot_parallel_coordinate(self, save_path: Optional[str] = None):
        """绘制平行坐标图"""
        try:
            import optuna.visualization as vis
        except ImportError:
            logger.warning("Optuna visualization not available")
            return

        fig = vis.plot_parallel_coordinate(self.study)

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved parallel coordinate plot to {save_path}")
        else:
            fig.show()


class SearchSpace:
    """
    搜索空间定义
    
    提供常用的超参数搜索空间。
    
    Example:
        >>> space = SearchSpace()
        >>> space.add_float("lr", 1e-5, 1e-1, log=True)
        >>> space.add_int("batch_size", 16, 128, step=16)
        >>> space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    """
    
    def __init__(self):
        self.params = {}
    
    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
    ):
        """添加浮点参数"""
        self.params[name] = {
            "type": "float",
            "low": low,
            "high": high,
            "log": log,
            "step": step,
        }
        return self
    
    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
    ):
        """添加整数参数"""
        self.params[name] = {
            "type": "int",
            "low": low,
            "high": high,
            "step": step,
            "log": log,
        }
        return self
    
    def add_categorical(self, name: str, choices: list):
        """添加分类参数"""
        self.params[name] = {
            "type": "categorical",
            "choices": choices,
        }
        return self
    
    def suggest(self, trial):
        """从试验中建议参数"""
        suggested = {}
        
        for name, config in self.params.items():
            if config["type"] == "float":
                suggested[name] = trial.suggest_float(
                    name,
                    config["low"],
                    config["high"],
                    log=config["log"],
                    step=config.get("step"),
                )
            elif config["type"] == "int":
                suggested[name] = trial.suggest_int(
                    name,
                    config["low"],
                    config["high"],
                    step=config["step"],
                    log=config["log"],
                )
            elif config["type"] == "categorical":
                suggested[name] = trial.suggest_categorical(
                    name,
                    config["choices"],
                )
        
        return suggested


def create_default_search_space() -> SearchSpace:
    """
    创建默认的搜索空间
    
    Returns:
        默认搜索空间
    """
    space = SearchSpace()
    
    # 学习率
    space.add_float("lr", 1e-5, 1e-1, log=True)
    
    # 批次大小
    space.add_int("batch_size", 16, 128, step=16)
    
    # 优化器
    space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    
    # 权重衰减
    space.add_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Dropout
    space.add_float("dropout", 0.0, 0.5, step=0.1)
    
    return space


class ModelTuner:
    """
    模型调优器
    
    简化模型超参数调优的流程。
    
    Args:
        model_fn: 模型创建函数
        train_fn: 训练函数
        eval_fn: 评估函数
        search_space: 搜索空间
        
    Example:
        >>> def model_fn(params):
        ...     return MyModel(dropout=params["dropout"])
        >>> def train_fn(model, params):
        ...     # 训练逻辑
        ...     pass
        >>> def eval_fn(model):
        ...     # 评估逻辑
        ...     return accuracy
        >>> tuner = ModelTuner(model_fn, train_fn, eval_fn)
        >>> best_params = tuner.tune(n_trials=50)
    """
    
    def __init__(
        self,
        model_fn: Callable,
        train_fn: Callable,
        eval_fn: Callable,
        search_space: Optional[SearchSpace] = None,
        direction: str = "maximize",
    ):
        self.model_fn = model_fn
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.search_space = search_space or create_default_search_space()
        self.direction = direction
    
    def objective(self, trial):
        """目标函数"""
        # 建议参数
        params = self.search_space.suggest(trial)
        
        # 创建模型
        model = self.model_fn(params)
        
        # 训练模型
        self.train_fn(model, params)
        
        # 评估模型
        score = self.eval_fn(model)
        
        return score
    
    def tune(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        study_name: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        运行调优
        
        Args:
            n_trials: 试验次数
            timeout: 超时时间
            study_name: 研究名称
            **kwargs: 传递给 optimize 的额外参数
            
        Returns:
            最佳参数
        """
        tuner = HyperparameterTuner(
            self.objective,
            direction=self.direction,
            study_name=study_name,
        )
        
        best_params = tuner.optimize(
            n_trials=n_trials,
            timeout=timeout,
            **kwargs,
        )
        
        self.tuner = tuner
        
        return best_params
    
    def plot_results(self, output_dir: str = "outputs/tuning"):
        """绘制结果"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, "tuner"):
            self.tuner.plot_optimization_history(
                f"{output_dir}/optimization_history.html"
            )
            self.tuner.plot_param_importances(
                f"{output_dir}/param_importances.html"
            )
            self.tuner.plot_parallel_coordinate(
                f"{output_dir}/parallel_coordinate.html"
            )


def tune_hyperparameters(
    objective_fn: Callable,
    search_space: Optional[SearchSpace] = None,
    n_trials: int = 100,
    direction: str = "maximize",
    study_name: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    超参数调优的便捷函数
    
    Args:
        objective_fn: 目标函数
        search_space: 搜索空间
        n_trials: 试验次数
        direction: 优化方向
        study_name: 研究名称
        **kwargs: 额外参数
        
    Returns:
        最佳参数
        
    Example:
        >>> def objective(trial):
        ...     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        ...     return train_and_evaluate(lr)
        >>> best_params = tune_hyperparameters(objective, n_trials=50)
    """
    tuner = HyperparameterTuner(
        objective_fn,
        direction=direction,
        study_name=study_name,
    )
    
    return tuner.optimize(n_trials=n_trials, **kwargs)
