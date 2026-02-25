"""监控指标"""


class MetricsCollector:
    """指标收集器"""

    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {}

    def record(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_metrics(self) -> dict[str, list[float]]:
        return self.metrics


collector = MetricsCollector()
