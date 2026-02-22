"""监控指标"""


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_metrics(self) -> dict:
        return self.metrics


collector = MetricsCollector()
