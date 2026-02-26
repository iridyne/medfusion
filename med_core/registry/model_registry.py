"""模型注册表"""

import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    """模型注册表"""

    def __init__(self, registry_path: str = "models/registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = self._load()

    def _load(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}

    def _save(self) -> None:
        self.registry_path.write_text(json.dumps(self.models, indent=2))

    def register(
        self, name: str, version: str, path: str, metadata: dict[str, Any] | None = None,
    ) -> None:
        key = f"{name}:{version}"
        self.models[key] = {"path": path, "metadata": metadata or {}}
        self._save()

    def get(self, name: str, version: str = "latest") -> dict[str, Any] | None:
        key = f"{name}:{version}"
        return self.models.get(key)

    def list_models(self) -> list[str]:
        return list(self.models.keys())


registry = ModelRegistry()
