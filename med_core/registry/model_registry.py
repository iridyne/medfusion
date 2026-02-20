"""模型注册表"""
from pathlib import Path
from typing import Dict, Optional
import json

class ModelRegistry:
    """模型注册表"""
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = self._load()
    
    def _load(self) -> Dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}
    
    def _save(self):
        self.registry_path.write_text(json.dumps(self.models, indent=2))
    
    def register(self, name: str, version: str, path: str, metadata: Optional[Dict] = None):
        key = f"{name}:{version}"
        self.models[key] = {"path": path, "metadata": metadata or {}}
        self._save()
    
    def get(self, name: str, version: str = "latest") -> Optional[Dict]:
        key = f"{name}:{version}"
        return self.models.get(key)
    
    def list_models(self):
        return list(self.models.keys())

registry = ModelRegistry()
