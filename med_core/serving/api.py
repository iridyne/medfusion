"""FastAPI 模型服务"""

import torch
import torch.nn as nn


class ModelServer:
    """模型服务器"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def predict(self, data: list[float]):
        with torch.no_grad():
            x = torch.tensor(data).unsqueeze(0)
            output = self.model(x)
            return output.squeeze().tolist()


def create_server(model: nn.Module):
    return ModelServer(model)
