from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict
import torch
from network import Network  # 确保 network.py 在相同目录或已加入 PYTHONPATH

app = FastAPI()

# 模型配置类
class ModelConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def get_config(dataset: str):
        configs = {
            "bdgp": {
                "view": 2,
                "dims": [1750, 79],
                "feature_dim": 512,
                "high_feature_dim": 512,
                "class_num": 5
            },
            "mnist": {
                "view": 2,
                "dims": [784, 784],
                "feature_dim": 512,
                "high_feature_dim": 512,
                "class_num": 10
            },
            "fashion": {
                "view": 2,
                "dims": [784, 784],
                "feature_dim": 1024,
                "high_feature_dim": 1024,
                "class_num": 10
            },
            "handwritten": {
                "view": 2,
                "dims": [240, 76],
                "feature_dim": 512,
                "high_feature_dim": 512,
                "class_num": 10
            }
        }

        if dataset not in configs:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return configs[dataset]


# 模型路径映射
model_paths = {
    "bdgp": "models/bdgp_model.pth",
    "mnist": "models/mnist_model.pth",
    "fashion": "models/fashion_model.pth",
    "handwritten": "models/handwritten_model.pth"
}

# 缓存已加载的模型
loaded_models = {}


def load_model(dataset: str):
    config = ModelConfig.get_config(dataset)
    model = Network(config["view"], config["dims"],
                    config["feature_dim"], config["high_feature_dim"],
                    config["class_num"], ModelConfig.device)

    model_path = model_paths[dataset]
    model.load_state_dict(torch.load(model_path, map_location=ModelConfig.device))
    model.to(ModelConfig.device)
    model.eval()
    return model


def get_model(dataset: str):
    if dataset not in loaded_models:
        loaded_models[dataset] = load_model(dataset)
    return loaded_models[dataset]


# 输入数据格式定义
class InputData(BaseModel):
    view1: List[float]
    view2: List[float]


# 预测接口
@app.post("/predict")
async def predict(data: InputData, dataset: str = Query("bdgp")):
    model = get_model(dataset)
    config = ModelConfig.get_config(dataset)

    # 转换为 tensor
    view1_tensor = torch.tensor(data.view1).float().to(ModelConfig.device)
    view2_tensor = torch.tensor(data.view2).float().to(ModelConfig.device)

    with torch.no_grad():
        _, qs, _, _, _ = model([view1_tensor.unsqueeze(0), view2_tensor.unsqueeze(0)])
        confi1, y_pred1 = qs[0].topk(k=1, dim=1)
        y_pred1 = y_pred1.cpu().numpy()
        confi2, y_pred2 = qs[1].topk(k=1, dim=1)
        y_pred2 = y_pred2.cpu().numpy()

    return {
        "dataset": dataset,
        "prediction_view1": int(y_pred1[0][0]),
        "prediction_view2": int(y_pred2[0][0])
    }


# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8085)