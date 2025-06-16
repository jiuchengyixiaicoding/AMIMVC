import requests
import json

url = "http://127.0.0.1:8085/predict?dataset=fashion"

# 自动生成符合维度的数据
data = {
    # "view1": [0.0] * 1750, # bdgp
    # "view2": [0.0] * 79
    # "view1": [0.0] * 784, # mnist
    # "view2": [0.0] * 784
    # "view1": [0.0] * 240, # handwritten
    # "view2": [0.0] * 76
    "view1": [0.0] * 784,# fashion
    "view2": [0.0] * 784
}

response = requests.post(url, data=json.dumps(data))
print(response.status_code)
print(response.json())