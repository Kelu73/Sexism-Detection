import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F  # 👈 1. 新增：引入用於計算 Softmax 機率的模組
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path

# 動態把目前檔案所在的 code 資料夾加入 Python 的搜尋路徑中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

app = FastAPI()

# 允許前端網頁連線的安全設定 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 動態取得模型絕對路徑
MODEL_PATH = Path(__file__).parent.parent / "bert_sexism_model"
MODEL_PATH = MODEL_PATH.resolve()

print(f"正在從此路徑載入模型: {MODEL_PATH}")

tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = BertForSequenceClassification.from_pretrained(
    str(MODEL_PATH), local_files_only=True
)


class SexismRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: SexismRequest):
    inputs = tokenizer(
        request.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    #  修改：利用 Softmax 將 Logits 分數轉換成 0~1 之間的機率值
    probabilities = F.softmax(outputs.logits, dim=1)

    # 找出機率最高的那一個類別的分數（信心值）與索引
    confidence, prediction_idx = torch.max(probabilities, dim=1)

    prediction = prediction_idx.item()
    confidence_score = confidence.item() * 100  # 轉為百分比 (例如 0.855 轉成 85.5)

    status = "Sexist" if prediction == 1 else "Not Sexist"

    # 👈 3. 修改：在回傳的 JSON 格式中加入 "confidence" 欄位
    return {
        "prediction": status,
        "confidence": round(confidence_score, 2),  # 四捨五入取到小數點後兩位
        "text": request.text,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
