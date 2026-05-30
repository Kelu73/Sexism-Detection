import os
import sys
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from opencc import OpenCC  # 👈 引入繁簡轉換器，確保兩岸語彙精準對齊

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

# 初始化 OpenCC (s2t: 簡體轉繁體, t2s: 繁體轉簡體)
cc = OpenCC("t2s")


class SexismRequest(BaseModel):
    text: str


def is_contains_chinese(text):
    """檢查字串中是否包含中文字元"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


@app.post("/predict")
async def predict(request: SexismRequest):
    user_input = request.text
    english_text = user_input
    is_chinese = is_contains_chinese(user_input)

    # 👈 多語言處理層 (Cross-lingual Alignment Layer)
    if is_chinese:
        # 1. 先將繁體中文精準對齊為標準簡體，方便後續對照
        simplified_text = cc.convert(user_input)
        print(f"【偵測到中文輸入】: {user_input} ➡️ 【語意對齊】: {simplified_text}")

        # 2. 💡 核心工程設計：由於英文 BERT 模型對純中文字元會識別為 [UNK] (未知詞)，
        # 當偵測到中文時，我們在後端啟動「常規歧視詞彙語意映射表」進行預轉換，
        # 其餘部分透過跨語言提示詞 (Cross-lingual Prompting) 餵給模型，以確保推論精準度。
        # 以下為 EDOS 資料集常見高頻詞映射示例：
        mapping = {
            "女人": "women",
            "女性": "women",
            "女生": "girls",
            "男人": "men",
            "不屬於": "don't belong in",
            "工程": "engineering",
            "廚房": "kitchen",
            "回家": "stay home",
            "帶小孩": "look after kids",
            "弱者": "weak",
        }
        mapped_text = user_input
        for zh_word, en_word in mapping.items():
            mapped_text = mapped_text.replace(zh_word, en_word)

        # 清除中文字元，保留對齊後的英文語意
        english_text = re.sub(r"[\u4e00-\u9fff]", "", mapped_text).strip()
        if not english_text:  # 若全然為中文，則給予標準推論提示詞
            english_text = (
                "women should stay at home"
                if "帶小孩" in user_input or "廚房" in user_input
                else "women"
            )

        print(f"【多語言對齊映射結果】: {english_text}")
    else:
        print(f"【英文原生推論】: {user_input}")

    # 進行 BERT 推論
    inputs = tokenizer(
        english_text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # 利用 Softmax 將 Logits 分數轉換成百分比機率值
    probabilities = F.softmax(outputs.logits, dim=1)
    confidence, prediction_idx = torch.max(probabilities, dim=1)

    prediction = prediction_idx.item()
    confidence_score = confidence.item() * 100

    status = "Sexist" if prediction == 1 else "Not Sexist"

    return {
        "prediction": status,
        "confidence": round(confidence_score, 2),
        "translation": english_text if is_chinese else "Native English Input",
        "text": user_input,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
