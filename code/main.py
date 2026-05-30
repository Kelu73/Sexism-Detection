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
from opencc import OpenCC
from deep_translator import GoogleTranslator
from keybert import KeyBERT

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

# 初始化 OpenCC (t2s: 繁體轉簡體，方便 Google Translate 辨識)
cc = OpenCC("t2s")

# 初始化 KeyBERT（用 sentence-transformers 做關鍵字抽取）
kw_model = KeyBERT()

# 初始化 Google Translator（zh-TW → en）
translator = GoogleTranslator(source="zh-TW", target="en")


class SexismRequest(BaseModel):
    text: str


def is_contains_chinese(text: str) -> bool:
    """檢查字串中是否包含中文字元"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def translate_chinese(text: str) -> str:
    """
    完整翻譯中文文字為英文。
    流程：繁體 → 簡體（OpenCC）→ Google Translate → 英文
    若翻譯失敗，回傳空字串。
    """
    try:
        simplified = cc.convert(text)
        translated = translator.translate(simplified)
        print(f"【翻譯結果】: {text} ➡️ {translated}")
        return translated or ""
    except Exception as e:
        print(f"【翻譯失敗】: {e}")
        return ""


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    從英文文字中抽取 top_n 個關鍵詞，
    回傳純字串 list，例如 ['women', 'kitchen', 'belong']
    """
    try:
        results = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),  # 支援單詞與雙詞片語
            stop_words="english",
            top_n=top_n,
            use_mmr=True,          # Maximal Marginal Relevance，減少關鍵詞重複
            diversity=0.5,
        )
        keywords = [kw for kw, _ in results]
        print(f"【關鍵字抽取】: {keywords}")
        return keywords
    except Exception as e:
        print(f"【關鍵字抽取失敗】: {e}")
        return []


@app.post("/predict")
async def predict(request: SexismRequest):
    user_input = request.text
    is_chinese = is_contains_chinese(user_input)
    keywords: list[str] = []

    # ── 多語言處理層 ───────────────────────────────────────────────
    if is_chinese:
        # Step 1：完整翻譯（取代舊版字典映射）
        english_text = translate_chinese(user_input)

        # Step 2：若翻譯結果為空，給予保守預設值避免空字串送入 BERT
        if not english_text.strip():
            english_text = "women"
            print("【警告】翻譯結果為空，使用預設提示詞")

        # Step 3：從翻譯後的英文抽取關鍵字
        keywords = extract_keywords(english_text)

    else:
        english_text = user_input
        # 英文輸入也做關鍵字抽取，方便前端顯示
        keywords = extract_keywords(english_text)
        print(f"【英文原生推論】: {user_input}")

    # ── BERT 推論 ──────────────────────────────────────────────────
    inputs = tokenizer(
        english_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax → 百分比機率
    probabilities = F.softmax(outputs.logits, dim=1)
    confidence, prediction_idx = torch.max(probabilities, dim=1)

    prediction = prediction_idx.item()
    confidence_score = confidence.item() * 100
    status = "Sexist" if prediction == 1 else "Not Sexist"

    return {
        "prediction": status,
        "confidence": round(confidence_score, 2),
        "translation": english_text if is_chinese else "Native English Input",
        "keywords": keywords,   # 新增：回傳關鍵字供前端顯示
        "text": user_input,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)