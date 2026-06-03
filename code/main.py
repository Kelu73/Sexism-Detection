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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 💡 方案 A：偏見核心詞彙歸一化函數（只保留這一個正確的） ───────────────────
def align_bias_keywords(text: str) -> str:
    """
    將英文文本中的變體弱勢稱呼（如 girls, female），預先對齊到 BERT 模型最敏感的核心詞（women）。
    防止模型因訓練資料集的樣本偏差，而對同義詞產生泛化漏洞（例如能認出 women can't，卻認不出 girls can't）。
    """
    mapping = {
        r"\bgirls\b": "women",
        r"\bgirl\b": "women",
        r"\bfemale\b": "women",
        r"\bfemales\b": "women",
        r"\blady\b": "women",
        r"\bladies\b": "women",
        r"\bboys\b": "men",
        r"\bboy\b": "men",
    }

    aligned_text = text
    # 使用 re.sub 逐一進行不區分大小寫的語意替換
    for pattern, replacement in mapping.items():
        aligned_text = re.sub(pattern, replacement, aligned_text, flags=re.IGNORECASE)

    return aligned_text


# ── 1. 定義並載入兩個模型的路徑 ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()

# Task A: 二元分類模型 (Sexist / Not Sexist)
MODEL_A_PATH = BASE_DIR / "bert_sexism_model"
# Task B: 四類細分模型 (Threats, Derogation, Animosity, Prejudiced)
MODEL_B_PATH = BASE_DIR / "bert_sexism_category_model"

print(f"【載入 Task A 模型】: {MODEL_A_PATH}")
tokenizer_a = BertTokenizer.from_pretrained(str(MODEL_A_PATH), local_files_only=True)
model_a = BertForSequenceClassification.from_pretrained(
    str(MODEL_A_PATH), local_files_only=True
)

# 定義 Task B 的類別對應表 (根據 EDOS 官方定義)
TASK_B_CLASSES = {
    0: "1. threats, plans to harm and incitement",
    1: "2. derogation",
    2: "3. animosity",
    3: "4. prejudiced discussion",
}

# 檢查是否有 Task B 模型
has_task_b = MODEL_B_PATH.exists()
if has_task_b:
    print(f"【載入 Task B 模型】: {MODEL_B_PATH}")
    tokenizer_b = BertTokenizer.from_pretrained(
        str(MODEL_B_PATH), local_files_only=True
    )

    try:
        model_b = BertForSequenceClassification.from_pretrained(
            str(MODEL_B_PATH), local_files_only=True
        )
    except Exception as e:
        print("偵測到新版 PyTorch 權重相容性問題，正在切換至相容模式載入...")
        import builtins

        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(
            *args, **{**kwargs, "weights_only": False}
        )

        model_b = BertForSequenceClassification.from_pretrained(
            str(MODEL_B_PATH), local_files_only=True
        )
        torch.load = original_load
else:
    print(
        "⚠️ 【警告】未偵測到 bert_sexism_category_model 資料夾，Task B 功能將暫時以模擬文字輸出"
    )

# 初始化外部工具
cc = OpenCC("t2s")
kw_model = KeyBERT()
translator = GoogleTranslator(source="zh-TW", target="en")


class SexismRequest(BaseModel):
    text: str


def is_contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def translate_chinese(text: str) -> str:
    try:
        simplified = cc.convert(text)
        translated = translator.translate(simplified)
        print(f"【翻譯結果】: {text} ➡️ {translated}")
        return translated or ""
    except Exception as e:
        print(f"【翻譯失敗】: {e}")
        return ""


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    try:
        results = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5,
        )
        keywords = [kw for kw, _ in results]
        print(f"【關鍵字抽取】: {keywords}")
        return keywords
    except Exception as e:
        print(f"【關鍵字抽取失敗】: {e}")
        return []


# ── 🚀 完整升級後的推論路由 ───────────────────────────────────────
@app.post("/predict")
async def predict(request: SexismRequest):
    user_input = request.text
    is_chinese = is_contains_chinese(user_input)
    keywords: list[str] = []

    # ── 多語言處理層 ───────────────────────────────────────────────
    if is_chinese:
        english_text = translate_chinese(user_input)
        if not english_text.strip():
            english_text = "women"
            print("【警告】翻譯結果為空，使用預設提示詞")
    else:
        english_text = user_input
        print(f"【英文原生輸入】: {user_input}")

    # ── 🌟 方案 A 核心機制：預處理詞彙歸一化層 ────────────────────────
    aligned_english_text = align_bias_keywords(english_text)

    if aligned_english_text != english_text:
        print(
            f'【偏見詞彙對齊層啟動】: "{english_text}" ➡️ 自動歸一化為 "{aligned_english_text}"'
        )

    # 關鍵字抽取
    keywords = extract_keywords(aligned_english_text)

    # ── Task A: 判斷是否有性別歧視 ──────────────────────────────────
    inputs_a = tokenizer_a(
        aligned_english_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs_a = model_a(**inputs_a)

    probs_a = F.softmax(outputs_a.logits, dim=1)
    conf_a, pred_idx_a = torch.max(probs_a, dim=1)

    is_sexist = pred_idx_a.item() == 1
    status = "Sexist" if is_sexist else "Not Sexist"
    confidence_score_a = round(conf_a.item() * 100, 2)

    # ── Task B: 若為 Sexist，進一步細分類型 ──────────────────────────
    category_result = "N/A (Not Sexist)"
    confidence_score_b = 0.0

    if is_sexist:
        if has_task_b:
            inputs_b = tokenizer_b(
                aligned_english_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs_b = model_b(**inputs_b)

            probs_b = F.softmax(outputs_b.logits, dim=1)
            conf_b, pred_idx_b = torch.max(probs_b, dim=1)

            category_result = TASK_B_CLASSES.get(pred_idx_b.item(), "Unknown")
            confidence_score_b = round(conf_b.item() * 100, 2)
        else:
            category_result = "2. derogation (Mocked Result)"
            confidence_score_b = 85.50

    return {
        "prediction": status,
        "confidence": confidence_score_a,
        "category": category_result,
        "category_confidence": confidence_score_b,
        "translation": (
            aligned_english_text
            if (is_chinese or aligned_english_text != english_text)
            else "Native English Input"
        ),
        "keywords": keywords,
        "text": user_input,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
