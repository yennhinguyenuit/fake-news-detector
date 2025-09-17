import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, joblib
from duckduckgo_search import DDGS
import trafilatura

# --- Load đúng tên file model m đã lưu ---
VEC_PATH = os.path.join("models", "tfidf.joblib")
CLF_PATH = os.path.join("models", "lr_fake.joblib")
vec = joblib.load(VEC_PATH)
clf = joblib.load(CLF_PATH)

def analyze(text: str):
    """
    Dự đoán xác suất fake/real.
    Trả về:
      - label: "FAKE" hoặc "REAL"
      - confidence: max(prob_fake, prob_real)
      - probs: dict {'fake': p_fake, 'real': p_real}
    """
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]  # [p(class=0), p(class=1)]
    # Trong train.py: y=1 nếu label=='fake', y=0 nếu real
    p_real = float(proba[0])
    p_fake = float(proba[1])
    label = "FAKE" if p_fake >= p_real else "REAL"
    confidence = max(p_fake, p_real)
    return label, confidence, {"fake": p_fake, "real": p_real}

def fact_check(query: str, max_results: int = 3):
    """
    Tìm 3–N nguồn liên quan, trích đoạn nội dung bằng trafilatura.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url")
            title = r.get("title") or "(No title)"
            snippet = r.get("body") or ""
            text = None
            if url:
                try:
                    downloaded = trafilatura.fetch_url(url, no_ssl=True)
                    if downloaded:
                        text = trafilatura.extract(
                            downloaded,
                            include_comments=False,
                            include_tables=False
                        )
                except Exception:
                    pass
            results.append({
                "title": title,
                "url": url or "",
                "snippet": (text[:300] + "...") if text else snippet
            })
    return results
