# src/verifier_openai.py
import os, json
from typing import List, Dict
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = (
    "You are a meticulous fact-checking assistant. "
    "Extract discrete factual claims from the ARTICLE (1–6 claims). "
    "Check each claim ONLY against the provided SOURCES (URLs + snippets). "
    "Return a compact JSON with keys: overall_verdict (likely_true | likely_false | uncertain) "
    "and claims (array of {claim, status: supported|contradicted|insufficient, evidence: [urls], rationale}). "
    "If evidence is weak or missing, use 'insufficient'. "
    "Never invent sources; cite only given URLs. Respond with JSON only."
)

def verify_with_openai(article_text: str, sources: List[Dict]) -> Dict:
    """
    sources: [{"url": "...", "snippet": "..."}]
    Returns: {"overall_verdict": "...", "claims": [ {...}, ... ]}
    """
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "overall_verdict": "uncertain",
            "claims": [{
                "claim": "(OPENAI_API_KEY missing)",
                "status": "insufficient",
                "evidence": [],
                "rationale": "Set environment variable OPENAI_API_KEY."
            }]
        }

    # Gộp nguồn (giới hạn snippet để tiết kiệm token)
    context = "\n\n".join(
        f"- {s.get('url','')}\n{(s.get('snippet') or '')[:1200]}"
        for s in sources
    )
    user_prompt = f"ARTICLE:\n{article_text}\n\nSOURCES:\n{context}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",            # có thể đổi sang "gpt-4.1-mini" nếu tài khoản hỗ trợ
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    content = resp.choices[0].message.content or "{}"

    # Cố gắng parse JSON; nếu thất bại, bọc vào cấu trúc chuẩn
    try:
        data = json.loads(content)
        # đảm bảo các trường tối thiểu
        if "overall_verdict" not in data:
            data["overall_verdict"] = "uncertain"
        if "claims" not in data or not isinstance(data["claims"], list):
            data["claims"] = []
        return data
    except Exception:
        return {
            "overall_verdict": "uncertain",
            "claims": [{
                "claim": "(Could not parse JSON from model)",
                "status": "insufficient",
                "evidence": [],
                "rationale": content[:800]
            }]
        }
