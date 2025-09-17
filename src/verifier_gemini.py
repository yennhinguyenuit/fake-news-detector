import os
import google.generativeai as genai

# Lấy key từ biến môi trường
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ Bạn cần đặt GEMINI_API_KEY trong môi trường.")

genai.configure(api_key=API_KEY)

# Chọn model free quota (Flash nhanh, Pro thông minh hơn)
MODEL_NAME = "gemini-1.5-flash"

def verify_with_gemini(claim: str, sources: list[dict]) -> str:
    """
    Dùng Gemini kiểm chứng thông tin claim dựa trên các nguồn web đã crawl.
    sources: list gồm {title, url, snippet}
    Trả về: Chuỗi text đánh giá
    """
    context_texts = []
    for s in sources:
        context_texts.append(f"- {s['title']} ({s['url']}): {s['snippet']}")
    context = "\n".join(context_texts)

    prompt = f"""
    Bạn là một hệ thống kiểm chứng tin tức.
    Nhiệm vụ: Phân tích đoạn tin sau và so sánh với nguồn tham khảo.

    🔎 Tin cần kiểm chứng:
    {claim}

    📚 Nguồn tham khảo:
    {context}

    Hãy trả lời:
    1. Tin này có vẻ thật hay giả?
    2. Lý do?
    3. Nếu có nguồn đáng tin, hãy trích dẫn.
    """

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)

    return resp.text.strip() if resp and resp.text else "❌ Không tạo được phản hồi từ Gemini."
