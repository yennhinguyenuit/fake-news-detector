import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import matplotlib.pyplot as plt
from src.infer import analyze, fact_check
from src.verifier_gemini import verify_with_gemini  # ✅ dùng Gemini

# ================== CONFIG ==================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #f5f7fa; }
    .stApp { background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%); }
    h1 { color: #1a73e8; text-align: center; }
    .stTextArea textarea {
        background-color: #fff8f0;
        border: 2px solid #1a73e8;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== APP UI ==================
st.title("📰 Fake News Detector with Fact-Checking + AI (Gemini)")
st.write("Nhập một đoạn tin để kiểm tra độ tin cậy, đối chiếu nguồn và phân tích bằng AI (Gemini – free quota).")

text = st.text_area("✍️ Nhập tin cần kiểm chứng:", height=150)

if st.button("🔎 Phân tích & Kiểm chứng"):
    if text.strip():
        # 1️⃣ Phân tích model (LogReg)
        label, confidence, probs = analyze(text)

        st.subheader("📊 Kết quả dự đoán")
        if label == "REAL":
            st.success(f"✅ **{label}** (Độ tin cậy: {confidence:.2f})")
        else:
            st.error(f"❌ **{label}** (Độ tin cậy: {confidence:.2f})")

        # --- Biểu đồ tròn Fake vs Real ---
        fig, ax = plt.subplots()
        ax.pie(
            [probs['real'], probs['fake']],
            labels=["REAL", "FAKE"],
            autopct='%1.1f%%',
            colors=["#4CAF50", "#F44336"],
            startangle=90
        )
        ax.axis('equal')
        st.pyplot(fig)

        # 2️⃣ Crawl nguồn Internet
        st.subheader("🌍 Kiểm chứng thêm từ Internet")
        sources = fact_check(text, max_results=3)
        if not sources:
            st.warning("Không tìm thấy nguồn tham khảo nào.")
        else:
            for s in sources:
                with st.expander(f"🔗 {s['title']}"):
                    if s["url"]:
                        st.markdown(f"[Mở nguồn]({s['url']})")
                    st.caption(s["snippet"])

        # 3️⃣ Phân tích bằng Gemini (free)
        st.subheader("🤖 Báo cáo kiểm chứng AI (Gemini)")
        if sources:
            try:
                with st.spinner("Đang phân tích bằng Gemini..."):
                    report_text = verify_with_gemini(text, sources)  # ✅ gọi Gemini
                st.write(report_text)  # ✅ Gemini trả về text
            except Exception as e:
                st.error(f"❌ Lỗi khi gọi Gemini: {e}")
        else:
            st.info("⚠️ Không có nguồn nên AI chưa thể kiểm chứng.")
    else:
        st.warning("⚠️ Hãy nhập nội dung trước khi kiểm chứng!")
