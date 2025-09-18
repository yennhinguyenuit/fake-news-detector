import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.infer import analyze, fact_check
from src.verifier_gemini import verify_with_gemini

# ================== CONFIG ==================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
    }
    .stTextArea textarea {
        background-color: #f0faff;
        border: 2px solid #1a73e8;
        border-radius: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #444;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== HEADER ==================
banner_path = os.path.join("assets", "banner.png")

if os.path.exists(banner_path):
    st.image(banner_path, use_column_width=True)
else:
    st.markdown(
        """
        <div style="text-align:center; margin: 20px 0;">
          <h1>📰 Fake News Detector</h1>
          <p class="subtitle">Ứng dụng CNTT & AI trong đấu tranh chống tin giả</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================== APP ==================
st.title("🔎 Kiểm chứng tin tức trên không gian mạng")
st.write("Nhập một đoạn tin để phân tích độ tin cậy, đối chiếu nguồn và kiểm chứng bằng AI.")

text = st.text_area("✍️ Nhập tin cần kiểm chứng:", height=150)

if st.button("🚀 Phân tích & Kiểm chứng"):
    if text.strip():
        # 1️⃣ Model truyền thống
        label, confidence, probs = analyze(text)

        st.subheader("📊 Kết quả dự đoán")
        if label == "REAL":
            st.success(f"✅ **{label}** (Độ tin cậy: {confidence:.2f})")
        else:
            st.error(f"❌ **{label}** (Độ tin cậy: {confidence:.2f})")

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

        # 3️⃣ Phân tích bằng Gemini
        st.subheader("🤖 Báo cáo kiểm chứng AI")
        if sources:
            with st.spinner("Đang phân tích bằng Gemini..."):
                try:
                    report = verify_with_gemini(text, sources)

                    badge_map = {
                        "likely_true": "✅ Có vẻ đúng",
                        "likely_false": "❌ Có vẻ sai",
                        "uncertain": "⚠️ Chưa đủ bằng chứng"
                    }
                    st.markdown(
                        f"**Kết luận chung:** {badge_map.get(report.get('overall_verdict','uncertain'),'—')}"
                    )

                    for i, c in enumerate(report.get("claims", []), 1):
                        st.markdown(f"**{i}. Claim:** {c['claim']}")
                        st.write(f"• Trạng thái: `{c['status']}`")
                        if c.get("evidence"):
                            st.write("• Nguồn:")
                            for u in c["evidence"]:
                                st.write(f"  - {u}")
                        st.caption(f"• Lý do: {c['rationale']}")
                        st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Lỗi khi gọi Gemini: {e}")
        else:
            st.info("⚠️ Không có nguồn nên AI chưa thể kiểm chứng.")
    else:
        st.warning("⚠️ Hãy nhập nội dung trước khi kiểm chứng!")
