import streamlit as st
import os
import json
import time
from datetime import datetime
import pandas as pd
from rag_pipeline import initialize_vector_db, get_qa_chain
from ocr_pipeline import extract_receipt_data, verify_receipt_rules

# --- Page Configuration ---
st.set_page_config(
    page_title="KMITL Budget AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Custom Styling (Premium & Mobile-First) ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Sans+Thai:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'IBM+Plex+Sans+Thai', sans-serif;
    }

    /* Theme-Aware Backgrounds & Containers */
    .stApp {
        background-color: transparent !important;
    }
    
    /* Mobile-Specific Tweaks */
    @media (max-width: 640px) {
        .main .block-container { padding: 1rem !important; }
        h1 { font-size: 1.8rem !important; }
        .stButton>button { height: 3.5rem !important; font-size: 1.1rem !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px !important; }
    }

    /* Professional Elements & Cards */
    .stButton>button {
        border-radius: 12px;
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(128, 128, 128, 0.2);
        font-weight: 600;
    }
    .stButton>button:active { transform: scale(0.98); }
    
    /* Input Boxes Visibility for API Keys */
    .stTextInput input {
        border: 1px solid rgba(128, 128, 128, 0.5) !important;
        background-color: rgba(128, 128, 128, 0.1) !important;
    }

    /* Custom Reference Card (Glassmorphism inspired) */
    .reference-card {
        padding: 12px;
        background: rgba(128, 128, 128, 0.05);
        border-left: 4px solid #ff4b4b;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
def load_json_safe(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content: return []
                return json.loads(content)
        except: return []
    return []

def save_json_safe(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_feedback(question, answer, is_upvote):
    data = load_json_safe("feedback_log.json")
    data.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question, "answer": answer,
        "score": 1 if is_upvote else 0,
        "feedback": "👍 ดี/ถูกต้อง" if is_upvote else "👎 ไม่ถูกต้อง/มั่ว"
    })
    save_json_safe("feedback_log.json", data)

# ==========================================
# ⚙️ Sidebar Navigation & Settings
# ==========================================
with st.sidebar:
    st.title("🎓 Budget AI")
    
    st.markdown("### 📌 เมนูหลัก")
    page = st.radio("Navigation", ["💬 วิเคราะห์ระเบียบ", "📸 ตรวจสอบใบเสร็จ", "📊 Admin Dashboard"], label_visibility="collapsed")
    
    if st.button("🗑️ ล้างประวัติแชท (Clear Chat)", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    with st.expander("🔑 API Settings", expanded=True):
        api_key_input = st.text_input("OpenRouter Key:", type="password", help="ใช้สำหรับ RAG (Gemma 3)")
        gemini_key_input = st.text_input("Google Gemini Key:", type="password", help="ใช้สำหรับ OCR (Gemini 1.5 Flash)")
        st.caption("💡 แนะนำให้บันทึกใน secrets.toml เพื่อความสะดวก")

    openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", api_key_input).strip()
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", gemini_key_input).strip()

    if not openrouter_api_key or not gemini_api_key:
        st.error("⚠️ กรุณระบุ API Keys เพื่อเริ่มใช้งาน")
        st.stop()
    
    with st.expander("🛠️ จัดการระบบ (Admin)"):
        if st.button("🔄 รีเซ็ตฐานข้อมูล Vector", use_container_width=True):
            with st.spinner("กำลังดำเนินการ..."):
                initialize_vector_db(openrouter_api_key)
                st.success("Rebuild สำเร็จ!")

# ==========================================
# 💬 หน้าแชทบอท (วิเคราะห์ระเบียบ)
# ==========================================
if "💬" in page:
    st.title("💬 วิเคราะห์ระเบียบการ")
    st.markdown("_ถามเกี่ยวกับขั้นตอนการเบิกจ่าย พัสดุ หรือกฎเกณฑ์ สจล._")

    if "messages" not in st.session_state: st.session_state.messages = []

    # Display chat messages
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if "sources" in m and m["sources"]:
                    with st.expander("📄 แหล่งอ้างอิงเอกสารที่ใช้"):
                        for d in m["sources"]:
                            st.markdown(f'<div class="reference-card">{d.page_content}</div>', unsafe_allow_html=True)
                
                # --- ปุ่ม Feedback (👍/👎) ---
                c1, c2, c3 = st.columns([1, 1, 10])
                if c1.button("👍", key=f"up_{i}"):
                    save_feedback(st.session_state.messages[i-1]["content"], m["content"], True)
                    st.toast("บันทึกการตอบรับเชิงบวกเรียบร้อย! ✨")
                if c2.button("👎", key=f"down_{i}"):
                    save_feedback(st.session_state.messages[i-1]["content"], m["content"], False)
                    st.toast("บันทึกคำติชมเชิงลบแล้ว เราจะปรับปรุงให้เก่งขึ้นครับ! 🛠️")

    # Chat Input
    if prompt := st.chat_input("ตัวอย่าง: เบิกค่าที่พักได้คืนละเท่าไหร่..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI กำลังค้นหาระเบียบที่เกี่ยวข้อง..."):
                qa = get_qa_chain(openrouter_api_key, mode="chat")
                res = qa.invoke({"input": prompt})
                ans, ctx = res["answer"], res.get("context", [])
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans, "sources": ctx})

# ==========================================
# 📸 หน้าตรวจสอบใบเสร็จ (Receipt Verification)
# ==========================================
elif "📸" in page:
    st.title("📸 ตรวจสอบใบเสร็จ")
    
    # Responsive Columns
    col_up, col_res = st.columns([1, 1.2], gap="large")
    
    with col_up:
        st.markdown("### 📥 อัปโหลดเอกสาร")
        tab_f, tab_c = st.tabs(["📂 ไฟล์ภาพ", "📷 ถ่ายรูป"])
        up = tab_f.file_uploader("Upload Image", type=["png","jpg","jpeg"], label_visibility="collapsed")
        cam = tab_c.camera_input("Camera Input", label_visibility="collapsed")
        
        target = cam if cam else up
        if target:
            st.image(target, use_container_width=True, caption="รูปภาพที่เลือก")
            if st.button("🚀 เริ่มการวิเคราะห์", type="primary", use_container_width=True):
                with st.spinner("AI OCR กำลังสกัดข้อมูล..."):
                    ocr = extract_receipt_data(target.getvalue(), gemini_api_key)
                if "error" in ocr: 
                    st.error(f"ระบบ OCR ผิดพลาด: {ocr['error']}")
                else:
                    st.session_state.ocr = ocr
                    qa = get_qa_chain(openrouter_api_key, mode="audit")
                    with st.spinner("กำลังตรวจสอบกับระเบียบสถาบัน..."):
                        st.session_state.v_res = verify_receipt_rules(qa, ocr)

    with col_res:
        st.markdown("### 🚦 ผลการตรวจสอบ")
        if "v_res" in st.session_state:
            v, o = st.session_state.v_res, st.session_state.ocr
            
            # Status Badge
            if v["status"] == "PASS": 
                st.success("✅ **อนุมัติ:** เอกสารมีความสมบูรณ์ตามระเบียบ")
            elif v["status"] == "FAIL": 
                st.error("❌ **แจ้งแก้ไข:** พบข้อบกพร่องตามระเบียบ")
            else: 
                st.warning("⚠️ **ตรวจสอบเพิ่ม:** พบความไม่แน่นอนบางประการ")
            
            st.markdown(f"**💡 บทวิเคราะห์ระเบียบ:**")
            st.write(v['analysis'])
            
            with st.expander("🛠️ ตูข้อมูลดิบที่สกัดได้ (Raw Data)"):
                st.json(o)
            
            st.divider()
            if st.button("📫 ส่งข้อมูลเข้าเวิร์กโฟลว์ DMS (Simulation)", use_container_width=True):
                bar = st.progress(0, "Connecting to Cloud DMS...")
                for p in range(100):
                    time.sleep(0.01)
                    bar.progress(p+1)
                
                # Persist to local log
                logs = load_json_safe("approved_logs.json")
                logs.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "vendor": o.get("vendor_name"),
                    "total": o.get("total_amount"),
                    "status": v["status"]
                })
                save_json_safe("approved_logs.json", logs)
                st.balloons()
                st.success("บันทึกเข้าระบบ Cloud สำเร็จ!")
        else:
            st.info("กรุณาอัปโหลดหรือถ่ายภาพใบเสร็จเพื่อเริ่มการวิเคราะห์")

# ==========================================
# 📊 หน้าแดชบอร์ดผู้ดูแล (Admin Dashboard)
# ==========================================
else:
    st.title("📊 Admin Dashboard")
    
    feedbacks = load_json_safe("feedback_log.json")
    approvals = load_json_safe("approved_logs.json")
    eval_report = load_json_safe("evaluation_report.json")
    
    # --- Metrics Grid ---
    st.markdown("### 🎯 ภาพรวมระบบ")
    m1, m2, m3 = st.columns(3)
    m1.metric("จำนวนแชท/ฟีดแบ็ค", len(feedbacks))
    m2.metric("บิลที่ส่งตรวจสอบสำเร็จ", len(approvals))
    
    good_rates = [f["score"] for f in feedbacks if "score" in f]
    avg_sat = sum(good_rates)/len(good_rates) if good_rates else 0
    m3.metric("คะแนนความพึงพอใจ AI", f"{avg_sat*100:.1f}%")
    
    # --- RAG Performance Section (The "Old Way" Restored) ---
    st.markdown("---")
    st.markdown("### 🧬 ผลการตอบโจทย์ (Evaluation Metrics)")
    
    if eval_report:
        # Extract RAGAS-like scores (Safely with .get())
        faith_scores = [e["evaluation_scores"].get("Faithfulness") for e in eval_report if "evaluation_scores" in e and e["evaluation_scores"].get("Faithfulness") is not None]
        rel_scores = [e["evaluation_scores"].get("Answer_Relevance") for e in eval_report if "evaluation_scores" in e and e["evaluation_scores"].get("Answer_Relevance") is not None]
        prec_scores = [e["evaluation_scores"].get("Context_Precision") for e in eval_report if "evaluation_scores" in e and e["evaluation_scores"].get("Context_Precision") is not None]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness", f"{sum(faith_scores)/len(faith_scores):.2f}/5" if faith_scores else "N/A")
        c2.metric("Answer Relevance", f"{sum(rel_scores)/len(rel_scores):.2f}/5" if rel_scores else "N/A")
        c3.metric("Context Precision", f"{sum(prec_scores)/len(prec_scores):.2f}/5" if prec_scores else "N/A")
        
        st.divider()
        # --- กราฟแสดงแนวโน้ม (The Missing Graphs Restored) ---
        st.markdown("#### 📉 แนวโน้มประสิทธิภาพ AI (Evaluation Trend)")
        chart_data = pd.DataFrame({
            "Faithfulness": faith_scores,
            "Relevance": rel_scores,
            "Precision": prec_scores
        })
        st.area_chart(chart_data, height=300)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 ดูข้อมูลประเมินรายข้อ (Show Data Mode)"):
            eval_rows = []
            for e in eval_report:
                sc = e.get("evaluation_scores", {})
                eval_rows.append({
                    "คำถาม": e.get("question"),
                    "Faithful": sc.get("Faithfulness"),
                    "Relevance": sc.get("Answer_Relevance"),
                    "Precision": sc.get("Context_Precision"),
                    "Reasoning": sc.get("Reasoning")
                })
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)
    else:
        st.info("ยังไม่มีข้อมูลการประเมิน RAG ในขณะนี้")

    # --- Logs Grid ---
    st.markdown("---")
    st.markdown("### 📜 ประวัติการบันทึก (Logs)")
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("#### 💌 ล่าสุดจากผู้ใช้ (Feedback)")
        if feedbacks:
            st.dataframe(pd.DataFrame(feedbacks).tail(10), use_container_width=True)
        else: st.caption("ยังไม่มีข้อมูล")
    
    with l2:
        st.markdown("#### 📦 ล่าสุดที่เข้า DMS (Approvals)")
        if approvals:
            st.dataframe(pd.DataFrame(approvals).tail(10), use_container_width=True)
        else: st.caption("ยังไม่มีข้อมูล")