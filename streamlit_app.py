import streamlit as st
import streamlit.components.v1 as components
import os
import json
import time
from datetime import datetime
import pandas as pd
from rag_pipeline import initialize_vector_db, get_qa_chain, _get_vectorstore

from ocr_pipeline import extract_receipt_data, verify_receipt_rules

# --- Caching Data Loaders ---
@st.cache_data(show_spinner=False)
def get_all_reports():
    """Load all JSON evaluation reports into memory once."""
    feedbacks = load_json_safe("feedback_log.json")
    approvals = load_json_safe("approved_logs.json")
    eval_report = load_json_safe("evaluation_report.json")
    retrieval_report = load_json_safe("retrieval_eval_report.json")
    return feedbacks, approvals, eval_report, retrieval_report

@st.cache_data(show_spinner=False)
def get_html_report(path):
    """Memory-efficient loading of the HTML dashboard."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(openrouter_key, gemini_key, mode, provider):
    return get_qa_chain(openrouter_key, gemini_api_key=gemini_key, mode=mode, provider=provider)

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
        "feedback": "👍 Correct/Helpful" if is_upvote else "👎 Incorrect/Hallucination"
    })
    save_json_safe("feedback_log.json", data)

def display_assistant_message(content, sources=None, index=0, chat_history=None):
    """
    Unified function to display assistant messages with content, sources, and feedback buttons.
    """
    st.markdown(content)
    
    if sources:
        with st.expander("📄 Supporting Source Documents"):
            for d in sources:
                source_file = os.path.basename(d.metadata.get("source", "Unknown File"))
                page_num = d.metadata.get("page", None)
                page_str = f" (Page {page_num+1})" if page_num is not None else ""
                
                header = f"**📌 {source_file}{page_str}**"
                st.markdown(f'<div class="reference-card">{header}<br>{d.page_content}</div>', unsafe_allow_html=True)
    
    # --- ปุ่ม Feedback (👍/👎) ---
    c1, c2, c3 = st.columns([1, 1, 10])
    
    # ดึงค่าที่เคยกดไว้ (ถ้ามี)
    already_voted = st.session_state.get("feedback", {}).get(index)
    
    if c1.button("👍", key=f"up_{index}", disabled=(already_voted is not None)):
        if chat_history and index-1 < len(chat_history):
            save_feedback(chat_history[index-1]["content"], content, True)
            if "feedback" not in st.session_state: st.session_state.feedback = {}
            st.session_state.feedback[index] = "up"
            st.toast("Feedback recorded! Thank you. ✨")
            time.sleep(0.5)
            st.rerun()
    
    if c2.button("👎", key=f"down_{index}", disabled=(already_voted is not None)):
        if chat_history and index-1 < len(chat_history):
            save_feedback(chat_history[index-1]["content"], content, False)
            if "feedback" not in st.session_state: st.session_state.feedback = {}
            st.session_state.feedback[index] = "down"
            st.toast("Feedback recorded. We'll improve soon! 🛠️")
            time.sleep(0.5)
            st.rerun()

# ==========================================
# ⚙️ Sidebar Navigation & Settings
# ==========================================
with st.sidebar:
    st.title("🎓 Budget AI")
    
    st.markdown("### 📌 Main Menu")
    page = st.radio("Navigation", ["💬 Regulation Chat", "📸 Receipt Audit", "📊 Admin Dashboard"], label_visibility="collapsed")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    with st.expander("🔑 API Settings", expanded=True):
        api_key_input = st.text_input("OpenRouter Key:", type="password", help="Used for RAG (Gemma 3)")
        gemini_key_input = st.text_input("Google Gemini Key:", type="password", help="Used for OCR (Gemini 1.5 Flash)")
        st.caption("💡 Tip: Store keys in HF Settings > Secrets for auto-login")

    # --- API Key Retrieval ---
    def get_api_key(secret_name, user_input=""):
        val = ""
        try:
            if secret_name in st.secrets:
                val = st.secrets[secret_name]
        except Exception:
            pass

        if not val:
            val = os.environ.get(secret_name, "")

        return val.strip() if val else user_input.strip()

    openrouter_api_key = get_api_key("OPENROUTER_API_KEY", api_key_input)
    gemini_api_key = get_api_key("GEMINI_API_KEY", gemini_key_input)

    if not openrouter_api_key and not gemini_api_key:
        st.error("⚠️ Please provide at least one API Key to start. You can type it above or add it to HF 'Secrets'.")
        st.stop()

    # API Status
    st.markdown("**API Status:**")
    if openrouter_api_key:
        st.success("✅ OpenRouter Key: Ready")
    else:
        st.warning("⚠️ OpenRouter Key: Not configured")
    if gemini_api_key:
        st.success("✅ Google Gemini Key: Ready")
    else:
        st.warning("⚠️ Gemini Key: Not configured")

    # Auto-build ChromaDB
    if not os.path.exists("./chroma_db_v2"):
        st.info("🔄 Building Document Database (First time, may take 1-2 mins)...")
        initialize_vector_db(openrouter_api_key or gemini_api_key)
        st.cache_resource.clear()
        st.success("✅ Database Ready!")
        st.rerun()
    
    with st.expander("🛠️ System Management (Admin)"):
        if st.button("🔄 Rebuild Vector Database", use_container_width=True):
            with st.spinner("Processing..."):
                initialize_vector_db(openrouter_api_key)
                st.success("Rebuild Successful!")

# ==========================================
# 💬 หน้าแชทบอท (วิเคราะห์ระเบียบ)
# ==========================================
if "💬" in page:
    st.title("💬 Regulation Analyzer")
    st.markdown("_Ask about procurement steps, budget codes, or KMITL regulations._")

    if "messages" not in st.session_state: st.session_state.messages = []
    if "feedback" not in st.session_state: st.session_state.feedback = {}

    # Display chat messages
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                display_assistant_message(
                    m["content"], 
                    sources=m.get("sources"), 
                    index=i, 
                    chat_history=st.session_state.messages
                )

    # Chat Input
    if prompt := st.chat_input("Example: What is the hotel reimbursement limit?..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI is searching for relevant regulations..."):
                providers_to_try = ["gemini", "openrouter"]
                success = False

                for p_name in providers_to_try:
                    try:
                        qa = get_cached_qa_chain(openrouter_api_key, gemini_api_key, "chat", p_name)
                        if qa:
                            res = qa.invoke({"input": prompt})
                            ans, ctx = res["answer"], res.get("context", [])
                            
                            # Live Rendering
                            display_assistant_message(
                                ans, 
                                sources=ctx, 
                                index=len(st.session_state.messages), 
                                chat_history=st.session_state.messages
                            )
                            
                            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": ctx})
                            success = True
                            break
                    except Exception as e:
                        if p_name == providers_to_try[-1]: # If last provider also fails
                            if "429" in str(e) or "rate_limit" in str(e).lower():
                                st.error("⚠️ AI models are busy (Rate Limit). Please wait a moment and try again.")
                            else:
                                st.error(f"Critical Error: {e}")
                        else:
                            continue # Try next provider
                
                if not success:
                    st.error("❌ AI Connection Failed — Check API Keys, Quota, or Rebuild Vector DB in Admin.")

# ==========================================
# 📸 Receipt Auditor (OCR & Compliance)
# ==========================================
elif "📸" in page:
    st.title("📸 Receipt Auditor")
    
    # Responsive Columns
    col_up, col_res = st.columns([1, 1.2], gap="large")
    
    with col_up:
        st.markdown("### 📥 Document Upload")
        tab_f, tab_c = st.tabs(["📂 Image File", "📷 Camera"])
        up = tab_f.file_uploader("Upload Image", type=["png","jpg","jpeg"], label_visibility="collapsed")
        cam = tab_c.camera_input("Camera Input", label_visibility="collapsed")
        
        target = cam if cam else up
        if target:
            st.image(target, use_container_width=True, caption="Selected Image")
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("AI OCR is extracting data..."):
                    ocr = extract_receipt_data(target.getvalue(), gemini_api_key)
                if "error" in ocr: 
                    st.error(f"OCR Error: {ocr['error']}")
                else:
                    st.session_state.ocr = ocr
                    providers_to_try = ["gemini", "openrouter"]
                    success = False

                    for p_name in providers_to_try:
                        try:
                            qa = get_cached_qa_chain(openrouter_api_key, gemini_api_key, "audit", p_name)
                            if qa:
                                with st.spinner(f"Auditing against KMITL regulations ({p_name})..."):
                                    v_res = verify_receipt_rules(qa, ocr)
                                    if v_res.get("status") == "ERROR":
                                        continue  # Try next provider
                                    st.session_state.v_res = v_res
                                    success = True
                                    break
                        except Exception as e:
                            if p_name == providers_to_try[-1]:
                                st.error(f"AI Connection failed for all providers: {e}")
                            else:
                                continue
                    
                    if not success:
                        st.info("Please check API settings or wait before retrying.")

    with col_res:
        st.markdown("### 🚦 Audit Results")
        if "v_res" in st.session_state:
            v, o = st.session_state.v_res, st.session_state.ocr
            
            # Status Badge
            if v["status"] == "PASS": 
                st.success("✅ **APPROVED:** Document is compliant with regulations.")
            elif v["status"] == "FAIL": 
                st.error("❌ **REJECTED:** Found non-compliance issues.")
            else: 
                st.warning("⚠️ **MANUAL REVIEW:** Uncertainties found, needs manual check.")
            
            st.markdown(f"**💡 Regulation Analysis:**")
            st.write(v['analysis'])
            
            with st.expander("🛠️ View Extracted Raw Data"):
                st.json(o)
            
            st.divider()
            if st.button("📫 Submit to DMS Workflow (Simulation)", use_container_width=True):
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
                st.success("Successfully saved to Cloud system!")
        else:
            st.info("Please upload or take a photo of a receipt to start analysis.")

# ==========================================
# 📊 Admin Dashboard
# ==========================================
else:
    st.title("📊 Admin Dashboard")
    
    # Load all reports using memory-efficient caching
    feedbacks, approvals, eval_report, retrieval_report = get_all_reports()
    
    # --- Metrics Grid ---
    st.markdown("### 🎯 System Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chat Feedbacks Total", len(feedbacks))
    m2.metric("Processed Bills", len(approvals))
    
    good_rates = [x.get("score", 0) for x in feedbacks if "score" in x]
    avg_sat = sum(good_rates)/len(good_rates) if good_rates else 0
    m3.metric("AI Satisfaction Score", f"{avg_sat*100:.1f}%")
    
    # --- NEW: Retrieval Performance Section (Dynamic Calculation) ---
    st.divider()
    st.markdown("### 🔍 Retrieval Performance (Live Analysis)")
    if retrieval_report:
        per_q = retrieval_report.get("per_question", [])
        if per_q:
            # Recalculate averages from per-question data
            recalls = [q["metrics"].get("Recall@10", 0) for q in per_q]
            precs = [q["metrics"].get("Precision@10", 0) for q in per_q]
            mrrs = [q["metrics"].get("MRR", 0) for q in per_q]
            cosines = [q["metrics"].get("Avg_Cosine", 0) for q in per_q]
            
            avg_recall = sum(recalls) / len(recalls)
            avg_prec = sum(precs) / len(precs)
            avg_mrr = sum(mrrs) / len(mrrs)
            avg_cosine = sum(cosines) / len(cosines)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Recall@10", f"{avg_recall*100:.2f}%")
            c2.metric("Avg Precision@10", f"{avg_prec*100:.2f}%")
            c3.metric("Avg MRR", f"{avg_mrr:.4f}")
            c4.metric("Avg Cosine Score", f"{avg_cosine:.4f}")
        else:
            st.info("No per-question data found in Retrieval Report")
    else:
        st.info("No retrieval evaluation data available yet")
    
    # --- RAG Performance Section ---
    st.divider()
    st.markdown("### 🧬 RAG Quality Metrics (LLM-as-a-Judge)")
    
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
        # --- Trend charts ---
        st.markdown("#### 📉 Performance Trends")
        chart_data = pd.DataFrame({
            "Faithfulness": faith_scores,
            "Relevance": rel_scores,
            "Precision": prec_scores
        })
        st.area_chart(chart_data, height=300)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Detailed View (Per-Question Data)"):
            eval_rows = []
            for e in eval_report:
                sc = e.get("evaluation_scores", {})
                eval_rows.append({
                    "Question": e.get("question"),
                    "Faithful": sc.get("Faithfulness"),
                    "Relevance": sc.get("Answer_Relevance"),
                    "Precision": sc.get("Context_Precision"),
                    "Reasoning": sc.get("Reasoning")
                })
            st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)
    else:
        st.info("No RAG evaluation data available yet")

    # --- Logs Grid ---
    st.divider()
    st.markdown("### 📜 Activity Logs")
    l1, l2 = st.columns(2)
    with l1:
        st.markdown("#### 💌 Recent User Feedback")
        if feedbacks:
            st.dataframe(pd.DataFrame(feedbacks).tail(10), use_container_width=True)
        else: st.caption("No data available")
    
    with l2:
        st.markdown("#### 📦 Recent DMS Approvals")
        if approvals:
            st.dataframe(pd.DataFrame(approvals).tail(10), use_container_width=True)
        else: st.caption("No data available")

    # --- NEW: Full Detailed Evaluation Report (CEIPP) ---
    st.divider()
    st.markdown("### 🔍 Full Analytical Report (CEIPP)")
    
    with st.expander("🚀 Open Detailed Performance Dashboard (HTML Report)", expanded=False):
        try:
            report_path = "evaluation_dashboard.html"
            html_content = get_html_report(report_path) # Optimized cached load
            
            if html_content:
                # Action Buttons
                c1, c2 = st.columns([1, 4])
                c1.download_button(
                    label="💾 Download .html",
                    data=html_content,
                    file_name=f"CEIPP_Report_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html"
                )
                st.caption("Tip: Use Fullscreen or Download for high-resolution viewing")
                
                # Embed the Chart.js Dashboard
                components.html(html_content, height=1200, scrolling=True)
            else:
                st.error(f"Report file not found at {report_path}")
        except Exception as e:
            st.error(f"Error loading report: {e}")
