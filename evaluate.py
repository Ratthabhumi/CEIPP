import os
import json
import time
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# โหลด API Keys
try:
    import streamlit as st
    gemini_key = st.secrets.get("GEMINI_API_KEY", "").strip()
except Exception:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

if not gemini_key:
    print("No GEMINI_API_KEY found in .streamlit/secrets.toml or environment")
    exit(1)

# Build chain โดยตรง ไม่ผ่าน get_qa_chain (หลีกเลี่ยง @st.cache_resource นอก Streamlit)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

DB_DIR = "./chroma_db_v2"

def clean_thai_text(text):
    if not text:
        return ""
    text = re.sub(r'([ก-ฮ])\s+า', r'\1า', text)
    text = re.sub(r'([ก-ฮ])\s*ำ', r'\1ำ', text)
    text = re.sub(r'([ก-ฮ])\s+([ะ-ู])', r'\1\2', text)
    text = re.sub(r'([ก-ฮ])\s+([่-๋])', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_chain():
    print("กำลังโหลด Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    if not os.path.exists(DB_DIR):
        print(f"ไม่พบ ChromaDB ที่ {DB_DIR} กรุณา rebuild ก่อน")
        return None

    print("กำลังโหลด ChromaDB...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever_obj = vectorstore.as_retriever(search_kwargs={"k": 10})

    def wrap_clean(docs):
        for doc in docs:
            doc.page_content = clean_thai_text(doc.page_content)
        return docs

    retriever = retriever_obj | RunnableLambda(wrap_clean)

    print("กำลังโหลด Gemini LLM...")
    llm = None
    llm_is_gemma = False
    for model_name in ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemma-3-4b-it", "gemma-3-1b-it"]:
        is_gemma = model_name.startswith("gemma")
        try:
            candidate = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_key,
                temperature=0,
                max_retries=1,
                convert_system_message_to_human=is_gemma,
            )
            candidate.invoke("test")
            llm = candidate
            llm_is_gemma = is_gemma
            print(f"  ใช้ model: {model_name}")
            break
        except Exception as e:
            print(f"  {model_name}: {str(e)[:60]}")
            continue

    system_msg = (
        "### [ROLE: HELPFUL CONSULTANT]\n"
        "คุณคือ AI ที่ปรึกษาด้านระเบียบการพัสดุและการเงิน ของ สจล. คุณจะตอบคำถามอย่างสุภาพและแม่นยำ\n\n"
        "### [KMITL/Public Procurement Ground Truth]\n"
        "- หลักการจัดซื้อจัดจ้าง (พรบ. 2560 มาตรา 8): ความคุ้มค่า, ความโปร่งใส, ประสิทธิภาพและประสิทธิผล, การตรวจสอบได้\n"
        "- ราคากลาง: อ้างอิงตามลำดับ (1.คณะกรรมการราคากลาง, 2.กรมบัญชีกลาง, 3.สำนักงบประมาณ, 4.สืบราคาตลาด, 5.ราคาซื้อล่าสุดใน 2 ปี)\n"
        "- e-catalog: ระบบบัญชีรายชื่อสินค้าและบริการอิเล็กทรอนิกส์บนแพลตฟอร์ม e-GP ของกรมบัญชีกลาง\n"
        "- e-bidding: วงเงินเกิน 500,000 บาท\n"
        "- วิธีเฉพาะเจาะจง: วงเงินไม่เกิน 500,000 บาท; ไม่เกิน 100,000 บาท ไม่ต้องทำสัญญาเป็นหนังสือ\n"
        "- วิธีคัดเลือก: ผู้ขายคุณสมบัติครบน้อยกว่า 3 ราย / เร่งด่วน / ความเชี่ยวชาญเฉพาะ / ความลับ\n"
        "- ผู้ถือหุ้นรายใหญ่: ถือหุ้นเกินร้อยละ 25\n"
        "- การอุทธรณ์: ยื่นภายใน 7 วันทำการ; คณะกรรมการพิจารณาภายใน 30 วัน\n"
        "- อำนาจหัวหน้าหน่วยงาน: คัดเลือก ≤ 100 ล้าน, e-bidding ≤ 200 ล้าน, เฉพาะเจาะจง ≤ 50 ล้านบาท\n"
        "- e-Market ค่าเริ่มต้น: E6=3 วันทำการ, E8=15 วัน, E9=7 วัน; R1=สัญญาซื้อขายทั่วไป, R2=7 วัน, R3=5%, R4=ปรับเฉพาะที่ยังไม่ส่ง, R5=0.20%, R6-8=1 เดือน, R9=15 วัน\n\n"
        "### [RULES]\n"
        "1. ตอบจาก Context ที่ได้รับเป็นลำดับแรก\n"
        "2. หาก Context ไม่มีข้อมูล ให้ใช้ Ground Truth ด้านบนหรือความรู้ระเบียบจัดซื้อจัดจ้าง พ.ศ. 2560 ตอบแทนได้\n"
        "3. ปิดท้ายด้วยคำแนะนำให้ติดต่อ 'กองคลัง สจล.'"
    )

    if llm_is_gemma:
        # gemma ไม่รองรับ system message — รวมเป็น human message เดียว
        prompt = ChatPromptTemplate.from_messages([
            ("human", system_msg + "\n\nContext: {context}\n\nคำถาม: {input}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "Context: {context}\n\nคำถาม: {input}")
        ])

    if not llm:
        print("ไม่สามารถโหลด Gemini ได้ (quota หมดหรือ API key ผิด) รอพรุ่งนี้")
        return None

    def run_rag(inputs):
        question = inputs["input"]
        docs = retriever.invoke(question)
        context_str = "\n\n".join(doc.page_content for doc in docs)
        messages = prompt.format_messages(context=context_str, input=question)
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer, "context": docs}

    return RunnableLambda(run_rag)


def invoke_with_retry(chain, question, max_retries=4):
    """เรียก chain พร้อม retry + exponential backoff เมื่อ rate limit"""
    for attempt in range(max_retries):
        try:
            return chain.invoke({"input": question})
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
                print(f"  ⏳ Rate limit (attempt {attempt+1}/{max_retries}) รอ {wait} วินาที...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"ยังติด rate limit หลังจาก {max_retries} ครั้ง")


def run_evaluation(dataset_file="golden_dataset.json", report_file="evaluation_report.json"):
    print("=" * 50)
    print("เริ่มต้นการทดสอบ RAG Evaluation (Golden Dataset)")
    print("=" * 50)

    if not os.path.exists(dataset_file):
        print(f"ไม่พบไฟล์ชุดทดสอบ {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    qa_chain = build_chain()
    if not qa_chain:
        return

    results = []
    completed_ids = set()
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                results = json.load(f)
                completed_ids = {r.get("question_id") for r in results if "question_id" in r}
                print(f"Resume: ข้ามไป {len(completed_ids)} ข้อที่เสร็จแล้ว")
        except Exception:
            pass

    total = len(dataset)

    for i, item in enumerate(dataset, 1):
        if i in completed_ids:
            continue

        question = item.get("question", "")
        expected = item.get("expected_answer", "")

        print(f"\n[{i}/{total}] Q: {question[:80]}")

        context_texts = []
        try:
            t0 = time.time()
            response = invoke_with_retry(qa_chain, question)
            ai_answer = response["answer"]
            for doc in response.get("context", []):
                context_texts.append(doc.page_content)
            print(f"  OK ({time.time()-t0:.1f}s): {ai_answer[:100]}")
        except Exception as e:
            ai_answer = f"Error: {str(e)}"
            print(f"  FAIL: {e}")

        results.append({
            "question_id": i,
            "question": question,
            "expected_answer": expected,
            "ai_answer": ai_answer,
            "context_texts": context_texts,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        time.sleep(3)

    print("\n" + "=" * 50)
    print(f"เสร็จสิ้น! บันทึกรายงานที่ {report_file}")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
    from ai_evaluator import llm_as_a_judge
    llm_as_a_judge()
