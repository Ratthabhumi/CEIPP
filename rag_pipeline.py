import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db_v2"
DOCS_DIR = "./Docs"

def initialize_vector_db(api_key):
    """อ่านไฟล์ PDF ทั้งหมดในโฟลเดอร์ Docs และสร้างฐานข้อมูลเวกเตอร์ ChromaDB"""
    
    # ตั้งค่า API Key ให้กับ environment (Langchain จะนำไปใช้โดยอัตโนมัติ)
    # ค้นหาไฟล์ PDF ทั้งหมด (ใช้ set เพื่อป้องกันไฟล์ซ้ำในระบบที่ Case-insensitive เช่น Windows)
    all_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf")) + glob.glob(os.path.join(DOCS_DIR, "*.PDF"))
    pdf_files = list(set(all_files))
    
    if not pdf_files:
        st.error(f"ไม่พบไฟล์ PDF ในโฟลเดอร์ {DOCS_DIR}")
        return False
        
    documents = []
    
    # สร้างตัวแสดงสถานะ
    progress_bar = st.progress(0, text="กำลังเตรียมไฟล์...")
    
    # 1. โหลดเอกสาร PDF
    for i, pdf_file in enumerate(pdf_files):
        try:
            loader = PyMuPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"ไม่สามารถโหลดไฟล์ {pdf_file} ได้: {e}")
            
        progress_bar.progress((i + 1) / len(pdf_files), text=f"กำลังโหลด... {os.path.basename(pdf_file)}")
            
    # 2. ตัดแบ่งข้อความ (Chunking) เพื่อให้โมเดลประมวลผลได้ดีขึ้น
    progress_bar.progress(0.9, text="กำลังตัดแบ่งข้อความ (Splitting)... อาจใช้เวลาสักครู่")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # 3. สร้าง Embeddings ด้วย Local Model (ไม่ต้องใช้ API ของ Google อีกต่อไป)
    progress_bar.progress(0.95, text="กำลังโหลดโมเดลภาษาไทย (Local Embedding) และบันทึกลง ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # วิธีนี้จะสร้าง folder ชื่อ chroma_db เพื่อเก็บข้อมูลไว้เปิดครั้งต่อไปได้เลย
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    batch_size = 20
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        vectorstore.add_documents(documents=batch)
    
    progress_bar.empty()
    st.success(f"ดำเนินการเสร็จสิ้น! โหลดเอกสารทั้งหมด {len(pdf_files)} ไฟล์ เรียบร้อยแล้ว")
    return True

def get_qa_chain(api_key):
    """สร้าง Chain สำหรับตอบคำถามอ้างอิงจาก Vector DB ที่มีอยู่"""
    
    if not os.path.exists(DB_DIR):
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # โหลดจากฐานข้อมูลเดิมที่เคยบันทึกไว้
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # รายชื่อโมเดลฟรีที่เสถียรที่สุดในกรณีที่ตัวใดตัวหนึ่งคิวเต็ม (429 Rate Limit)
    models_to_try = [
        "google/gemma-3-4b-it:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free"
    ]
    
    llm = None
    for model_name in models_to_try:
        try:
            llm = ChatOpenAI(
                model=model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                temperature=0,
                max_retries=1 # ลองแค่ครั้งเดียวแล้วข้ามถ้าไม่ได้
            )
            # ทดสอบเบื้องต้น (ทำ Dummy call หรือปล่อยผ่านไปก่อน)
            break
        except:
            continue

    if not llm:
        # ถ้าพังหมดจริงๆ ให้กลับไปที่ตัวแรกสุด
        llm = ChatOpenAI(
            model=models_to_try[0],
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0,
        )
    
    # ปรับจูน Prompt ให้มีความเข้าใจโลกความจริงและไม่เถรตรงจนเกินไป
    combined_prompt = (
        "คุณคือ AI ผู้เชี่ยวชาญด้านการตรวจสอบหลักฐานการจ่ายเงิน ของสถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง (สจล.)\n"
        "หน้าที่ของคุณคือประเมินเอกสารตาม 'ความเป็นจริง' และ 'ระเบียบการคลัง' อย่างสมบูรณ์\n\n"
        "--- กฎการความสำคัญสูงสุด (PRIORITY RULES) ---\n"
        "1. หากเอกสารเป็น 'ใบกำกับภาษีเต็มรูป' (Full Tax Invoice) หรือ 'ใบเสร็จรับเงิน/ใบกำกับภาษี' ที่ออกโดยระบบคอมพิวเตอร์ของบริษัทจดทะเบียน (เช่น Makro, BigC, 7-Eleven, HomePro, นครชัยแอร์) ให้ถือว่า 'สมบูรณ์' และ 'จ่ายเงินแล้ว' โดยสภาพ\n"
        "2. **สำคัญมาก**: สำหรับบริษัทใหญ่เหล่านี้ 'ไม่จำเป็น' ต้องมีตราประทับ 'จ่ายเงินแล้ว' สีแดงประทับทับอีกครั้ง เพราะระบบคอมพิวเตอร์ยืนยันการรับเงินในหัวกระดาษอยู่แล้ว ห้ามนำเรื่องขาดตราประทับมาเป็นเหตุผลในการไม่ให้ผ่าน (Reject/FAIL) ในกรณีนี้เด็ดขาด!\n"
        "3. ตรวจสอบ 'ชื่อผู้ซื้อ': ต้องระบุเป็น 'สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง' หรือหน่วยงานในสังกัดให้ถูกต้อง หากผิดจุดนี้ให้ FAIL ทันที\n\n"
        "--- ข้อมูลระเบียบอ้างอิง (Context) ---\n"
        "{context}\n\n"
        "--- พฤติการณ์เอกสารที่รับมา ---\n"
        "{input}\n\n"
        "คำสั่งการสรุปผล:\n"
        "- วิเคราะห์ตามกฎ PRIORITY RULES ด้านบนเป็นหลัก\n"
        "- หากเป็นบิลเงินสดทั่วไป (เขียนมือ) ยังต้องมีตราประทับและชื่อผู้รับเงินตามระเบียบข้อ 46\n"
        "- ให้เหตุผลที่ชัดเจนและอ้างอิงเลขหน้า/ข้อของระเบียบประกอบ\n"
        "คำตอบของคุณ:"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", combined_prompt),
    ])
    
    # สร้าง Chain (ประกอบร่างโมเดลกับ Prompt เข้าด้วยกัน)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
