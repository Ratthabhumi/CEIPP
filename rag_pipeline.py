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
        "หน้าที่ของคุณคือตัดสิน 'ผ่าน' หรือ 'ไม่ผ่าน' โดยยึดตามกฎลำดับความสำคัญ (PRIORITY RULES) ด้านล่างนี้เป็นอันดับ 1 (สำคัญกว่าระเบียบกระทรวงการคลังทั่วไป)\n\n"
        
        "### 🚨 [กฎลำดับความสำคัญสูงสุด - MANDATORY PRIORITY RULES] 🚨\n"
        "ข้อ 1: หากเอกสารเป็น 'ใบกำกับภาษีเต็มรูป/ใบเสร็จรับเงิน' ที่ออกโดยระบบคอมพิวเตอร์ของบริษัทจดทะเบียน (เช่น Makro, CP Axtra, BigC, HomePro, 7-Eleven, นครชัยแอร์) ให้ถือว่า 'สมบูรณ์และจ่ายเงินแล้ว 100%' ทันที\n"
        "ข้อ 2: **[ห้ามละเมิดเด็ดขาด]** สำหรับบริษัทตามข้อ 1 'ไม่จำเป็น' ต้องมีตราประทับสีแดง 'จ่ายเงินแล้ว' และ 'ห้าม' นำเรื่องการขาดตราประทับนี้มาเป็นเหตุผลในการให้ FAIL หรือแจ้งแก้ไขเด็ดขาด เพราะหัวเอกสารกำกับภาษียืนยันสถานะการรับเงินโดยสมบูรณ์ตามกฎหมายแล้ว\n"
        "ข้อ 3: ต้องตรวจสอบ 'ชื่อผู้ซื้อ' ต้องระบุเป็น 'สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง' (หรือชื่อย่อ สจล.) หากผิดจุดนี้ถึงจะให้ FAIL ได้\n\n"
        
        "--- ข้อมูลพฤติการณ์เอกสาร (Input Details) ---\n"
        "{input}\n\n"
        
        "--- กฎระเบียบทางเทคนิค (Context References) ---\n"
        "{context}\n\n"
        
        "### 📝 คำสั่งสรุปผล (Final Decision Instructions)\n"
        "1. หากเข้าข่ายกฎข้อ 1 และ 2 ด้านบน ให้สรุปผลเป็น [STATUS: PASS] ทันที แม้จะไม่มีตราประทับสีแดงก็ตาม\n"
        "2. ห้ามเขียนวิเคราะห์ในเชิงลบเกี่ยวกับเรื่อง 'ขาดตราประทับจ่ายเงินแล้ว' หากเป็นบริษัทขนาดใหญ่ ให้ชูข้อดีว่าเป็นเอกสารที่น่าเชื่อถือแทน\n"
        "3. อ้างอิงระเบียบกระทรวงการคลังข้อ 46 ประกอบเฉพาะส่วนที่สมบูรณ์ (เช่น วันที่, ยอดเงิน) เพื่อยืนยันความถูกต้อง\n"
        "คำตอบของคุณ:"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", combined_prompt),
    ])
    
    # สร้าง Chain (ประกอบร่างโมเดลกับ Prompt เข้าด้วยกัน)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
