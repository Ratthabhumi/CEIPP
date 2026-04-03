import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ตั้งค่าฐานข้อมูล Vector
DB_DIR = "./chroma_db_v2"

def get_qa_chain(api_key, mode="chat"):
    """
    สร้าง RAG Chain สำหรับการตอบคำถาม
    mode: "chat" สำหรับคำถามทั่วไป, "audit" สำหรับการตรวจสอบใบเสร็จ
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # โหลดจากฐานข้อมูลเดิม
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # ระบบ Fallback สำหรับโมเดลฟรีใน OpenRouter
    models_to_try = [
        "google/gemma-3-27b-it:free",
        "google/gemma-3-4b-it:free",
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
                max_retries=1
            )
            break
        except:
            continue

    if not llm:
        llm = ChatOpenAI(model=models_to_try[0], base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # --- เลือก Prompt ตามโหมดการใช้งาน (Brain Refresh) ---
    if mode == "audit":
        system_message = (
            "### [ROLE: STRICT AUDITOR]\n"
            "คุณคือ AI ผู้เชี่ยวชาญด้านการตรวจสอบหลักฐานการจ่ายเงิน ของ สจล. หน้าที่เดียวของคุณคือตัดสิน 'ผ่าน' หรือ 'ไม่ผ่าน' เท่านั้น\n"
            "โดยยึดตาม PRIORITY RULES เป็นอันดับ 1 (สำคัญที่สุด):\n\n"
            "### 🚨 [MANDATORY PRIORITY RULES]\n"
            "1. บิลจากบริษัทใหญ่ (Makro, HomePro, 7-11, CP Axtra, Central, BigC, etc.) ที่ออกโดยระบบคอมพิวเตอร์ = [STATUS: PASS] เสมอ\n"
            "2. **ห้าม** แจ้งว่าไม่ผ่านเพียงเพราะไม่มีตราประทับ 'จ่ายเงินแล้ว' สีแดงสำหรับบิลตามข้อ 1\n"
            "3. ชื่อผู้ซื้อต้องเป็น 'สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง'\n\n"
            "### [INSTRUCTIONS]\n"
            "- วิเคราะห์ข้อมูลที่ได้รับ (Input) เทียบกับระเบียบ (Context)\n"
            "- ให้เหตุผลประกอบสั้นๆ และชัดเจน\n"
            "- ปิดท้ายคำตอบด้วย [STATUS: PASS] หรือ [STATUS: FAIL] เท่านั้น\n"
            "- **ห้าม** ตอบคำถามทั่วไปในโหมดนี้"
        )
        human_message = "Context: {context}\n\nInput (OCR Data): {input}"
    else:
        system_message = (
            "### [ROLE: HELPFUL CONSULTANT]\n"
            "คุณคือ AI ที่ปรึกษาด้านระเบียบการพัสดุและการเงิน ของ สจล. คุณจะตอบคำถามทั่วไปอย่างสุภาพและให้ข้อมูลที่ถูกต้อง\n\n"
            "### [RULES]\n"
            "1. ตอบคำถามตามระเบียบจาก <context> อ้างอิงเลขหน้าหรือข้อให้ชัดเจน\n"
            "2. **ห้าม** พูดเรื่อง [STATUS: PASS/FAIL] หรือกฎลำดับความสำคัญ (Priority Rules) ของการตรวจบิลในโหมดนี้เด็ดขาด\n"
            "3. หากไม่พบข้อมูล ให้บอกว่าไม่พบในระเบียบ สจล. และแนะนำให้ติดต่อกองคลัง\n"
            "4. รักษามารยาทและภาพลักษณ์ที่ดีของสถาบัน\n\n"
            "--- ลืมกฎการตรวจบิล (Audit Rules) ทั้งหมด และเน้นให้คำแนะนำที่ครอบคลุม ---"
        )
        human_message = "Context: {context}\n\nคำถามจากผู้ใช้: {input}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
