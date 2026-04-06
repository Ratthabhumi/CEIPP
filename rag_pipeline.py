import re
import os
import glob
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ตั้งค่าฐานข้อมูล Vector
DB_DIR = "./chroma_db_v2"
DOCS_DIR = "./Docs"

def clean_thai_text(text: str) -> str:
    """ทำความสะอาดข้อความภาษาไทยที่มีการเว้นวรรคผิดเพี้ยนจากการทำ OCR"""
    if not text: return ""
    # แก้ไข 'ก า' -> 'กำ' และกรณีสระอำอื่นๆ
    text = re.sub(r'([ก-ฮ])\s+า', r'\1า', text)
    # กรณีสระอำที่แยกออกจากกันชัดเจน เช่น 'ก ำ'
    text = re.sub(r'([ก-ฮ])\s*ำ', r'\1ำ', text)
    # ลบการเว้นวรรคที่เกิดจากการขึ้นบรรทัดใหม่กลางคำ (OCR Artifacts)
    text = re.sub(r'([ก-ฮ])\s+([ะ-ู])', r'\1\2', text)
    text = re.sub(r'([ก-ฮ])\s+([่-๋])', r'\1\2', text)
    # ลบช่องว่างที่มากเกินไป
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def initialize_vector_db(api_key):
    """อ่านไฟล์ PDF ทั้งหมดในโฟลเดอร์ Docs และสร้างฐานข้อมูลเวกเตอร์ ChromaDB"""
    
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
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
            loaded_docs = loader.load()
            # ทำความสะอาดข้อความทันทีหลังจากโหลด
            for doc in loaded_docs:
                doc.page_content = clean_thai_text(doc.page_content)
            documents.extend(loaded_docs)
        except Exception as e:
            st.warning(f"ไม่สามารถโหลดไฟล์ {pdf_file} ได้: {e}")
            
        progress_bar.progress((i + 1) / len(pdf_files), text=f"กำลังโหลด... {os.path.basename(pdf_file)}")
            
    # 2. ตัดแบ่งข้อความ (Chunking) เพื่อให้โมเดลประมวลผลได้ดีขึ้น
    progress_bar.progress(0.9, text="กำลังตัดแบ่งข้อความ (Splitting)... อาจใช้เวลาสักครู่")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    
    # 3. สร้าง Embeddings ด้วย Local Model
    progress_bar.progress(0.95, text="กำลังโหลดโมเดลภาษาไทย (Local Embedding) และบันทึกลง ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # สร้าง/โหลด Vector Store
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR) # Clean start for rebuild
        
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    
    progress_bar.empty()
    st.success(f"ดำเนินการเสร็จสิ้น! โหลดเอกสารทั้งหมด {len(pdf_files)} ไฟล์ (รวม {len(splits)} chunks) เรียบร้อยแล้ว")
    return True

@st.cache_resource(show_spinner=False)
def _get_embeddings():
    """โหลด Embedding Model ครั้งเดียว แล้ว cache ไว้ตลอด session"""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

@st.cache_resource(show_spinner=False)
def _get_vectorstore():
    """โหลด ChromaDB ครั้งเดียว แล้ว cache ไว้ตลอด session"""
    if not os.path.exists(DB_DIR):
        return None
    return Chroma(persist_directory=DB_DIR, embedding_function=_get_embeddings())

def get_qa_chain(api_key, gemini_api_key=None, mode="chat", provider=None):
    """
    สร้าง RAG Chain สำหรับการตอบคำถาม
    api_key: OpenRouter API Key
    gemini_api_key: Google Gemini API Key
    mode: "chat" หรือ "audit"
    provider: "openrouter" หรือ "gemini" (บังคับใช้ตัวใดตัวหนึ่ง)
    """
    embeddings = _get_embeddings()

    # โหลดจากฐานข้อมูลเดิม (cached)
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return None

    retriever_obj = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # --- เพิ่มการทำความสะอาดข้อความทันที (Immediate Cleaning) ---
    def wrap_clean_docs(docs):
        for doc in docs:
            doc.page_content = clean_thai_text(doc.page_content)
        return docs
    retriever = retriever_obj | RunnableLambda(wrap_clean_docs)
    
    llm = None
    
    # 1. พยายามใช้ OpenRouter (ถ้าไม่ได้บังคับเป็น gemini)
    if (not provider or provider == "openrouter") and api_key:
        models_to_try = [
            "google/gemma-3-27b-it:free",
            "google/gemma-3-4b-it:free",
            "google/gemma-2-9b-it:free"
        ]
        
        for model_name in models_to_try:
            try:
                llm = ChatOpenAI(
                    model=model_name,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    temperature=0,
                    max_retries=1 
                )
                if llm: break
            except:
                continue

    # 2. พยายามใช้ Gemma/Gemini ผ่าน Google AI Studio key (ถ้าไม่ได้บังคับเป็น openrouter)
    if not llm and (not provider or provider == "gemini") and gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_models_to_try = [
            "gemma-3-27b-it",
            "gemma-3-4b-it",
            "gemini-2.0-flash",
        ]
        for model_name in google_models_to_try:
            try:
                is_gemma = model_name.startswith("gemma")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=gemini_api_key,
                    temperature=0,
                    max_retries=2,
                    convert_system_message_to_human=is_gemma
                )
                break
            except:
                continue

    # สุดท้ายถ้ายังไม่ได้อะไรเลย ให้ Error
    if not llm:
        return None

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
            "คุณคือ AI ที่ปรึกษาด้านระเบียบการพัสดุและการเงิน ของ สจล. คุณจะตอบคำถามอย่างสุภาพและแม่นยำ\n\n"
            "### [KMITL/Public Procurement Ground Truth]\n"
            "- หลักการจัดซื้อจัดจ้าง (พรบ. 2560 มาตรา 8): ต้องประกอบด้วย 4 ประการ:\n"
            "  1. ความคุ้มค่า (Value for Money)\n"
            "  2. ความโปร่งใส (Transparency)\n"
            "  3. ประสิทธิภาพและประสิทธิผล (Efficiency and Effectiveness)\n"
            "  4. การตรวจสอบได้ (Accountability)\n"
            "- ราคากลาง: ราคาที่ใช้เป็นฐานในการเปรียบเทียบราคา ซึ่งสามารถจัดซื้อจัดจ้างได้จริง อ้างอิงตามลำดับ (1. คณะกรรมการราคากลาง, 2. กรมบัญชีกลาง, 3. สำนักงบประมาณ, 4. สืบราคาตลาด, 5. ราคาซื้อล่าสุดใน 2 ปีงบประมาณ)\n"
            "- e-catalog: ระบบบัญชีรายชื่อสินค้าและบริการอิเล็กทรอนิกส์ที่ผู้ขายลงทะเบียนไว้บนแพลตฟอร์ม e-GP ของกรมบัญชีกลาง เพื่อให้หน่วยงานของรัฐจัดซื้อสินค้ามาตรฐานได้สะดวกและโปร่งใส\n"
            "- e-bidding: ใช้กับวงเงินเกิน 500,000 บาท\n"
            "- วิธีเฉพาะเจาะจง: วงเงินไม่เกิน 500,000 บาท; วงเงินไม่เกิน 100,000 บาท ไม่ต้องทำสัญญาเป็นหนังสือ และใช้ผู้ตรวจรับคนเดียวได้\n"
            "- วิธีคัดเลือก: ใช้เมื่อผู้ขายที่คุณสมบัติครบน้อยกว่า 3 ราย / มีความจำเป็นเร่งด่วน / ต้องการความเชี่ยวชาญเฉพาะ / ต้องเก็บความลับ\n"
            "- ผู้ถือหุ้นรายใหญ่: ถือหุ้นเกินกว่าร้อยละ 25 ของจำนวนหุ้นทั้งหมด\n"
            "- การอุทธรณ์ผลจัดซื้อจัดจ้าง: ยื่นอุทธรณ์ภายใน 7 วันทำการนับแต่วันทราบผล; คณะกรรมการต้องพิจารณาให้แล้วเสร็จภายใน 30 วัน\n"
            "- อำนาจหัวหน้าหน่วยงาน: วิธีคัดเลือก ≤ 100 ล้าน, e-bidding ≤ 200 ล้าน, เฉพาะเจาะจง ≤ 50 ล้านบาท\n"
            "- เว็บไซต์ e-GP: http://egp3uat.cgd.go.th/EGPWeb/jsp/\n"
            "- e-Market (RFQ) ขั้นตอนหลัก: เพิ่มโครงการ → รายงานขอซื้อขอจ้าง/ร่างเอกสาร → ประกาศเชิญชวน → บันทึกผู้ชนะ → อนุมัติสั่งซื้อ → แต่งตั้งคณะกรรมการ → ร่างประกาศผู้ชนะ → ประกาศผลในระบบ\n"
            "- e-Market ค่าเริ่มต้น: E6 ระยะเวลาเผยแพร่เว็บ = 3 วันทำการ; E8 ยื่นราคาไม่น้อยกว่า = 15 วัน; E9 ส่งมอบพัสดุไม่เกิน = 7 วัน\n"
            "- e-Market ค่าเริ่มต้นสัญญา: R1 ประเภทสัญญา = สัญญาซื้อขายทั่วไป; R2 ทำสัญญาภายใน = 7 วัน; R3 หลักประกันสัญญา = ร้อยละ 5.00; R4 ประเภทค่าปรับ = ปรับเฉพาะที่ยังไม่ส่งมอบ; R5 อัตราค่าปรับ = ร้อยละ 0.20; R6-8 รับประกันชำรุดบกพร่อง ≥ 1 เดือน; R9 แก้ไขซ่อมแซมภายใน = 15 วัน\n"
            "- รายการพิจารณาสีแดงใน e-Market: หมายถึง รายการที่เสนอเกินวงเงินงบประมาณ หรือจำนวนผู้เสนอราคาน้อยกว่าที่กำหนด\n\n"
            "### [RULES]\n"
            "1. ตอบคำถามตามระเบียบจากข้อมูลที่ได้รับ (Context) เป็นลำดับแรก อ้างอิงเลขหน้าหรือข้อให้ชัดเจน\n"
            "2. **ห้าม** พูดเรื่อง [STATUS: PASS/FAIL] หรือกฎลำดับความสำคัญ (Priority Rules) ในโหมดนี้เด็ดขาด\n"
            "3. หาก Context ที่ได้รับไม่มีข้อมูลที่เกี่ยวข้อง ให้ใช้ Ground Truth ด้านบนหรือความรู้เกี่ยวกับระเบียบจัดซื้อจัดจ้างของรัฐ พ.ศ. 2560 ตอบแทนได้ โดยระบุว่า '(อ้างอิงจากระเบียบกระทรวงการคลัง พ.ศ. 2560)'\n"
            "4. รักษามารยาทและภาพลักษณ์ที่ดีของสถาบัน\n"
            "5. ปิดท้ายด้วยคำแนะนำให้ติดต่อ 'กองคลัง สจล.' สำหรับรายละเอียดเพิ่มเติมเสมอ\n\n"
            "--- เน้นให้คำแนะนำที่ครอบคลุมและอ้างอิงตามเนื้อหาในเอกสารที่ค้นหาได้จริง ---"
        )
        human_message = "Context: {context}\n\nคำถามจากผู้ใช้: {input}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])

    def run_rag(inputs: dict) -> dict:
        question = inputs["input"]
        docs = retriever.invoke(question)
        context_str = "\n\n".join(doc.page_content for doc in docs)
        messages = prompt.format_messages(context=context_str, input=question)
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer, "context": docs}

    return RunnableLambda(run_rag)
