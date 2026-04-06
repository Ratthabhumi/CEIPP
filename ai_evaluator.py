import os
import json
import time
import re

# พยายามโหลด st.secrets และดึง API KEY
try:
    import streamlit as st
    api_key = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("❌ ไม่พบ API Key กรุณาระบุใน .streamlit/secrets.toml หรือ Environment Variables")
    exit(1)

import warnings
warnings.filterwarnings("ignore")

from langchain_google_genai import ChatGoogleGenerativeAI

def llm_as_a_judge(report_file="evaluation_report.json"):
    print("="*60)
    print("⚖️ เริ่มกระบวนการประเมิน RAG ด้วย LLM-as-a-Judge (เหมือน RAGAS)")
    print("="*60)

    if not os.path.exists(report_file):
        print(f"❌ ไม่พบไฟล์ {report_file} (รบกวนรัน python evaluate.py ใหม่อีกรอบเพื่อให้แบนบันทึก context_texts ด้วยครับ)")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ตรวจสอบว่าในไฟล์มี context_texts หรือยัง
    if len(data) > 0 and "context_texts" not in data[0]:
        print("❌ ไฟล์ evaluation_report.json เป็นเวอร์ชันเก่า (ไม่มี Context)")
        print("กรุณารันคำสั่ง 'python evaluate.py' ใหม่อีกครั้งก่อนใช้งานตัวประเมินนี้ครับ")
        return

    # ใช้ Gemma 3 4B ผ่าน LangChain_Google_GenAI เพิ่อเลี่ยง Deprecation Warning
    os.environ["GOOGLE_API_KEY"] = api_key
    judge_model = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)
    
    graded_results_count = 0
    
    for i, item in enumerate(data, 1):
        q_id = item["question_id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        ai_answer = item["ai_answer"]
        # ตัด Context ให้สั้นลงเพื่อไม่ให้เกิน Token Limit
        context_texts = "\n---\n".join(item.get("context_texts", ["(ไม่พบ Context)"]))
        context_texts = context_texts[:2000] + "..." if len(context_texts) > 2000 else context_texts
        
        print(f"\nกำลังตรวจข้อสอบข้อที่ [{q_id}/{len(data)}]...")
        
        prompt = f"""
        คุณคือ "ผู้เชี่ยวชาญด้านการประเมินผล AI แชตบอท (LLM-as-a-Judge)" 
        หน้าที่ของคุณคือการให้คะแนนการประมวลผลของระบบ RAG (Retrieval-Augmented Generation) 
        โดยวิเคราะห์จาก 3 องค์ประกอบ (ให้คะแนน 1-5 โดยที่ 5 คือดีที่สุด)

        [ข้อมูลสำหรับการประเมิน]
        Question (คำถามจากผู้ใช้): {question}
        Ground Truth (เฉลยที่ถูกต้อง): {expected_answer}
        Retrieved Context (ข้อมูลที่ดึงมาจาก PDF): {context_texts}
        AI Answer (คำตอบจากแชทบอทที่จะให้คะแนน): {ai_answer}

        [เกณฑ์การให้คะแนน]
        1. Faithfulness (คะแนน 1-5): คำตอบของ AI อ้างอิงจากชิ้นส่วนเอกสาร (Retrieved Context) ได้อย่างซื่อสัตย์หรือไม่? 
           มีการมั่วข้อมูล (Hallucination) หรือแต่งเนื้อหาขึ้นมาเองที่ไม่ปรากฏใน Context หรือไม่? (ถ้าไม่มั่ว ให้ 5)
        2. Answer Relevance (คะแนน 1-5): คำตอบของ AI ตรงกับสิ่งที่ผู้ใช้ถาม (Question) แค่ไหน? 
           ตอบตรงประเด็นหรือไม่กว้างเกินไปใช่ไหม?
        3. Context Precision (คะแนน 1-5): สารสนเทศที่ระบบดึงมาให้ (Retrieved Context) มีเนื้อหาของเฉลย (Ground Truth) ประกอบอยู่ครบถ้วนและชัดเจนแค่ไหน? 
           (ถ้าดึงเอกสารมาถูกหน้าเป๊ะๆ จะได้ 5, ถ้าดึงอะไรมาไม่รู้เรื่องเลยได้ 1)

        กรุณาส่งออกผลลัพธ์เป็นรูปแบบ JSON เท่านั้น ห้ามมีข้อความอื่นปะปน
        รูปแบบ:
        {{
            "Faithfulness": <คะแนน 1-5>,
            "Answer_Relevance": <คะแนน 1-5>,
            "Context_Precision": <คะแนน 1-5>,
            "Reasoning": "<อธิบายเหตุผลสั้นๆ 2-3 บรรทัด ว่าทำไมถึงให้คะแนนเท่านี้>"
        }}
        """
        
        # ตรวจสอบว่าเคยประเมินผ่านไปแล้วหรือยัง
        score_dict = item.get("evaluation_scores", {})
        if score_dict and "error" not in score_dict:
            # ข้ามข้อที่ประเมินไปแล้ว
            graded_results_count += 1
            continue
        
        try:
            # ใช้ LangChain invoke
            response = judge_model.invoke(prompt)
            # แกะ JSON ออกมาจากข้อความที่ตอบกลับมา
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                judgement = json.loads(json_match.group())
            else:
                raise ValueError("ไม่พบ JSON ในคำตอบของกรรมการ")
            
            print(f"  ✅ Faithfulness: {judgement.get('Faithfulness')}/5")
            print(f"  ✅ Answer Relevance: {judgement.get('Answer_Relevance')}/5")
            print(f"  ✅ Context Precision: {judgement.get('Context_Precision')}/5")
            print(f"  💬 เหตุผลจากกรรมการ: {judgement.get('Reasoning')}")
            
            item["evaluation_scores"] = judgement
            graded_results_count += 1
            
            # บันทึกทับไฟล์เดิมทันทีเพื่อ Auto-save
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
        except Exception as e:
            print(f"  ❌ กรรมการประมวลผลคลาดเคลื่อน: {e}")
            item["evaluation_scores"] = {"error": str(e)}
            # แม้จะ error ก็ auto-save เผื่อกลับมาดู
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
        time.sleep(15) # หน่วงเวลาให้นานขึ้นเพื่อไม่ให้ Token limit เตะ
            
    # (เซฟไปทีละข้อแล้ว)
    print("\n" + "="*60)
    print(f"🎉 การประเมินเสร็จสิ้น! บันทึกคะแนนสำเร็จทั้งหมด {graded_results_count} ข้อลงในไฟล์ {report_file} เรียบร้อยครับ")
    print("="*60)

if __name__ == "__main__":
    llm_as_a_judge()
