import os
import base64
import json
import re
import google.generativeai as genai

PROMPT = """You are a Thai financial document OCR expert. Extract data from the receipt image and return ONLY valid JSON, no other text.

Required JSON format (use null if not found, numbers must be plain floats with no commas):
{
    "receipt_type": "ประเภทเอกสาร (ใบเสร็จรับเงิน / ใบกำกับภาษีเต็มรูป / ใบสำคัญรับเงิน / ใบรับรองการจ่ายเงิน / ใบเสนอราคา / ใบสั่งซื้อ / บันทึกขอให้จัดหา) หรือ null",
    "buyer_name": "ชื่อลูกค้า/ผู้ซื้อที่ระบุในบิล (เช่น สจล. หรือ ชื่อบุคลากร) หรือ null",
    "vendor_name": "ชื่อร้านค้า หรือ บริษัทที่ออกบิล หรือ null",
    "tax_id": "เลขประจำตัวผู้เสียภาษี 13 หลักของร้าน หรือ null",
    "transaction_date": "วันที่ในเอกสาร หรือ null",
    "total_amount": 0.0,
    "amount_in_words": "ตัวหนังสือกำกับยอดเงิน (เช่น 'หนึ่งร้อยบาทถ้วน') ถ้าไม่มีข้อความระบุ ให้ใส่ null",
    "has_receiver_signature": true หรือ false (ดูว่ามีรอยขีดเขียนลายเซ็นของผู้รับเงินในบิลหรือไม่),
    "has_paid_stamp": true หรือ false (ดูว่ามีหมึกประทับตาข้อความว่า 'จ่ายเงินแล้ว' พร้อมวันที่/ลายเซ็นหรือไม่),
    "items": [
        {"description": "item name", "price": 0.0}
    ]
}
"""

def extract_receipt_data(image_bytes: bytes, api_key: str):
    """
    ใช้ google.generativeai SDK ออพฟิเชียล เพื่อประสิทธิภาพที่ดีที่สุดกับภาษาไทย (Gemini 1.5 Flash)
    """
    if not api_key:
        raise ValueError("API Key is required for OCR extraction.")
        
    genai.configure(api_key=api_key)
    
    # ระบุชื่อรุ่นที่เสถียรที่สุด (รองรับ API v1 และ v1beta)
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
    ]
    
    # พยายามตรวจสอบรุ่นที่ใช้เนื้อหาภาพได้ (Vision/Multimodal)
    contents = [
        PROMPT,
        {"mime_type": "image/jpeg", "data": image_bytes}
    ]
    
    last_error = None
    for model_id in models_to_try:
        try:
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(contents)
            
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "ไม่สามารถแปลงผลลัพธ์เป็น JSON ได้", "raw_content": text}
                
        except Exception as e:
            last_error = str(e)
            continue
            
    # ถ้ายังไม่ได้ ให้ลองไล่ดูรุ่นที่รองรับ generateContent ใน API Key นี้
    try:
        available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for model_id in available_models:
            if "flash" in model_id.lower() or "pro" in model_id.lower():
                try:
                    model = genai.GenerativeModel(model_id)
                    response = model.generate_content(contents)
                    text = response.text
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                except:
                    continue
    except:
        pass

    return {"error": f"ไม่พบรุ่นโมเดลที่ใช้งานได้ (404 ทั้งหมด): {last_error}. โปรดตรวจสอบความถูกต้องของ API Key หรือโควตาการใช้งาน"}

def verify_receipt_rules(qa_chain, ocr_data):
    """
    ตรวจสอบข้อมูลที่ดึงมาจาก OCR โดยถาม RAG
    ส่งกลับเป็น Dictionary พร้อมสถานะ PASS/FAIL
    """
    if "error" in ocr_data:
        return {"analysis": f"ไม่สามารถตรวจสอบคำขอได้เนื่องจาก: {ocr_data['error']}", "status": "ERROR"}

    vendor_name = ocr_data.get("vendor_name", "ไม่ระบุ")
    total_amount = ocr_data.get("total_amount", 0.0)
    receipt_type = ocr_data.get("receipt_type", "ไม่ระบุประเภท")
    buyer_name = ocr_data.get("buyer_name", "ไม่ระบุ")
    transaction_date = ocr_data.get("transaction_date", "ไม่ระบุวันที่")
    items = ocr_data.get("items", [])
    amount_in_words_raw = ocr_data.get("amount_in_words", "ไม่มี")
    has_sig = "มี" if ocr_data.get("has_receiver_signature") else "ไม่มี"
    has_stamp = "มี" if ocr_data.get("has_paid_stamp") else "ไม่มี"

    items_text = "\n".join([f"- {it.get('description', '')}: {it.get('price', 0)} บาท" for it in items])

    question = (
        f"จงค้นหากฎเกณฑ์เกี่ยวกับ 'หลักฐานการเบิกจ่ายเงิน ใบเสร็จรับเงิน ใบสำคัญรับเงิน ตามระเบียบกระทรวงการคลัง' "
        f"และประเมินว่าเอกสารรูปนี้มีคุณสมบัติครบถ้วนพอจะมาเบิกเงินหรือไม่?\n\n"
        f"ข้อมูลที่สกัดได้จากเอกสาร (พฤติการณ์เอกสาร):\n"
        f"- ประเภทเอกสาร: '{receipt_type}'\n"
        f"- วันที่ในเอกสาร: '{transaction_date}'\n"
        f"- ยอดเงิน: {total_amount} บาท จากผู้ขาย: '{vendor_name}'\n"
        f"- ชื่อลูกค้า/ผู้ซื้อ: '{buyer_name}'\n"
        f"- จำนวนเงินตัวอักษรจากต้นฉบับ: '{amount_in_words_raw}'\n"
        f"- ลายมือชื่อผู้รับเงิน: {has_sig}\n"
        f"- ตราประทับ 'จ่ายเงินแล้ว': {has_stamp}\n"
        f"- รายการสินค้า:\n{items_text}\n\n"
        f"คำสั่งพิเศษ:\n"
        f"1. ห้ามคำนวณจำนวนเงินตัวอักษรใหม่เองเด็ดขาด ให้ใช้คำว่า '{amount_in_words_raw}' จากข้อมูล้นฉบับเท่านั้นในการสรุปผล\n"
        f"2. ประเมินความครบถ้วน และปิดท้ายคำตอบด้วยบรรทัดใหม่ที่มีเพียงคำว่า [STATUS: PASS] หากผ่านเกณฑ์ หรือ [STATUS: FAIL] หากต้องแก้ไข\n\n"
        f"คำถาม: เอกสารใบนี้มีรายละเอียดครบถ้วนตามระเบียบหรือไม่? อะไรคือสิ่งที่ขาดหายไป? โปรดอธิบายเหตุผลและอ้างอิงหลักเกณฑ์"
    )

    try:
        result = qa_chain.invoke({"input": question})
        answer = result["answer"]
        
        # ค้นหาสถานะในคำตอบ
        status = "FAIL"
        if "[STATUS: PASS]" in answer:
            status = "PASS"
        elif "[STATUS: FAIL]" in answer:
            status = "FAIL"
            
        clean_answer = answer.replace("[STATUS: PASS]", "").replace("[STATUS: FAIL]", "").strip()
        
        return {
            "analysis": clean_answer,
            "status": status
        }
    except Exception as e:
        return {
            "analysis": f"เกิดข้อผิดพลาดในการตรวจสอบตามกฎระเบียบ: {str(e)}",
            "status": "ERROR"
        }
