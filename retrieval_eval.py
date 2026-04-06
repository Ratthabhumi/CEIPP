"""
retrieval_eval.py
-----------------
วัดคุณภาพของ HuggingFace Embedding (ส่วน Retriever) แยกออกจาก LLM
โดยใช้ Metrics มาตรฐาน:
  - Recall@K          : มีคำตอบที่ถูกอยู่ใน K Chunk แรกไหม?
  - Precision@K       : ใน K Chunk ที่ดึงมา กี่ % ที่เกี่ยวข้องจริง?
  - MRR               : Chunk ที่เกี่ยวจริงอยู่อันดับที่เท่าไหร่?
  - Cosine Similarity : ความหมายของ Query กับ Chunk ใกล้กันแค่ไหน?

หมายเหตุ:
  - ใช้ sklearn cosine_similarity คำนวณตรงๆ (ChromaDB ส่ง L2 distance ซึ่งไม่ใช่ cosine)
  - ใช้ unicodedata.normalize เพื่อแก้ปัญหา Thai encoding ต่างกัน (ทำ vs ทํา)
"""

import os
import unicodedata
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from langchain_huggingface import HuggingFaceEmbeddings
import re
from langchain_chroma import Chroma

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DB_DIR         = "./chroma_db_v2"
GOLDEN_DATASET = "golden_dataset.json"
OUTPUT_FILE    = "retrieval_eval_report.json"
EMBEDDING_MODEL = "BAAI/bge-m3"
K = 10  # ตรงกับที่ใช้ใน rag_pipeline.py


# ──────────────────────────────────────────────
# HELPER: ตรวจว่า chunk เกี่ยวข้องกับ expected_answer ไหม
# โดยใช้ Keyword Overlap (ไม่ต้องมี ground-truth chunk IDs)
# ──────────────────────────────────────────────
def normalize_thai(text: str) -> str:
    """NFC normalization แก้ปัญหา ทำ (U+0E17 U+0E33) vs ทํา (U+0E17 U+0E4D U+0E32) """
    return unicodedata.normalize("NFC", text)


def tokenize_thai(text: str) -> list[str]:
    """
    แตกข้อความภาษาไทย/อังกฤษเป็น token โดย:
    - ตัดด้วย whitespace และเครื่องหมายวรรคตอน
    - แตก compound Thai token ที่ยาวเกิน 4 ตัวอักษรเป็น n-gram ย่อย
      เพื่อให้ match กับ chunk ที่ตัดคำต่างกันได้
    """
    puncts = r'[,.\(\)\[\]、:;!?\"\'๐-๙]'
    words = re.split(r'\s+', re.sub(puncts, ' ', text))
    tokens = set()
    for w in words:
        w = w.strip()
        if len(w) < 3:
            continue
        tokens.add(w)
        # สร้าง substring n-gram (4+ ตัวอักษร) เพื่อรับมือตัดคำต่างกัน
        for n in (4, 5, 6):
            for i in range(len(w) - n + 1):
                tokens.add(w[i:i+n])
    return list(tokens)


def is_chunk_relevant(expected_answer: str, chunk_text: str, min_matches: int = 2) -> bool:
    """
    คืน True ถ้า chunk_text มีคำสำคัญจาก expected_answer ครบตาม min_matches คำ
    - ใช้ n-gram tokenization เพื่อรับมือการตัดคำภาษาไทยที่ต่างกัน
    - Normalize Unicode ก่อนเพื่อแก้ปัญหา Thai encoding
    """
    norm_expected = normalize_thai(expected_answer)
    norm_chunk    = normalize_thai(chunk_text)

    keywords = tokenize_thai(norm_expected)
    if not keywords:
        return norm_expected in norm_chunk

    matches = sum(1 for kw in keywords if kw in norm_chunk)
    threshold = max(min_matches, int(len(keywords) * 0.15))
    return matches >= threshold


# ──────────────────────────────────────────────
# MAIN EVALUATION
# ──────────────────────────────────────────────
def evaluate_retrieval(k: int = K):
    print("=" * 65)
    print("🔬  Retrieval Evaluation — HuggingFace Embedding")
    print(f"    Model : {EMBEDDING_MODEL}")
    print(f"    K     : {k}")
    print("=" * 65)

    # 1. โหลด Golden Dataset
    if not os.path.exists(GOLDEN_DATASET):
        print(f"❌ ไม่พบไฟล์ {GOLDEN_DATASET}")
        return
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 2. โหลด Embedding Model และ ChromaDB
    print("\n⏳ กำลังโหลด Embedding Model (อาจใช้เวลาสักครู่ครั้งแรก)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not os.path.exists(DB_DIR):
        print(f"❌ ไม่พบ ChromaDB ที่ {DB_DIR}  กรุณา build_db ก่อนครับ")
        return
    print("⏳ กำลังโหลด ChromaDB...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # ──────────────────────────────────────────
    per_question_results = []

    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected_answer"]

        print(f"\n{'─'*65}")
        print(f"[{idx}/{len(dataset)}] 📝 {question}")
        print(f"  ✅ เฉลย : {expected[:80]}...")

        # 3. ดึง Top-K Chunks (ไม่ใช้ relevance_scores ของ ChromaDB เพราะส่ง L2 distance)
        docs = vectorstore.similarity_search(question, k=k)

        # 4. Embed query + chunks แล้วคำนวณ True Cosine Similarity
        query_vec  = np.array(embeddings.embed_query(question)).reshape(1, -1)
        chunk_vecs = np.array(embeddings.embed_documents([d.page_content for d in docs]))
        cosine_scores = sk_cosine(query_vec, chunk_vecs)[0].tolist()  # shape: (k,)

        # 5. คำนวณ Metrics
        precision_hits  = 0
        first_hit_rank  = None
        reciprocal_rank = 0.0

        chunk_details = []
        for rank, (doc, score) in enumerate(zip(docs, cosine_scores), start=1):
            relevant = is_chunk_relevant(expected, doc.page_content)

            if relevant:
                precision_hits += 1
                if first_hit_rank is None:
                    first_hit_rank  = rank
                    reciprocal_rank = 1.0 / rank

            chunk_details.append({
                "rank"        : rank,
                "cosine_score": round(score, 4),
                "is_relevant" : relevant,
                "preview"     : doc.page_content[:120].replace("\n", " ") + "…",
            })

        # Recall@K = 1 ถ้ามี chunk ที่ตรงอยู่ใน K อันดับแรก
        recall_at_k   = 1 if first_hit_rank is not None else 0
        precision_at_k = precision_hits / k
        avg_cosine    = float(np.mean(cosine_scores))
        max_cosine    = float(np.max(cosine_scores))

        # แสดงผล
        print(f"\n  📊 Metrics:")
        print(f"     Recall@{k:<3}     = {'✅ 1 (พบ)' if recall_at_k else '❌ 0 (ไม่พบ)'}")
        print(f"     Precision@{k:<3}  = {precision_at_k:.2%}  ({precision_hits}/{k} chunks เกี่ยวข้อง)")
        print(f"     MRR              = {reciprocal_rank:.4f}  (first hit at rank {first_hit_rank})")
        print(f"     Avg Cosine       = {avg_cosine:.4f}")
        print(f"     Max Cosine       = {max_cosine:.4f}")

        per_question_results.append({
            "question_id"  : idx,
            "question"     : question,
            "expected"     : expected,
            "metrics": {
                f"Recall@{k}"    : recall_at_k,
                f"Precision@{k}" : round(precision_at_k, 4),
                "MRR"            : round(reciprocal_rank, 4),
                "First_Hit_Rank" : first_hit_rank,
                "Avg_Cosine"     : round(avg_cosine, 4),
                "Max_Cosine"     : round(max_cosine, 4),
            },
            "retrieved_chunks": chunk_details,
        })

    # ──────────────────────────────────────────
    # 5. สรุปภาพรวม
    # ──────────────────────────────────────────
    avg_recall    = float(np.mean([r["metrics"][f"Recall@{k}"]    for r in per_question_results]))
    avg_precision = float(np.mean([r["metrics"][f"Precision@{k}"] for r in per_question_results]))
    avg_mrr       = float(np.mean([r["metrics"]["MRR"]            for r in per_question_results]))
    avg_cosine    = float(np.mean([r["metrics"]["Avg_Cosine"]     for r in per_question_results]))

    summary = {
        "model"            : EMBEDDING_MODEL,
        "K"                : k,
        "total_questions"  : len(dataset),
        "overall_metrics": {
            f"Avg_Recall@{k}"    : round(avg_recall,    4),
            f"Avg_Precision@{k}" : round(avg_precision, 4),
            "Avg_MRR"            : round(avg_mrr,       4),
            "Avg_Cosine_Score"   : round(avg_cosine,    4),
        },
        "interpretation": {
            "Recall@K"   : "1 = พบ chunk ที่เกี่ยวข้องใน K อันดับแรก, 0 = ไม่พบ",
            "Precision@K": "สัดส่วน chunk ที่เกี่ยวข้องใน K อันดับแรก",
            "MRR"        : "ค่าใกล้ 1 = chunk ที่ดีอยู่อันดับสูง, ใกล้ 0 = อยู่ท้าย",
            "Cosine"     : "ค่าใกล้ 1.0 = ความหมายใกล้กันมาก, ใกล้ 0 = ห่างกัน",
        },
        "per_question": per_question_results,
    }

    # 6. บันทึก JSON Report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*65}")
    print("📊  สรุปผล Retrieval Evaluation (ภาพรวม)")
    print(f"{'='*65}")
    print(f"  Avg Recall@{k}     = {avg_recall:.2%}")
    print(f"  Avg Precision@{k}  = {avg_precision:.2%}")
    print(f"  Avg MRR            = {avg_mrr:.4f}  (max = 1.0)")
    print(f"  Avg Cosine Score   = {avg_cosine:.4f}  (max = 1.0)")
    print(f"\n✅ บันทึกรายงานแบบละเอียดไว้ที่ → {OUTPUT_FILE}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_retrieval(k=K)
