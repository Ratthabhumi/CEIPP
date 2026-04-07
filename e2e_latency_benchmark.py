import time
import json
import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db_v2"
GOLDEN_DATASET = "./golden_dataset.json"
EMBEDDING_MODEL = "BAAI/bge-m3"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
TEST_SAMPLE = 5
K_VALUES = [10, 15]

SYSTEM_PROMPT = (
    "คุณคือ AI ที่ปรึกษาด้านระเบียบพัสดุของ สจล. "
    "ตอบคำถามจาก Context เท่านั้น สั้นๆ กระชับ"
)


def load_questions(n):
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item["question"] for item in dataset[:n]]
    print(f"Loaded {len(questions)} questions")
    return questions


def build_chain(k, embeddings, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        temperature=0,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nคำถาม: {input}"),
    ])

    def run(inputs):
        docs = retriever.invoke(inputs["input"])
        ctx = "\n\n".join(d.page_content for d in docs)
        msgs = prompt.format_messages(context=ctx, input=inputs["input"])
        return llm.invoke(msgs).content

    return run


def run_benchmark(run_fn, questions, k):
    latencies = []
    print(f"\nTesting k={k} ({len(questions)} questions, E2E with LLM)")
    print("-" * 50)

    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q[:60]}...")
        for attempt in range(3):
            start = time.perf_counter()
            try:
                run_fn({"input": q})
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                print(f"    {elapsed:.2f}s")
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = (attempt + 1) * 15
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    Error: {err[:100]}")
                    break
        else:
            print(f"    Skipped after 3 retries")

        if i < len(questions):
            print(f"    Waiting 12s...")
            time.sleep(12)

    if not latencies:
        return {"k": k, "error": "All requests failed"}

    return {
        "k": k,
        "n_queries": len(latencies),
        "avg_s": round(np.mean(latencies), 2),
        "min_s": round(np.min(latencies), 2),
        "max_s": round(np.max(latencies), 2),
        "p95_s": round(np.percentile(latencies, 95), 2),
        "latencies": [round(l, 2) for l in latencies],
    }


def main():
    print("E2E Latency Benchmark (Retrieval + LLM)")
    print(f"Model: {LLM_MODEL} | K: {K_VALUES} | Queries: {TEST_SAMPLE} each")
    print("=" * 50)

    if not os.path.exists(DB_DIR):
        print(f"ChromaDB not found at {DB_DIR}")
        return

    questions = load_questions(TEST_SAMPLE)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Loading ChromaDB...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    results = {}
    for k in K_VALUES:
        run_fn = build_chain(k, embeddings, vectorstore)
        r = run_benchmark(run_fn, questions, k)
        results[k] = r
        if "error" not in r:
            print(f"\nk={k} summary: avg={r['avg_s']}s | min={r['min_s']}s | max={r['max_s']}s")

    print("\nComparison:")
    r10 = results.get(10, {})
    r15 = results.get(15, {})
    if "avg_s" in r10 and "avg_s" in r15:
        print(f"  k=10 avg: {r10['avg_s']}s")
        print(f"  k=15 avg: {r15['avg_s']}s")
        faster = 10 if r10["avg_s"] < r15["avg_s"] else 15
        slower_avg = max(r10["avg_s"], r15["avg_s"])
        faster_avg = min(r10["avg_s"], r15["avg_s"])
        pct = round(((slower_avg - faster_avg) / slower_avg) * 100, 1)
        print(f"  k={faster} is faster ({pct}% reduction)")

    with open("e2e_latency_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("\nResults saved to e2e_latency_results.json")


if __name__ == "__main__":
    main()
