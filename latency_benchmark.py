import time
import json
import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./chroma_db_v2"
GOLDEN_DATASET = "./golden_dataset.json"
EMBEDDING_MODEL = "BAAI/bge-m3"
TEST_SAMPLE = 20
K_VALUES = [10, 15]


def load_questions(n):
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item["question"] for item in dataset[:n]]
    print(f"Loaded {len(questions)} questions")
    return questions


def run_benchmark(vectorstore, embeddings, questions, k):
    latencies = []
    for q in questions:
        start = time.perf_counter()
        embeddings.embed_query(q)
        vectorstore.similarity_search(q, k=k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    return {
        "k": k,
        "n_queries": len(latencies),
        "avg_s": round(np.mean(latencies), 3),
        "min_s": round(np.min(latencies), 3),
        "max_s": round(np.max(latencies), 3),
        "p95_s": round(np.percentile(latencies, 95), 3),
        "latencies": [round(l, 3) for l in latencies],
    }


def main():
    print("Retrieval Latency Benchmark")
    print(f"Model: {EMBEDDING_MODEL} | K values: {K_VALUES} | Runs: {TEST_SAMPLE}")
    print("-" * 50)

    if not os.path.exists(DB_DIR):
        print(f"ChromaDB not found at {DB_DIR}")
        return
    if not os.path.exists(GOLDEN_DATASET):
        print("golden_dataset.json not found")
        return

    questions = load_questions(TEST_SAMPLE)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Loading ChromaDB...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    results = {}
    for k in K_VALUES:
        print(f"\nBenchmarking k={k}...")
        r = run_benchmark(vectorstore, embeddings, questions, k)
        results[k] = r
        print(f"  avg: {r['avg_s']}s | min: {r['min_s']}s | max: {r['max_s']}s | p95: {r['p95_s']}s")

    if len(K_VALUES) >= 2:
        k_a, k_b = K_VALUES[0], K_VALUES[1]
        avg_a = results[k_a]["avg_s"]
        avg_b = results[k_b]["avg_s"]
        print("\nComparison:")
        print(f"  k={k_a}: {avg_a}s")
        print(f"  k={k_b}: {avg_b}s")
        if avg_a < avg_b:
            diff = round(((avg_b - avg_a) / avg_b) * 100, 1)
            print(f"  k={k_a} is faster by {diff}%")
        elif avg_b < avg_a:
            diff = round(((avg_a - avg_b) / avg_a) * 100, 1)
            print(f"  k={k_b} is faster by {diff}%")
        else:
            print("  No significant difference")

    output_file = "latency_benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
