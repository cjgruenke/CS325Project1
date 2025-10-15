#!/usr/bin/env python3
"""
Stage 4: Similarity calculation and top job selection.

Reads:
 - jobs_embeddings.jsonl  (one JSON per line: {job_key, meta, embedding})
 - resume_embedding.json  (single JSON: { model, embedding, text_sample })

Produces:
 - top_jobs.json   (top N ranked results with similarity)
 - top_jobs.csv    (CSV summary)
 - prints top N to stdout

Usage:
  pip install numpy
  python stage4_similarity.py [--top N] [--jobs jobs_embeddings.jsonl] [--resume resume_embedding.json]
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import csv
import math

# -----------------------
# Defaults / paths
# -----------------------
DEFAULT_JOBS_FILE = Path("jobs_embeddings.jsonl")
DEFAULT_RESUME_FILE = Path("resume_embedding.json")
OUTPUT_JSON = Path("top_jobs.json")
OUTPUT_CSV = Path("top_jobs.csv")

# -----------------------
# Helpers
# -----------------------
def load_jobs_embeddings(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Jobs embeddings file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # expect keys: job_key, meta, embedding
            if "embedding" not in obj:
                raise ValueError("Each jobs JSONL line must contain an 'embedding' field.")
            records.append(obj)
    return records

def load_resume_embedding(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"Resume embedding file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "embedding" not in obj:
        raise ValueError("resume_embedding.json must contain an 'embedding' field.")
    return obj["embedding"]

def to_numpy(vec: List[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    return arr

def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between vectors a and b.
    Returns -1..1 (but for embeddings typically 0..1). Handles zero vectors.
    """
    # convert to float32
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cos = float(np.dot(a, b) / (norm_a * norm_b))
    # numerical safety: clip
    if math.isnan(cos):
        return 0.0
    return max(min(cos, 1.0), -1.0)

# -----------------------
# Main ranking function
# -----------------------
def rank_jobs(jobs_file: Path, resume_file: Path, top_n: int = 10):
    jobs = load_jobs_embeddings(jobs_file)
    resume_vec = load_resume_embedding(resume_file)
    resume_arr = to_numpy(resume_vec)

    results = []
    for rec in jobs:
        emb = rec.get("embedding")
        # some saved embeddings may be lists of lists or other shapes; coerce
        emb_arr = to_numpy(emb)
        sim = safe_cosine_similarity(resume_arr, emb_arr)
        meta = rec.get("meta") or {}
        # collect useful display fields with safe fallbacks
        title = meta.get("title") or meta.get("job_title") or rec.get("meta", {}).get("title")
        company = meta.get("companyName") or meta.get("company") or rec.get("meta", {}).get("companyName")
        location = meta.get("location") or meta.get("location_normalized") or meta.get("location_raw") or rec.get("meta", {}).get("location")
        results.append({
            "job_key": rec.get("job_key"),
            "title": title,
            "company": company,
            "location": location,
            "similarity": sim,
            "meta": meta
        })

    # sort by similarity descending
    results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    top_results = results_sorted[:top_n]

    # save outputs
    OUTPUT_JSON.write_text(json.dumps(top_results, indent=2, ensure_ascii=False), encoding="utf-8")
    # CSV
    csv_fields = ["rank", "job_key", "title", "company", "location", "similarity"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for i, r in enumerate(top_results, start=1):
            writer.writerow({
                "rank": i,
                "job_key": r.get("job_key"),
                "title": r.get("title") or "",
                "company": r.get("company") or "",
                "location": r.get("location") or "",
                "similarity": f"{r.get('similarity'):.6f}"
            })

    # print nicely
    print(f"\nTop {top_n} jobs (by cosine similarity):\n")
    print(f"{'Rank':<4} {'Sim':>7}  {'Title':<40} {'Company':<30} {'Location'}")
    print("-" * 110)
    for i, r in enumerate(top_results, start=1):
        sim = r["similarity"]
        title = (r["title"] or "")[:38]
        company = (r["company"] or "")[:28]
        location = (r["location"] or "")[:40]
        print(f"{i:<4} {sim:7.4f}  {title:<40} {company:<30} {location}")
    print("\nSaved top results to", OUTPUT_JSON, "and", OUTPUT_CSV)
    return top_results

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Rank job embeddings by similarity to resume embedding.")
    parser.add_argument("--top", type=int, default=10, help="Number of top jobs to output (default 10)")
    parser.add_argument("--jobs", type=str, default=str(DEFAULT_JOBS_FILE), help="Path to jobs_embeddings.jsonl")
    parser.add_argument("--resume", type=str, default=str(DEFAULT_RESUME_FILE), help="Path to resume_embedding.json")
    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    resume_path = Path(args.resume)
    top_n = args.top
    rank_jobs(jobs_path, resume_path, top_n)

if __name__ == "__main__":
    main()
