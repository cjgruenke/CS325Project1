#!/usr/bin/env python3
"""
Stage 3: Create embeddings using OpenAI (text-embedding-3-small / text-embedding-3-large)

Outputs:
 - jobs_embeddings.jsonl   (one JSON object per line: {job_key, embedding, meta})
 - resume_embedding.json   (single JSON with resume embedding)
 - jobs_embeddings.npy     (numpy array of embeddings; optional)

Usage:
  1) pip install requests tqdm numpy
  2) export OPENAI_API_KEY="sk-..."
  3) python stage3_embed_openai.py

Notes:
 - Be mindful of costs. This script batches to reduce HTTP overhead.
 - Default model: text-embedding-3-small (change MODEL variable below to "text-embedding-3-large" if desired).
"""

import os
import time
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
import numpy as np

# -----------------------
# CONFIG
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY environment variable before running.")

# Pick model: "text-embedding-3-small" (default, cost-efficient) or "text-embedding-3-large"
MODEL = "text-embedding-3-small"

# Input files (produced by Stage 2)
CLEANED_JOBS_JSON = Path("cleaned_jobs.json")
PROCESSED_RESUME = Path("processed_resume.txt")

# Output files
OUT_JSONL = Path("jobs_embeddings.jsonl")
OUT_NPY = Path("jobs_embeddings.npy")
OUT_RESUME = Path("resume_embedding.json")

# Embedding API settings
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# Batching & rate-limit strategy (tune to your quota)
BATCH_SIZE = 64               # number of texts per request (OpenAI supports arrays up to token limit; keep moderate)
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0         # seconds
MAX_BACKOFF = 30.0

# Limits: max tokens per input is model dependent; keep job descriptions reasonably short
# If your text is very long, consider chunking (not done automatically here).
TRUNCATE_INPUT_CHARS = 32000  # if a description is huge, cut it (safe default)

# -----------------------
# Helpers
# -----------------------
def load_jobs() -> List[Dict[str, Any]]:
    if not CLEANED_JOBS_JSON.exists():
        raise FileNotFoundError(f"{CLEANED_JOBS_JSON} not found. Run Stage 2 first.")
    with CLEANED_JOBS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_resume_text() -> str:
    if not PROCESSED_RESUME.exists():
        raise FileNotFoundError(f"{PROCESSED_RESUME} not found. Generate processed resume in Stage 2.")
    return PROCESSED_RESUME.read_text(encoding="utf-8")

def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def call_openai_embeddings(inputs: List[str], model: str) -> List[List[float]]:
    """
    Call OpenAI Embeddings endpoint with a batch (list of strings).
    Returns list of embedding vectors (one per input).
    Implements retry/backoff on transient errors.
    """
    payload = {"model": model, "input": inputs}
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENAI_EMBED_URL, headers=HEADERS, json=payload, timeout=60)
            if resp.status_code == 200:
                j = resp.json()
                # response: { data: [ { embedding: [...], index: 0 }, ... ] }
                if "data" in j:
                    embeddings = [item["embedding"] for item in j["data"]]
                    return embeddings
                else:
                    raise RuntimeError(f"Unexpected response shape: {j}")
            elif resp.status_code in (429, 502, 503, 504):
                # transient: backoff and retry
                print(f"Transient error {resp.status_code}. Body: {resp.text[:400]}. Backing off {backoff}s (attempt {attempt}).")
            else:
                # non-retryable error (bad auth, validation, etc.)
                raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")
        except requests.RequestException as e:
            print(f"Request exception: {e}. Backing off {backoff}s (attempt {attempt}).")
        time.sleep(backoff + random.random() * 0.1)
        backoff = min(backoff * 2.0, MAX_BACKOFF)
    raise RuntimeError(f"Failed to get embeddings after {MAX_RETRIES} attempts.")

# -----------------------
# Main flow
# -----------------------
def main():
    jobs = load_jobs()
    print(f"Loaded {len(jobs)} cleaned job records from {CLEANED_JOBS_JSON}")

    # Prepare inputs and map to job ids
    inputs = []
    metas = []
    for r in jobs:
        # choose what to embed: combine title, company, location, and description
        title = (r.get("title") or "").strip()
        company = (r.get("companyName") or "").strip()
        location = (r.get("location_normalized") or r.get("location_raw") or "").strip()
        desc = (r.get("description") or "")
        text = " ".join([part for part in [title, company, location, desc] if part])
        if len(text) > TRUNCATE_INPUT_CHARS:
            text = text[:TRUNCATE_INPUT_CHARS] + " ... (truncated)"
        inputs.append(text)
        metas.append({
            "job_key": r.get("job_key"),
            "title": title,
            "companyName": company,
            "location": location
        })

    # batch and call API
    all_embeddings = []
    out_lines = []
    total_batches = math.ceil(len(inputs) / BATCH_SIZE) if inputs else 0
    idx = 0
    for chunk in tqdm(list(chunked(inputs, BATCH_SIZE)), desc="Embedding batches", unit="batch"):
        # call embeddings for this chunk
        embeddings = call_openai_embeddings(chunk, MODEL)
        for e in embeddings:
            meta = metas[idx]
            line = {"job_key": meta.get("job_key"), "meta": meta, "embedding": e}
            out_lines.append(line)
            all_embeddings.append(e)
            idx += 1
        # pause to avoid bursts 
        time.sleep(0.1)

    # write JSONL
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for obj in out_lines:
            f.write(json.dumps(obj) + "\n")
    print(f"Wrote job embeddings to {OUT_JSONL} ({len(out_lines)} vectors)")

    #save numpy array for quick load
    arr = np.array(all_embeddings, dtype=np.float32)
    np.save(OUT_NPY, arr)
    print(f"Wrote numpy embeddings to {OUT_NPY} (shape: {arr.shape})")

    # Resume embedding (single call)
    resume_text = read_resume_text().strip()
    if resume_text:
        if len(resume_text) > TRUNCATE_INPUT_CHARS:
            resume_text = resume_text[:TRUNCATE_INPUT_CHARS] + " ... (truncated)"
        resume_embedding = call_openai_embeddings([resume_text], MODEL)[0]
        resume_obj = {"model": MODEL, "embedding": resume_embedding, "text_sample": resume_text[:2000]}
        with OUT_RESUME.open("w", encoding="utf-8") as f:
            json.dump(resume_obj, f, indent=2)
        print(f"Wrote resume embedding to {OUT_RESUME}")
    else:
        print("Resume text empty; skipped resume embedding.")

if __name__ == "__main__":
    main()