# Job Matcher — Stage 1–4 Pipeline

This repository contains a full pipeline to collect job postings (via RapidAPI Indeed Scraper),
clean and preprocess job text, compute embeddings with OpenAI, and rank jobs similar to a single
resume. The pipeline is split into stages and an orchestrator to run them end-to-end.

**Stages**
- Stage 1 — Data acquisition: `src/stage1_indeed_scraper.py` (RapidAPI)
- Stage 2 — Preprocessing: `src/stage2_preprocess.py`
- Stage 3 — Embeddings: `src/stage3_embed_openai.py` (OpenAI)
- Stage 4 — Similarity & ranking: `src/stage4_similarity.py`
- `src/orchestrator.py` — runs all enabled stages in order

> **Important:** Do **not** commit raw files containing personally-identifiable information (PII).
> Use `scripts/anonymize_data.py` to create safe anonymized samples for publishing.

---

## Quick start (local)

### 1. Clone repository
```bash
git clone <your-github-repo-url>
cd job-matcher
