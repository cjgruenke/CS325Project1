# CS325 Project 1 — AI Job Matcher

**Author:** Cole Jacob Gruenke  
**Course:** CS325 — Operating Systems / Data Systems Integration  
**Semester:** Fall 2025  

---

## Overview

This project implements a **complete data pipeline** that collects, preprocesses, embeds, and matches **job postings** from **Indeed’s RapidAPI endpoint** with a **student’s resume** using **OpenAI embeddings**.

The goal is to produce a ranked list of job postings most relevant to the resume by computing **cosine similarity** between the job vectors and the resume vector.

A single orchestrator script (`main.py`) allows all stages to run automatically in sequence:
> Stage 1 → Stage 2 → Stage 3 → Stage 4

---

---

## Stage Breakdown

### **Stage 1 — Data Acquisition**
**Goal:** Retrieve job postings using the Indeed Scraper API (RapidAPI).

**Tools:** `requests`, `BeautifulSoup4`  
**API Endpoint:** `https://indeed-scraper-api.p.rapidapi.com/api/job`

**Key Features**
- Sends POST requests to the Indeed RapidAPI endpoint  
- Handles API authentication with environment variable `RAPIDAPI_KEY`  
- Supports retries and error handling  
- Saves JSON response for later processing  

**Output:**  
- `data/raw/indeed_response_raw.json`

**Ethical Considerations:**
- Uses the API (not direct scraping)  
- Respects rate limits (`time.sleep()`)  
- Honors `robots.txt` and platform ToS  

---

### **Stage 2 — Data Preprocessing**
**Goal:** Clean and normalize job posting text and parse the student’s resume.

**Tools:** `pandas`, `BeautifulSoup4`, `PyPDF2`, `re`  

**Processes:**
- Removes HTML tags, special characters, and extra whitespace  
- Normalizes inconsistent formatting (e.g., “St. Louis” → “Saint Louis, MO”)  
- Converts text to lowercase  
- Handles missing fields by filling with `"N/A"`  
- Parses and cleans the student’s resume  

**Outputs:**
- `cleaned_jobs.json`  
- `cleaned_jobs.csv`  
- `processed_resume.txt`  
- `resume_sections.json`

---

### **Stage 3 — Embedding Generation**
**Goal:** Convert text data into numerical vectors (embeddings).

**Model Used:** `text-embedding-3-small` (OpenAI)  
**Tools:** `requests`, `numpy`, `tqdm`  
**Endpoint:** `https://api.openai.com/v1/embeddings`  
**Environment Variable:** `OPENAI_API_KEY`

**Features:**
- Batches API calls for efficiency  
- Includes retry and exponential backoff for rate limits  
- Truncates overly long text safely  
- Stores embeddings for both job posts and resume  

**Outputs:**
- `jobs_embeddings.jsonl`  
- `resume_embedding.json`  
- `jobs_embeddings.npy`

---

### **Stage 4 — Similarity Calculation**
**Goal:** Find the jobs most similar to the student’s resume using **cosine similarity**.

**Tools:** `numpy`, `csv`  

**Process:**
1. Loads embeddings from Stage 3  
2. Computes cosine similarity between resume and all job vectors  
3. Sorts jobs by similarity (descending)  
4. Outputs top N results (default: top 10)

**Outputs:**
- `top_jobs.json`  
- `top_jobs.csv`


---

### **Stage 5 — Final Deliverables**
**Goal:** Package all project components into a clear and reproducible repository.

**Outputs:**
- Fully commented Python code for all stages  
- This `README.md`  
- `requirements.txt` for dependencies  
- `.gitignore` to exclude sensitive or large files  
- Optional anonymized dataset (`scripts/anonymize_data.py`)

**Ethical & Legal Notes:**
- Data collected only from permitted API endpoints  
- No personal information is stored or shared  
- Resume text is processed locally and never uploaded beyond OpenAI embedding API  

---

##  How to Use and Operate

### 1. Clone the repository
```bash
git clone https://github.com/cjgruenke/CS325Project1.git
cd CS325Project1

### 1. Clone the repository
git clone https://github.com/cjgruenke/CS325Project1.git
cd CS325Project1

### 2. Create and activate a virtual environment
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

---

## API Key Setup

### RapidAPI (Indeed Scraper)
Create an account at https://rapidapi.com and obtain a key for the Indeed Scraper API.

### OpenAI API
Create an account at https://platform.openai.com/api-keys and copy your key.

### Set environment variables

Windows PowerShell:
$env:RAPIDAPI_KEY="your_rapidapi_key_here"
$env:OPENAI_API_KEY="sk-your-openai-key"

macOS/Linux:
export RAPIDAPI_KEY="your_rapidapi_key_here"
export OPENAI_API_KEY="sk-your-openai-key"

---

## Run the Full Project

The main runs all stages in order.

python src/main.py --resume resume.pdf


