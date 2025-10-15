#!/usr/bin/env python3
"""
Stage 2: Data Preprocessing (no NLTK version)
Cleans job posting data and parses resume text for embedding.
Outputs:
 - cleaned_jobs.json
 - cleaned_jobs.csv
 - processed_resume.txt
 - resume_sections.json
"""

import re
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from bs4 import BeautifulSoup
import PyPDF2
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
INPUT_JSON = Path("stage1_jobs.json")
INPUT_CSV = Path("stage1_jobs.csv")

OUTPUT_JSON = Path("cleaned_jobs.json")
OUTPUT_CSV = Path("cleaned_jobs.csv")

RESUME_PATH = Path("Cole_Gruenke_Resume.pdf")

LOWERCASE = True
REMOVE_SPECIAL_CHARS = True
MISSING_POLICY = "fill"  # "fill" or "drop"

LOCATION_NORMALIZATION = {
    "st louis": "saint louis, mo",
    "st. louis": "saint louis, mo",
    "saint louis": "saint louis, mo",
    "st louis mo": "saint louis, mo",
    "st. louis, mo": "saint louis, mo",
    "saint louis, missouri": "saint louis, mo",
    "st louis, missouri": "saint louis, mo"
}

# -----------------------
# Helpers
# -----------------------
def clean_html_to_text(html: Optional[str]) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def remove_special_characters(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z\.\,\;\:\-\(\)\s%/@#&+']", " ", s)

def normalize_location(loc: Optional[str]) -> Optional[str]:
    if not loc:
        return loc
    s = loc.lower().strip()
    s = re.sub(r"[^\w\s,\.]", "", s)
    s = s.replace("missouri", "mo").replace("county", "")
    for k, v in LOCATION_NORMALIZATION.items():
        if k in s:
            return v
    return s

def load_stage1_data() -> List[Dict[str, Any]]:
    if INPUT_JSON.exists():
        print(f"Loading {INPUT_JSON}")
        return json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    elif INPUT_CSV.exists():
        print(f"Loading {INPUT_CSV}")
        return pd.read_csv(INPUT_CSV).to_dict(orient="records")
    else:
        raise FileNotFoundError("No stage1_jobs.json or stage1_jobs.csv found. Run Stage 1 first.")

def preprocess_job_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rec = dict(raw)
    title = rec.get("title") or rec.get("jobTitle") or ""
    company = rec.get("companyName") or rec.get("company") or ""
    formatted = rec.get("formattedAddress") or rec.get("formattedAddressLong") or ""
    if not formatted:
        city = rec.get("city") or rec.get("location") or ""
        state = rec.get("state") or ""
        formatted = f"{city} {state}".strip()
    normalized_loc = normalize_location(formatted)
    desc = rec.get("descriptionText") or rec.get("description") or ""
    if not desc:
        html = rec.get("descriptionHtml") or ""
        desc = clean_html_to_text(html)
    if LOWERCASE:
        title, company, desc = title.lower(), company.lower(), desc.lower()
    if REMOVE_SPECIAL_CHARS:
        title, company, desc = remove_special_characters(title), remove_special_characters(company), remove_special_characters(desc)
    title, company, desc = normalize_whitespace(title), normalize_whitespace(company), normalize_whitespace(desc)

    if MISSING_POLICY == "drop" and (not title or not desc):
        return None
    if MISSING_POLICY == "fill":
        title = title or "n/a"
        company = company or "n/a"
        normalized_loc = normalized_loc or "n/a"
        desc = desc or "n/a"

    return {
        "job_key": rec.get("job_key") or rec.get("jobKey") or rec.get("job_id"),
        "title": title,
        "companyName": company,
        "location_raw": formatted,
        "location_normalized": normalized_loc,
        "datePublished": rec.get("datePublished"),
        "jobUrl": rec.get("jobUrl") or rec.get("url"),
        "salary_text": rec.get("salary_text"),
        "description": desc
    }

# Resume helpers
def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text_parts.append(p.extract_text() or "")
    return "\n".join(text_parts)

def read_resume(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError("Unsupported resume format (.pdf or .txt only)")

def split_resume_sections(text: str) -> Dict[str, str]:
    headings = ["experience", "education", "skills", "projects", "summary", "certifications"]
    pattern = r"(?im)^\s*(?P<h>" + "|".join(headings) + r")\s*[:\-]?\s*$"
    parts = re.split(pattern, text)
    sections = defaultdict(str)
    if len(parts) <= 1:
        sections["full"] = text
        return sections
    if parts[0].strip():
        sections["summary"] = parts[0].strip()
    i = 1
    while i < len(parts)-1:
        heading = parts[i].strip().lower()
        body = parts[i+1].strip()
        sections[heading] = body
        i += 2
    sections["full"] = text
    return sections

def clean_resume_text(text: str) -> str:
    text = clean_html_to_text(text)
    if LOWERCASE:
        text = text.lower()
    if REMOVE_SPECIAL_CHARS:
        text = remove_special_characters(text)
    return normalize_whitespace(text)

# -----------------------
# Main
# -----------------------
def main(resume_arg: Optional[str] = None):
    try:
        data = load_stage1_data()
    except Exception as e:
        print("Error loading Stage 1 data:", e)
        return

    cleaned = []
    for rec in data:
        c = preprocess_job_record(rec)
        if c:
            cleaned.append(c)
    OUTPUT_JSON.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved cleaned job JSON → {OUTPUT_JSON} ({len(cleaned)} records)")

    csv_fields = ["job_key", "title", "companyName", "location_normalized", "datePublished", "jobUrl", "salary_text", "description"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in cleaned:
            d = r.copy()
            if len(d["description"]) > 1000:
                d["description"] = d["description"][:1000] + " ... (truncated)"
            writer.writerow(d)
    print(f"Saved cleaned job CSV → {OUTPUT_CSV}")

    # Resume
    resume_path = Path(resume_arg) if resume_arg else RESUME_PATH
    if resume_path.exists():
        print(f"Parsing resume from {resume_path.name}")
        raw_resume = read_resume(resume_path)
        sections = split_resume_sections(raw_resume)
        cleaned_resume = clean_resume_text(sections.get("full", raw_resume))
        Path("processed_resume.txt").write_text(cleaned_resume, encoding="utf-8")
        Path("resume_sections.json").write_text(json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote processed_resume.txt and resume_sections.json")
    else:
        print("No resume found; skipping resume parsing.")

if __name__ == "__main__":
    resume_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(resume_arg)