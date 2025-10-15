# indeed_full_scraper_bs4_fixed.py
"""
Indeed RapidAPI scraper (fixed for fromDays validation).

- Uses BeautifulSoup to clean descriptionHtml.
- Uses a safe single POST (no polling).
- Set RAPIDAPI_KEY environment variable before running.
- Save as indeed_full_scraper_bs4_fixed.py and run: python indeed_full_scraper_bs4_fixed.py

Requirements:
  pip install requests beautifulsoup4
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# -----------------------
# CONFIG
# -----------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
# If you must paste the key directly (not recommended), uncomment:
# RAPIDAPI_KEY = "your_key_here"

if not RAPIDAPI_KEY:
    raise RuntimeError("RAPIDAPI_KEY not found. Set the RAPIDAPI_KEY environment variable or edit the script.")

RAPIDAPI_HOST = "indeed-scraper-api.p.rapidapi.com"
ENDPOINT_POST = f"https://{RAPIDAPI_HOST}/api/job"

# Minimize API usage: single POST, small result set
DEFAULT_MAX_ROWS = 15

# IMPORTANT FIX: fromDays must be one of [1,3,7,14] — set to "7"
PAYLOAD = {
    "scraper": {
        "maxRows": DEFAULT_MAX_ROWS,
        "query": "Developer",
        "location": "Saint Louis MO",  # your requested location
        "jobType": "fulltime",
        "radius": "50",
        "sort": "relevance",
        "fromDays": "7",              # <-- fixed (allowed values: 1,3,7,14)
        "country": "us"
    }
}

OUTPUT_JSON = Path("stage1_jobs.json")
OUTPUT_CSV = Path("stage1_jobs.csv")
RAW_JSON = Path("indeed_response_raw.json")

# HTTP/retry settings (small to avoid extra calls)
TIMEOUT = 20
RETRIES = 1
BACKOFF_FACTOR = 0.5

# Polling disabled by default to save calls
POLL_IF_TASKID = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# -----------------------
# Helpers
# -----------------------
def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=RETRIES, backoff_factor=BACKOFF_FACTOR,
                  status_forcelist=[429, 500, 502, 503], allowed_methods=frozenset(["POST", "GET"]))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "Content-Type": "application/json",
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "User-Agent": "stage1-job-scraper/1.0 (student-project)"
    })
    return s

def save_raw(obj: Any):
    RAW_JSON.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved raw response to %s", RAW_JSON)

def html_to_text(html: Optional[str]) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def find_jobs(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        rv = obj.get("returnvalue")
        if isinstance(rv, dict) and isinstance(rv.get("data"), list):
            return rv.get("data")
        for key in ("data", "results", "items", "jobs", "listings"):
            v = obj.get(key)
            if isinstance(v, list):
                return v
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    return None

def flatten_job(job: Dict[str, Any]) -> Dict[str, Any]:
    loc = job.get("location") or {}
    if isinstance(loc, dict):
        city = loc.get("city") or loc.get("cityName")
        state = loc.get("region") or loc.get("state")
        formatted = loc.get("formattedAddressLong") or loc.get("formattedAddressShort") or loc.get("fullAddress")
    else:
        city = state = formatted = None

    description_html = job.get("descriptionHtml")
    if description_html:
        desc = html_to_text(description_html)
    else:
        desc = job.get("descriptionText") or job.get("description") or job.get("snippet") or ""

    salary = job.get("salary") or {}
    salary_text = salary.get("salaryText") or salary.get("text")
    salary_min = salary.get("salaryMin") or salary.get("salary_min")
    salary_max = salary.get("salaryMax") or salary.get("salary_max")

    attrs = job.get("attributes") or []
    attrs_str = "; ".join(str(a) for a in attrs) if isinstance(attrs, list) else str(attrs)

    return {
        "job_key": job.get("jobKey") or job.get("id") or job.get("job_id"),
        "title": job.get("title"),
        "companyName": job.get("companyName") or job.get("company") or job.get("company_name"),
        "city": city,
        "state": state,
        "formattedAddress": formatted,
        "datePublished": job.get("datePublished"),
        "jobUrl": job.get("jobUrl") or job.get("url") or job.get("link"),
        "salary_text": salary_text,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "descriptionText": desc,
        "attributes": attrs_str,
        "raw": job
    }

def write_outputs(flattened: List[Dict[str, Any]]):
    OUTPUT_JSON.write_text(json.dumps(flattened, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved flattened JSON to %s (%d records)", OUTPUT_JSON, len(flattened))

    csv_cols = [
        "job_key", "title", "companyName", "city", "state", "formattedAddress",
        "datePublished", "jobUrl", "salary_text", "salary_min", "salary_max", "attributes", "descriptionText"
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in flattened:
            row = {k: r.get(k, "") for k in csv_cols}
            desc = row.get("descriptionText") or ""
            if isinstance(desc, str) and len(desc) > 1200:
                row["descriptionText"] = desc[:1200] + " ... (truncated)"
            writer.writerow(row)
    logging.info("Saved CSV to %s", OUTPUT_CSV)


# -----------------------
# Main
# -----------------------
def main():
    session = build_session()
    logging.info("Posting search (location=%s, maxRows=%s)",
                 PAYLOAD["scraper"].get("location"), PAYLOAD["scraper"].get("maxRows"))
    try:
        r = session.post(ENDPOINT_POST, json=PAYLOAD, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.HTTPError as he:
        # show provider error body to help adapt payload
        body = ""
        try:
            body = r.text
        except Exception:
            body = "<could not read response text>"
        logging.error("HTTP error: %s. Response body: %s", he, body)
        # save whatever we got
        try:
            jr = r.json()
            save_raw(jr)
        except Exception:
            RAW_JSON.write_text(body, encoding="utf-8")
            logging.info("Saved raw non-JSON response to %s", RAW_JSON)
        return
    except Exception as e:
        logging.error("Request failed: %s", e)
        return

    result = r.json()
    save_raw(result)
    jobs = find_jobs(result)
    if not jobs:
        logging.error("No job list found in response. Inspect indeed_response_raw.json for details.")
        return

    flattened = [flatten_job(j) for j in jobs if isinstance(j, dict)]
    write_outputs(flattened)
    logging.info("Done. Processed %d records.", len(flattened))

if __name__ == "__main__":
    main()