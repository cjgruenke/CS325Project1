# indeed_full_scraper_try_alts.py
"""
Attempt the Indeed RapidAPI POST, show detailed error text on 400,
and automatically try a few safe alternate payloads (limited attempts).
Minimizes API calls: max 3 POSTs by default.

Requirements:
  pip install requests beautifulsoup4

Set RAPIDAPI_KEY in env or edit the script (env recommended).
"""
import os
import time
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
if not RAPIDAPI_KEY:
    raise RuntimeError("Set RAPIDAPI_KEY environment variable before running.")

RAPIDAPI_HOST = "indeed-scraper-api.p.rapidapi.com"
ENDPOINT_POST = f"https://{RAPIDAPI_HOST}/api/job"
ENDPOINT_RESULT = f"https://{RAPIDAPI_HOST}/api/job/result"

# Output files
RAW_JSON = Path("indeed_response_raw.json")
OUTPUT_JSON = Path("stage1_jobs.json")
OUTPUT_CSV = Path("stage1_jobs.csv")

# Low default to save calls
DEFAULT_MAX_ROWS = 15
MAX_TOTAL_ATTEMPTS = 3  # will do at most this many POST attempts (including retries)
POLL_IF_TASKID = False  # keep polling off to avoid multiple calls

# HTTP settings
TIMEOUT = 20
RETRIES = 1  # retries at HTTP layer are small to avoid extra calls
BACKOFF_FACTOR = 0.5

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------
# Payload variants (safe, limited)
# -----------------------
# Base payload (preferred)
BASE_PAYLOAD = {
    "scraper": {
        "maxRows": DEFAULT_MAX_ROWS,
        "query": "Developer",
        "location": "Saint Louis MO",
        "jobType": "fulltime",
        "radius": "50",
        "sort": "relevance",
        "fromDays": "30",
        "country": "us"
    }
}

# alternate payloads to try if the first returns 400
ALTERNATE_PAYLOADS = [
    # variant: add comma and state abbreviation
    {
        "scraper": {
            **BASE_PAYLOAD["scraper"],
            "location": "Saint Louis, MO"
        }
    },
    # variant: abbreviated city
    {
        "scraper": {
            **BASE_PAYLOAD["scraper"],
            "location": "St. Louis, MO"
        }
    },
    # minimal payload (remove optional fields that might be restricted)
    {
        "scraper": {
            "maxRows": 10,
            "query": "Developer",
            "location": "Saint Louis MO",
            "country": "us"
        }
    }
]

# combine into ordered list, but keep attempts limited
PAYLOAD_ATTEMPTS = [BASE_PAYLOAD] + [p for p in ALTERNATE_PAYLOADS]

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

def find_jobs(obj: Any):
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

    # description: prefer HTML cleaned
    description_html = job.get("descriptionHtml")
    if description_html:
        desc = html_to_text(description_html)
    else:
        desc = job.get("descriptionText") or job.get("description") or job.get("snippet") or ""

    salary = job.get("salary") or {}
    salary_text = salary.get("salaryText") or salary.get("text")
    salary_min = salary.get("salaryMin") or salary.get("salary_min")
    salary_max = salary.get("salaryMax") or salary.get("salary_max")

    return {
        "job_key": job.get("jobKey") or job.get("id"),
        "title": job.get("title"),
        "companyName": job.get("companyName") or job.get("company"),
        "city": city,
        "state": state,
        "formattedAddress": formatted,
        "datePublished": job.get("datePublished"),
        "jobUrl": job.get("jobUrl") or job.get("url"),
        "salary_text": salary_text,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "descriptionText": desc,
        "raw": job
    }

def write_outputs(flattened: List[Dict[str, Any]]):
    OUTPUT_JSON.write_text(json.dumps(flattened, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved flattened JSON to %s (%d records)", OUTPUT_JSON, len(flattened))
    csv_cols = ["job_key", "title", "companyName", "city", "state", "formattedAddress",
                "datePublished", "jobUrl", "salary_text", "salary_min", "salary_max", "descriptionText"]
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
# Main attempt loop
# -----------------------
def attempt_post_and_extract(session: requests.Session, payload: Dict[str, Any]):
    logging.info("Attempting POST with location=%s, maxRows=%s",
                 payload.get("scraper", {}).get("location"),
                 payload.get("scraper", {}).get("maxRows"))
    try:
        r = session.post(ENDPOINT_POST, json=payload, timeout=TIMEOUT)
        # if status_code >=400 this will raise and we will inspect r.text
        r.raise_for_status()
    except requests.HTTPError as he:
        # show the response body for debugging (often contains validation message)
        text = ""
        try:
            text = r.text
        except Exception:
            text = "<could not read response text>"
        logging.error("HTTP error (%s). Response body:\n%s", he, text)
        # save raw response body if any
        try:
            parsed = r.json()
            save_raw(parsed)
        except Exception:
            RAW_JSON.write_text(text, encoding="utf-8")
            logging.info("Saved raw non-JSON response to %s", RAW_JSON)
        raise
    except Exception as e:
        logging.error("Request failed: %s", e)
        raise

    # success path
    result = r.json()
    save_raw(result)
    jobs = find_jobs(result)
    return jobs or []

def main():
    session = build_session()
    attempt_count = 0
    final_jobs = []
    for payload in PAYLOAD_ATTEMPTS:
        if attempt_count >= MAX_TOTAL_ATTEMPTS:
            logging.info("Reached max allowed attempts (%d). Stopping.", MAX_TOTAL_ATTEMPTS)
            break
        attempt_count += 1
        try:
            jobs = attempt_post_and_extract(session, payload)
        except Exception:
            logging.warning("Attempt %d failed. Checking next payload variant (if any).", attempt_count)
            # small backoff to be polite
            time.sleep(1.0)
            continue
        if jobs:
            logging.info("Success: found %d job records on attempt %d.", len(jobs), attempt_count)
            final_jobs = jobs
            break
        else:
            logging.warning("No job list found in response for attempt %d. Trying next payload variant if available.", attempt_count)
            time.sleep(0.8)

    if not final_jobs:
        logging.error("All attempts completed and no job list was returned. Inspect indeed_response_raw.json for details.")
        return

    flattened = [flatten_job(j) for j in final_jobs if isinstance(j, dict)]
    write_outputs(flattened)
    logging.info("Done. Processed %d records.", len(flattened))

if __name__ == "__main__":
    main()
