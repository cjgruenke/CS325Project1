# extract_indeed_jobs.py
"""
Read indeed_response.json and produce stage1_jobs.json and stage1_jobs.csv

Usage:
  1) put this file in the same folder as indeed_response.json
  2) python extract_indeed_jobs.py

Outputs:
  - stage1_jobs.json  (full flattened records)
  - stage1_jobs.csv   (summary CSV)
"""
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

INPUT = Path("indeed_response.json")
OUT_JSON = Path("stage1_jobs.json")
OUT_CSV = Path("stage1_jobs.csv")

def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_jobs(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Heuristics to find the list of job dicts inside the response.
    Checks common locations:
      - obj["returnvalue"]["data"]
      - obj["returnvalue"]
      - obj["data"]
      - obj["results"], obj["items"], obj["jobs"], obj["listings"]
      - top-level list
    """
    if obj is None:
        return None
    # common nested path used by your file: returnvalue.data
    if isinstance(obj, dict):
        if "returnvalue" in obj and isinstance(obj["returnvalue"], dict):
            rv = obj["returnvalue"]
            # sometimes the jobs are directly in rv["data"]
            if isinstance(rv.get("data"), list):
                return rv["data"]
            # or rv itself might be a list (rare)
            if isinstance(rv, list):
                return rv
        # check top-level keys
        for key in ("data", "results", "items", "jobs", "listings"):
            if isinstance(obj.get(key), list):
                return obj.get(key)
        # attempt to locate any nested list-of-dicts
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    return None

def safe_get(d: Dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def flatten(job: Dict[str, Any]) -> Dict[str, Any]:
    # location fields
    loc = job.get("location") or {}
    if isinstance(loc, dict):
        city = loc.get("city")
        state = loc.get("region") or loc.get("state") or None
        country = loc.get("country") or loc.get("countryCode")
        formatted = loc.get("formattedAddressLong") or loc.get("formattedAddressShort") or loc.get("fullAddress")
    else:
        city = state = country = formatted = None

    # salary handling
    salary = job.get("salary") or {}
    salary_text = salary.get("salaryText") or salary.get("text") or None
    salary_min = salary.get("salaryMin") or salary.get("salary_min") or None
    salary_max = salary.get("salaryMax") or salary.get("salary_max") or None

    attributes = job.get("attributes") or []
    if isinstance(attributes, list):
        attributes_str = "; ".join(str(a) for a in attributes)
    else:
        attributes_str = str(attributes)

    flattened = {
        "job_key": job.get("jobKey") or job.get("id") or job.get("job_id"),
        "title": job.get("title"),
        "companyName": job.get("companyName") or job.get("company") or job.get("company_name"),
        "city": city,
        "state": state,
        "country": country,
        "formattedAddress": formatted,
        "datePublished": job.get("datePublished"),
        "isRemote": job.get("isRemote", False),
        "jobUrl": job.get("jobUrl") or job.get("url") or job.get("link"),
        "applyUrl": job.get("applyUrl"),
        "salary_text": salary_text,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "descriptionText": job.get("descriptionText"),
        "descriptionHtml": job.get("descriptionHtml"),
        "attributes": attributes_str,
        # keep raw record for auditability if you want to inspect later
        "raw": job
    }
    return flattened

def truncate(s: Optional[str], n: int = 800) -> str:
    if not s:
        return ""
    s2 = s.replace("\r", " ").replace("\n", " ").strip()
    if len(s2) <= n:
        return s2
    return s2[:n] + " ... (truncated)"

def main():
    if not INPUT.exists():
        print(f"Input file not found: {INPUT.resolve()}")
        return

    obj = load_json(INPUT)
    jobs = find_jobs(obj)
    if not jobs:
        print("No job list found in JSON. Please paste a short snippet of the file here if you need help.")
        return

    flattened = [flatten(j) for j in jobs if isinstance(j, dict)]
    # write cleaned JSON
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(flattened, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON} ({len(flattened)} records)")

    # write CSV summary
    csv_cols = [
        "job_key", "title", "companyName", "city", "state", "country",
        "formattedAddress", "datePublished", "isRemote", "jobUrl", "applyUrl",
        "salary_text", "salary_min", "salary_max", "attributes", "descriptionText"
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in flattened:
            row = {k: r.get(k, "") for k in csv_cols}
            # truncate long descriptions to keep CSV readable
            row["descriptionText"] = truncate(row.get("descriptionText", ""), 600)
            writer.writerow(row)
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
