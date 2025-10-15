import requests
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Your RapidAPI credentials
url = "https://indeed-scraper-api.p.rapidapi.com/api/job"
headers = {
    "Content-Type": "application/json",
    "x-rapidapi-host": "indeed-scraper-api.p.rapidapi.com",
    "x-rapidapi-key": "43848446eemsh310e3c79dbd911cp1c537cjsn5cd5e8f3ae05"
}

# Request body (based on your screenshot)
payload = {
    "scraper": {
        "maxRows": 15,
        "query": "Developer",
        "location": "San Francisco",
        "jobType": "fulltime",
        "radius": "50",
        "sort": "relevance",
        "fromDays": "7",
        "country": "us"
    }
}

# Output file
output_path = Path("indeed_response.json")

def main():
    logging.info("Sending POST request to Indeed Scraper API...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return

    # Save the raw JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response.json(), f, indent=2)
    logging.info(f"Saved JSON response to {output_path}")

    # Try to extract jobs from known keys
    data = response.json()
    job_list = None
    for key in ("jobs", "results", "data", "items", "listings"):
        if isinstance(data.get(key), list):
            job_list = data[key]
            break

    if not job_list:
        logging.warning("No job list found in response. Check indeed_response.json for details.")
        return

    # Save simplified job data (if present)
    simplified = []
    for job in job_list:
        simplified.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "description": job.get("description"),
            "url": job.get("url")
        })

    Path("indeed_jobs.json").write_text(json.dumps(simplified, indent=2))
    logging.info(f"Extracted {len(simplified)} jobs → saved to indeed_jobs.json")

if __name__ == "__main__":
    main()
