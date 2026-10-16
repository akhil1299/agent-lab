import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    raise ValueError("SEC_USER_AGENT not found - check your .env file")

MIN_GAP_SECONDS = 0.15
MAX_ATTEMPTS = 5
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

last_call_time = 0.0


def wait_for_turn():
    """Sleep just long enough that SEC calls are at least MIN_GAP_SECONDS apart."""
    global last_call_time
    now = time.monotonic()

    gap = now - last_call_time
    if gap < MIN_GAP_SECONDS:
        time.sleep((MIN_GAP_SECONDS - gap))

    last_call_time = time.monotonic()


def get_json(url):
    """Fetch a URL from the SEC politely and return the JSON as a Python dict."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            wait_seconds = 2 ** (attempt - 2)
            print(f"Retry {attempt - 1}: waiting {wait_seconds} s")
            time.sleep(wait_seconds)            

        wait_for_turn()
        try:
            response = SESSION.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as error:
            print(f"Attempt {attempt}: network error: {error}")
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRY_STATUS_CODES:
            print(f"Attempt {attempt}: SEC said {response.status_code}, will retry")
            continue

        raise RuntimeError(f"SEC said {response.status_code} for {url} - not retrying")

    raise RuntimeError(f"Gave up on {url} after {MAX_ATTEMPTS} attempts")


if __name__ == "__main__":
    start = time.monotonic()
    data = get_json("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")
    seconds = time.monotonic() - start

    print("Company:", data["entityName"])
    print("CIK:", data["cik"])
    print("us-gaap tags:", len(data["facts"]["us-gaap"]))
    print(f"Took {seconds:.2f} seconds")