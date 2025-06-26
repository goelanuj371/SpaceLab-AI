# utils/nasa_api.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")

BASE_URL_TECHTRANSFER = "https://api.nasa.gov/techtransfer"

# ✅ 1. Fetch TechTransfer Data (live working API)
def fetch_techtransfer_data(query="engine"):
    url = f"{BASE_URL_TECHTRANSFER}/patent/?{query}&api_key={NASA_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"TechTransfer API Error: {response.status_code} - {response.text}")

