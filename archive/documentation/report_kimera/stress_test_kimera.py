import requests
import time
from statistics import mean

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


API_BASE = "http://localhost:8001"  # Adjust port if needed
ENDPOINTS = ["/system/status", "/system/stability"]
ITERATIONS = 100

results = {ep: [] for ep in ENDPOINTS}
errors = {ep: 0 for ep in ENDPOINTS}

logger.info(f"Starting stress test on endpoints: {ENDPOINTS}")

for i in range(ITERATIONS):
    for ep in ENDPOINTS:
        url = API_BASE + ep
        start = time.time()
        try:
            resp = requests.get(url, timeout=5)
            elapsed = time.time() - start
            results[ep].append(elapsed)
            if resp.status_code != 200:
                errors[ep] += 1
                logger.error(f"[ERROR] {url} returned status {resp.status_code}")
        except Exception as e:
            errors[ep] += 1
            logger.error(f"[EXCEPTION] {url}: {e}")

logger.info("\n--- Stress Test Summary ---")
for ep in ENDPOINTS:
    times = results[ep]
    logger.info(f"Endpoint: {ep}")
    logger.info(f"  Requests: {len(times)
    logger.error(f"  Errors: {errors[ep]}")
    if times:
        logger.info(f"  Avg Response Time: {mean(times)
        logger.info(f"  Min Response Time: {min(times)
        logger.info(f"  Max Response Time: {max(times)
    logger.info()

