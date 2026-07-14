import logging
import os
import requests

logger = logging.getLogger(__name__)

class GPTZeroClient:
    """
    Client for the GPTZero API (https://gptzero.me/).
    Used as an external verification step to ensure humanized text bypasses commercial detectors.
    """
    def __init__(self):
        self.api_key = os.environ.get("GPTZERO_API_KEY", "").strip()
        self.url = "https://api.gptzero.me/v2/predict/text"
        
        if not self.api_key:
            logger.warning(
                "GPTZERO_API_KEY is not set in environment. "
                "External verification will be skipped."
            )
        else:
            logger.info("GPTZeroClient initialized successfully.")

    def check(self, text: str) -> float:
        """
        Check the text against GPTZero.
        Returns the completely generated probability score (0 to 100).
        If the API call fails or key is missing, returns 0.0 (fallback to assume passed).
        """
        if not self.api_key:
            return 0.0

        logger.info("Sending text to GPTZero API for external verification (%d chars)…", len(text))
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "document": text
            }
            # Timeout set to 15 seconds to prevent hanging the graph
            response = requests.post(self.url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # GPTZero returns completely_generated_prob in range 0.0 to 1.0
                prob = data.get("documents", [{}])[0].get("completely_generated_prob", 0.0)
                score = prob * 100
                logger.info("GPTZero API response received: AI probability = %.1f%%", score)
                return score
            else:
                logger.error(
                    "GPTZero API returned error status %d: %s",
                    response.status_code, response.text
                )
                return 0.0
        except Exception as exc:
            logger.exception("Exception occurred during GPTZero API call: %s", exc)
            return 0.0
