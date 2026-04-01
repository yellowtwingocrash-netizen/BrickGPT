"""
BFL FLUX API Python Client - Production Ready

Features: rate limiting, retries with exponential backoff, batch processing.

Usage:
    client = BFLClient(os.environ["BFL_API_KEY"])
    result = client.generate("flux-2-pro", "A sunset over mountains")
    client.download(result.url, "sunset.png")
"""

import os
import time
import hmac
import hashlib
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BFLError(Exception):
    def __init__(self, message: str, status_code: int = None, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class AuthenticationError(BFLError):
    pass


class RateLimitError(BFLError):
    def __init__(self, message: str, retry_after: int = 5):
        super().__init__(message, 429, "rate_limit_exceeded")
        self.retry_after = retry_after


class ValidationError(BFLError):
    pass


class GenerationError(BFLError):
    pass


@dataclass
class GenerationResult:
    id: str
    url: str
    width: int
    height: int
    raw: Dict[str, Any]


class BFLClient:
    BASE_URLS = {
        "global": "https://api.bfl.ai",
        "eu": "https://api.eu.bfl.ai",
        "us": "https://api.us.bfl.ai",
    }

    def __init__(self, api_key: str, region: str = "global", max_concurrent: int = 24, timeout: int = 120):
        self.api_key = api_key
        self.base_url = self.BASE_URLS.get(region, self.BASE_URLS["global"])
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.headers = {"x-key": api_key, "Content-Type": "application/json"}

    def generate(self, model: str, prompt: str, width: int = 1024, height: int = 1024,
                 seed: int = None, timeout: int = None, **kwargs) -> GenerationResult:
        self._validate_dimensions(width, height)
        payload = {"prompt": prompt, "width": width, "height": height, **kwargs}
        if seed is not None:
            payload["seed"] = seed
        with self.semaphore:
            return self._submit_and_poll(model, payload, timeout or self.timeout)

    def generate_i2i(self, model: str, prompt: str, input_image: str,
                     additional_images: List[str] = None, **kwargs) -> GenerationResult:
        payload = {"prompt": prompt, "input_image": input_image, **kwargs}
        if additional_images:
            for i, img in enumerate(additional_images[:7], start=2):
                payload[f"input_image_{i}"] = img
        with self.semaphore:
            return self._submit_and_poll(model, payload, self.timeout)

    def generate_batch(self, model: str, prompts: List[str], max_workers: int = None, **kwargs) -> List[GenerationResult]:
        max_workers = max_workers or min(len(prompts), self.max_concurrent)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.generate, model, p, **kwargs): p for p in prompts}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    results.append(None)
        return results

    def download(self, url: str, output_path: str) -> str:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path

    def _submit_and_poll(self, model: str, payload: Dict, timeout: int) -> GenerationResult:
        endpoint = f"{self.base_url}/v1/{model}"
        response = self._request_with_retry("POST", endpoint, json=payload)
        polling_url = response["polling_url"]
        generation_id = response.get("id", polling_url.split("=")[-1])
        result = self._poll(polling_url, timeout)
        return GenerationResult(
            id=generation_id, url=result["sample"],
            width=result.get("width", payload.get("width")),
            height=result.get("height", payload.get("height")), raw=result,
        )

    def _poll(self, polling_url: str, timeout: int) -> Dict:
        start_time = time.time()
        delay = 1.0
        while time.time() - start_time < timeout:
            response = self._request_with_retry("GET", polling_url)
            status = response.get("status")
            if status == "Ready":
                return response.get("result", response)
            elif status == "Error":
                raise GenerationError(response.get("error", "Generation failed"))
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)
        raise TimeoutError(f"Generation timed out after {timeout}s")

    def _request_with_retry(self, method: str, url: str, max_retries: int = 3, **kwargs) -> Dict:
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
                return self._handle_response(response)
            except RateLimitError as e:
                time.sleep(e.retry_after * (attempt + 1))
                last_exception = e
            except BFLError as e:
                if e.status_code and e.status_code >= 500:
                    time.sleep(2 ** attempt)
                    last_exception = e
                else:
                    raise
        raise last_exception

    def _handle_response(self, response: requests.Response) -> Dict:
        if response.status_code == 200:
            return response.json()
        try:
            error_data = response.json()
        except Exception:
            error_data = {"message": response.text}
        message = error_data.get("message", "Unknown error")
        if response.status_code == 401:
            raise AuthenticationError(message, 401)
        elif response.status_code == 429:
            raise RateLimitError(message, int(response.headers.get("Retry-After", 5)))
        elif response.status_code == 400:
            raise ValidationError(message, 400)
        else:
            raise BFLError(message, response.status_code)

    def _validate_dimensions(self, width: int, height: int):
        if width % 16 != 0:
            raise ValidationError(f"Width {width} must be a multiple of 16")
        if height % 16 != 0:
            raise ValidationError(f"Height {height} must be a multiple of 16")
        if width * height > 4_000_000:
            raise ValidationError(f"Total pixels ({width}x{height}) exceeds 4MP limit")


if __name__ == "__main__":
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        print("Set BFL_API_KEY environment variable")
        exit(1)
    client = BFLClient(api_key)
    print("Generating image...")
    result = client.generate("flux-2-pro", "A serene mountain landscape at golden hour", width=1024, height=1024)
    print(f"Generated: {result.url}")
    client.download(result.url, "output.png")
    print("Saved to output.png")
