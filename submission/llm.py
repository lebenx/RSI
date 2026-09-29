"""Small replayable API adapter: one JSON file per explicit request identity."""
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import requests


class JsonLLM:
    def __init__(self, directory, model="deepseek-chat", api_key=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.cache_hits = 0
        self.network_calls = 0
        self.network_retries = 0

    _TRANSIENT_ERRORS = {"SSLError", "ConnectionError", "ConnectTimeout",
                         "ReadTimeout", "Timeout", "ChunkedEncodingError",
                         "ProxyError", "RemoteDisconnected", "ConnectionResetError"}

    def ask(self, request_id, system, payload, max_tokens=1800):
        if any(x in request_id for x in ("/", "..")):
            raise ValueError("request_id must be a simple filename")
        path = self.directory / (request_id + ".json")
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
        if path.exists():
            entry = json.loads(path.read_text())
            if entry["request"] != body:
                raise ValueError(f"request identity reused with different input: {request_id}")
            if entry.get("ok"):
                self.cache_hits += 1
                return entry["parsed"], entry
            # Transport failures are safe to retry because the request body is
            # immutable and no parsed model response was archived. Semantic
            # failures (HTTP status, malformed JSON, truncation) stay visible.
            if entry.get("error_type") not in self._TRANSIENT_ERRORS:
                raise RuntimeError(f"archived request failed: {request_id}")
            path.unlink()
        if not self.key:
            raise RuntimeError("DEEPSEEK_API_KEY required for uncached request")
        started = time.monotonic()
        entry = {"request": body, "request_id": request_id,
                 "utc": datetime.now(timezone.utc).isoformat(), "ok": False,
                 "attempts": 0}
        for attempt in range(3):
            entry["attempts"] = attempt + 1
            try:
                response = requests.post("https://api.deepseek.com/chat/completions", json=body,
                                         headers={"Authorization": "Bearer " + self.key},
                                         timeout=(15, 90))
                self.network_calls += 1
                entry["http_status"] = response.status_code
                response.raise_for_status()
                entry["response"] = response.json()
                choice = entry["response"]["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("non-stop completion (possibly truncated)")
                entry["parsed"] = json.loads(choice["message"]["content"])
                if not isinstance(entry["parsed"], dict):
                    raise ValueError("expected JSON object")
                entry["ok"] = True
                break
            except Exception as exc:
                error_type = type(exc).__name__
                entry["error_type"] = error_type
                # No request headers or potentially secret-bearing exception text.
                if error_type not in self._TRANSIENT_ERRORS or attempt == 2:
                    break
                self.network_retries += 1
                time.sleep(1.0 * (attempt + 1))
        entry["elapsed_seconds"] = time.monotonic() - started
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2))
        if not entry["ok"]:
            raise RuntimeError(f"API failure {request_id}: {entry.get('error_type')}")
        return entry["parsed"], entry


def probability(x):
    if isinstance(x, bool):
        raise ValueError("boolean probability")
    x = float(x)
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("invalid probability")
    return x


def tokens(entry):
    return int(entry.get("response", {}).get("usage", {}).get("total_tokens", 0))
