from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:7394"


class DaemonClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self._base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        try:
            self._get("/health")
            return True
        except ConnectionError:
            return False

    def health(self) -> dict:
        return self._get("/health")

    def input(self, text: str) -> dict:
        return self._post("/input", {"text": text})

    def items(self, *, type: str | None = None, status: str | None = None) -> dict:
        params = {}
        if type:
            params["type"] = type
        if status:
            params["status"] = status
        return self._get("/items", params)

    def search(self, query: str) -> dict:
        return self._get("/search", {"q": query})

    def suggest(self, query: str, *, limit: int = 10) -> dict:
        return self._get("/suggest", {"q": query, "limit": str(limit)})

    def patch_item(self, item_id: str, **fields) -> dict:
        return self._patch(f"/items/{item_id}", fields)

    def delete_item(self, item_id: str) -> dict:
        return self._delete(f"/items/{item_id}")

    def settings(self) -> dict:
        return self._get("/settings")

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = self._base_url + path
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query_string}"
        return self._request(url)

    def _post(self, path: str, body: dict) -> dict:
        url = self._base_url + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(req)

    def _patch(self, path: str, body: dict) -> dict:
        url = self._base_url + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        return self._request(req)

    def _delete(self, path: str) -> dict:
        url = self._base_url + path
        req = urllib.request.Request(url, method="DELETE")
        return self._request(req)

    def _request(self, url_or_req) -> dict:
        try:
            with urllib.request.urlopen(url_or_req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(f"daemon not reachable: {exc}") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                raise ConnectionError(f"daemon error {exc.code}: {body}") from exc
