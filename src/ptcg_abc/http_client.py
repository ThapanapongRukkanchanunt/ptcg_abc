from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ptcg_abc.normalize import slugify


USER_AGENT = "ptcg-abc/0.1 (+https://github.com/ThapanapongRukkanchanunt/ptcg_abc)"


class HttpRequestError(RuntimeError):
    pass


def basic_auth_header(username: str, key: str) -> str:
    token = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def request_bytes(
    url: str,
    *,
    auth_header: str | None = None,
    timeout: int = 60,
) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if auth_header:
        headers["Authorization"] = auth_header
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpRequestError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except URLError as exc:
        raise HttpRequestError(f"Could not reach {url}: {exc}") from exc


def request_json(
    url: str,
    *,
    auth_header: str | None = None,
    timeout: int = 60,
) -> object:
    payload = request_bytes(url, auth_header=auth_header, timeout=timeout)
    return json.loads(payload.decode("utf-8"))


def cache_name_for_url(url: str) -> str:
    parsed = urlparse(url)
    stem = slugify(f"{parsed.netloc}-{parsed.path}-{parsed.query}", max_length=140)
    return f"{stem}.html"


def fetch_text_with_cache(
    url: str,
    cache_dir: Path,
    *,
    refresh: bool = False,
    timeout: int = 60,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name_for_url(url)
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    text = request_bytes(url, timeout=timeout).decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text
