from __future__ import annotations

import ipaddress
import html as html_lib
import re
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote, urlparse

import requests
import trafilatura


_URL_RE = re.compile(r"(https?://[^\s<>()]+)", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    return [m.group(1).rstrip(").,]}>\"'") for m in _URL_RE.finditer(text or "")]


def _is_private_host(hostname: str) -> bool:
    """
    Block localhost and private network targets to reduce SSRF risk.
    """
    host = (hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        # Resolve hostname to IPs and block private ranges
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return True
    except Exception:
        # If resolution fails, treat as non-private; request may still fail later.
        return False

    return False


def validate_url(url: str) -> Optional[str]:
    """
    Return an error string if URL is unsafe/unsupported; otherwise None.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"

    if parsed.scheme not in {"http", "https"}:
        return "Only http/https URLs are supported"

    if not parsed.netloc:
        return "URL is missing a hostname"

    if _is_private_host(parsed.hostname or ""):
        return "Blocked URL host (private/localhost)"

    return None


def _try_extract_x_oembed(url: str, *, timeout_s: int) -> LinkExtract | None:
    """
    Server-side extraction for X/Twitter status URLs using oEmbed.

    X frequently blocks HTML extraction (JS-only / anti-bot). oEmbed often still works
    and returns readable tweet HTML that we can strip to text.
    """
    try:
        oembed_url = (
            "https://publish.twitter.com/oembed"
            f"?url={quote(url, safe='')}&omit_script=true"
        )
        headers = {
            "User-Agent": "cortexa/0.1 (x-oembed; +https://example.local)",
            "Accept": "application/json,text/plain,*/*",
        }
        resp = requests.get(oembed_url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()

        data = resp.json()
        html_block = str(data.get("html") or "").strip()
        if not html_block:
            return None

        author = str(data.get("author_name") or "").strip()
        title = f"Tweet by {author}" if author else "Tweet"

        # Strip tags from the oEmbed HTML to get readable text.
        text = re.sub(r"<[^>]+>", " ", html_block)
        text = html_lib.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()

        # Basic sanity check: avoid embedding garbage.
        if len(text) < 40:
            return None

        text = _sanitize_extracted(text)
        if not text.strip():
            return None

        return LinkExtract(url=url, title=title, text=text)
    except Exception:
        return None


@dataclass
class LinkExtract:
    url: str
    title: str
    text: str


# Patterns that suggest prompt-injection attempts in ingested web content.
_INJECTION_LINE_RE = re.compile(
    r"^(system:|user:|assistant:|ignore previous|forget previous|###\s|<\|im_start\||<\|endoftext\|)",
    re.IGNORECASE,
)
_MAX_PARAGRAPH_CHARS = 2000


def _sanitize_extracted(text: str) -> str:
    """
    Remove lines that look like prompt-injection attempts and cap long paragraphs.
    This runs on text already extracted by trafilatura (plain text, no HTML).
    """
    clean_lines: list[str] = []
    for line in text.splitlines():
        if _INJECTION_LINE_RE.match(line.strip()):
            continue  # drop suspicious line
        # Cap any single line that is unreasonably long
        if len(line) > _MAX_PARAGRAPH_CHARS:
            line = line[:_MAX_PARAGRAPH_CHARS] + "…"
        clean_lines.append(line)
    return "\n".join(clean_lines)


def fetch_and_extract(url: str, timeout_s: int = 15, max_bytes: int = 2_000_000) -> LinkExtract:
    """
    Fetch a URL and extract readable main text.
    Safety:
    - caps download size
    - uses timeouts
    - does NOT execute JS (requests only)
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"x.com", "twitter.com"} or host.endswith(".x.com") or host.endswith(".twitter.com"):
        x = _try_extract_x_oembed(url, timeout_s=timeout_s)
        if x and x.text:
            return x

    headers = {
        "User-Agent": "cortexa/0.1 (link-ingestion; +https://example.local)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    with requests.get(url, headers=headers, timeout=timeout_s, stream=True) as resp:
        resp.raise_for_status()

        content_type = (resp.headers.get("content-type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            # Still try to parse if it looks like text, but avoid huge binaries
            if "text/" not in content_type:
                raise ValueError(f"Unsupported content-type: {content_type or 'unknown'}")

        chunks: list[bytes] = []
        total = 0
        for part in resp.iter_content(chunk_size=64 * 1024):
            if not part:
                continue
            total += len(part)
            if total > max_bytes:
                raise ValueError("Page too large to ingest safely")
            chunks.append(part)

    html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")

    downloaded = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = trafilatura.extract_metadata(html).title if trafilatura.extract_metadata(html) else ""

    text = _sanitize_extracted(downloaded.strip())
    if not text:
        raise ValueError("Could not extract readable text from page")

    return LinkExtract(url=url, title=(title or "").strip(), text=text)

