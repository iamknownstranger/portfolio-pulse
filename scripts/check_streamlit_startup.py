"""Probe a real Streamlit server, including the gzip-enabled static-file path.

Import checks and AppTest do not exercise HTTP middleware. In particular,
Streamlit 1.58 with Starlette 1.x can report healthy while browser assets fail.
Run with the same interpreter and dependencies used for deployment.
"""

import gzip
from html.parser import HTMLParser
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent.parent


class ScriptSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        source = dict(attrs).get("src")
        if tag == "script" and source:
            self.sources.append(source)


def check_startup():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/"
    # Only these loopback probes bypass proxies. The app process retains its
    # inherited proxy and trust configuration for external data providers.
    opener = build_opener(ProxyHandler({}))

    def fetch(path, content_type):
        url = urljoin(base_url, path)
        if urlparse(url).netloc != urlparse(base_url).netloc:
            raise RuntimeError("Expected an asset served by the local Streamlit server.")
        request = Request(url, headers={"Accept-Encoding": "gzip"})
        with opener.open(request, timeout=5) as response:
            if response.status != 200 or content_type not in response.headers.get("Content-Type", ""):
                raise RuntimeError(f"Unexpected response for {path}: {response.status}, {response.headers.get('Content-Type')}")
            body = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            if not body:
                raise RuntimeError(f"Empty response for {path}.")
            return body

    with tempfile.TemporaryFile(mode="w+t") as logs:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "1_💼_Portfolio_Pulse.py",
             "--server.address", "127.0.0.1", "--server.port", str(port),
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=logs, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit exited with code {process.returncode}.")
                try:
                    if fetch("_stcore/health", "text/plain").strip() != b"ok":
                        raise RuntimeError("Unexpected Streamlit health response.")
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Streamlit did not become healthy within 30 seconds.")
                    time.sleep(0.2)
            parser = ScriptSources()
            parser.feed(fetch("", "text/html").decode("utf-8"))
            if not parser.sources:
                raise RuntimeError("The app HTML did not include its JavaScript entrypoint.")
            for source in parser.sources:
                fetch(source, "javascript")
            icon = fetch("favicon.png", "image/png")
            if not icon.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("The favicon response was not a PNG.")
            print("Streamlit HTTP smoke check passed: health, HTML, JavaScript and favicon (gzip enabled).")
        except Exception:
            logs.seek(0)
            print(logs.read()[-12000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    check_startup()
