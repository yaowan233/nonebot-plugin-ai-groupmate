"""Standalone authenticated gateway; run this file on a Docker-capable host.

Only execution_backend.py and httpx are needed alongside this file. No Bot or
provider credentials are loaded. Each request owns a fresh offline container.
"""

import os
import hmac
import json
import asyncio
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

if __package__:
    from .execution_backend import ExecutionError, run_docker
else:
    # Direct script invocation puts this directory on sys.path.
    from execution_backend import ExecutionError, run_docker  # pyright: ignore[reportMissingImports]


def make_handler(token: str, image: str, timeout: float):
    gate = threading.BoundedSemaphore(2)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never log credentials, code, or file contents.

        def respond(self, status: int, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self.connection.settimeout(30)
            if self.path != "/execute" or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.respond(403, {"error": "Forbidden"})
                return
            if not gate.acquire(blocking=False):
                self.respond(429, {"error": "Executor busy"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    self.respond(413, {"error": "Request too large"})
                    return
                request = json.loads(self.rfile.read(size))
                result = asyncio.run(run_docker(request, image=image, timeout=timeout))
                self.respond(200, result)
            except ExecutionError as error:
                self.respond(200, {"error": str(error)})
            except (ValueError, KeyError, TypeError):
                self.respond(400, {"error": "Invalid execution request"})
            except Exception:
                self.respond(500, {"error": "Execution service failed"})
            finally:
                gate.release()

    return Handler


if __name__ == "__main__":
    secret = os.environ.get("EXECUTION_SERVICE_TOKEN", "")
    if len(secret) < 16:
        raise SystemExit("Set EXECUTION_SERVICE_TOKEN to a random secret of at least 16 characters.")
    server = ThreadingHTTPServer((os.environ.get("EXECUTION_SERVICE_HOST", "127.0.0.1"), int(os.environ.get("EXECUTION_SERVICE_PORT", "8091"))),
                                make_handler(secret, os.environ.get("EXECUTION_SERVICE_IMAGE", "ai-groupmate-executor:local"), float(os.environ.get("EXECUTION_SERVICE_TIMEOUT", "20"))))
    server.serve_forever()
