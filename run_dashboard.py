"""Serve the read-only college ICU replay dashboard on this computer only."""
from __future__ import annotations

import argparse
import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
import logging
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("dashboard")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class DashboardServer(ThreadingHTTPServer):
    """Serialize bounded patient/model caches while serving static assets freely."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, data, static_root=None):
        self.data = data
        self.data_lock = threading.RLock()
        self.static_root = Path(static_root or PROJECT_ROOT / "dashboard")
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ICUReplay/1.0"

    def log_message(self, message, *args):
        LOG.info("%s %s", self.client_address[0], message % args)

    def _send(self, status, payload, content_type="application/json; charset=utf-8", download_name=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if download_name is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Browser changed patient/model while a request was running.

    def _json(self, status, data):
        payload = json.dumps(data, allow_nan=False, ensure_ascii=False).encode("utf-8")
        self._send(status, payload)

    def _local_request(self):
        # Loopback binding + exact Host/Origin checks prevent other web pages and
        # DNS rebinding from turning this research demo into a patient-data API.
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            hosts.update({"127.0.0.1", "localhost"})
        if self.headers.get("Host", "").lower() not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in hosts}:
            return False
        return self.headers.get("Sec-Fetch-Site") not in {"cross-site"}

    @staticmethod
    def _one(query, key, default=None):
        values = query.get(key)
        if values is None:
            if default is not None:
                return default
            raise ValueError(f"Missing required parameter: {key}")
        if len(values) != 1 or not values[0]:
            raise ValueError(f"Provide one nonempty {key}")
        return values[0]

    def do_GET(self):
        if not self._local_request():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Open this demo directly from its localhost address."})
            return
        try:
            location = urlsplit(self.path)
            if location.path in STATIC_FILES:
                name, content_type = STATIC_FILES[location.path]
                self._send(HTTPStatus.OK, (self.server.static_root / name).read_bytes(), content_type)
                return
            if location.path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "mode": "simulated_icu_replay"})
                return
            known = {"/api/models", "/api/patients", "/api/patient", "/api/snapshot", "/api/review", "/api/shap.csv"}
            if location.path not in known:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})
                return
            query = parse_qs(location.query, keep_blank_values=True, max_num_fields=10)
            with self.server.data_lock:
                data = self.server.data
                if location.path == "/api/models":
                    result = {"models": data.model_catalog()}
                elif location.path == "/api/patients":
                    term = self._one(query, "query", "") if query.get("query") != [""] else ""
                    if len(term) > 100:
                        raise ValueError("Search must be 100 characters or fewer")
                    result = {"patients": data.catalog(query=term, limit=int(self._one(query, "limit", "50")))}
                elif location.path == "/api/patient":
                    result = data.patient_info(self._one(query, "patient_id"))
                elif location.path in {"/api/snapshot", "/api/shap.csv"}:
                    result = data.snapshot(self._one(query, "patient_id"), float(self._one(query, "hour")),
                                           model=self._one(query, "model", "baseline"))
                else:
                    result = data.retrospective(self._one(query, "patient_id"),
                                               model=self._one(query, "model", "baseline"))
            if location.path == "/api/shap.csv":
                stream = io.StringIO(newline="")
                writer = csv.writer(stream)
                writer.writerow(["patient_id", "ICULOS", "model", "rank", "feature", "feature_value", "shap_value", "units"])
                for rank, feature in enumerate(result["shap"]["features"], start=1):
                    writer.writerow([result["patient_id"], result["hour"], result["model"]["id"], rank,
                                     feature["feature"], feature["feature_value"], feature["shap_value"], "raw_log_odds"])
                # Plain UTF-8 CSV, complete local ranking. Excel accepts the BOM.
                safe_id = "".join(c for c in result["patient_id"] if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
                filename = f"{safe_id}_hour_{result['hour']:g}_{result['model']['id']}_all_SHAP.csv"
                self._send(HTTPStatus.OK, stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8",
                           download_name=filename)
            else:
                self._json(HTTPStatus.OK, result)
        except (ValueError, KeyError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except LookupError as error:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except FileNotFoundError:
            LOG.exception("Required local dashboard artifact is unavailable")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "A required saved model or raw patient file is unavailable. Check the terminal and DASHBOARD.md."})
        except Exception:
            LOG.exception("Dashboard request failed")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "This request could not be completed. Details are in the dashboard terminal."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.dashboard_data import DashboardData
    try:
        server = DashboardServer(("127.0.0.1", args.port), DashboardData())
    except Exception as error:
        LOG.error("Cannot start dashboard: %s", error)
        return 1
    LOG.info("Dashboard ready: http://127.0.0.1:%s", args.port)
    LOG.info("Simulated ICU replay using saved research models. No retraining. Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        LOG.info("Stopping dashboard")
    finally:
        server.server_close()
        with server.data_lock:
            server.data.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
