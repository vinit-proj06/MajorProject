"""HTTP boundary tests; no model training or access to private patient files."""
from contextlib import contextmanager
import http.client
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from run_dashboard import DashboardServer


class FakeData:
    def __init__(self):
        self.calls = []

    def catalog(self, query="", limit=50):
        self.calls.append(("catalog", query, limit))
        return [{"patient_id": "p006266", "dataset": "A"}]

    def model_catalog(self):
        return [{"id": "baseline"}]

    def patient_info(self, patient_id):
        if patient_id != "p006266":
            raise LookupError("Patient is not in the test set")
        return {"patient_id": patient_id, "first_hour": 4, "last_hour": 34}

    def snapshot(self, patient_id, hour, model="baseline"):
        self.calls.append(("snapshot", patient_id, hour, model))
        return {"patient_id": patient_id, "hour": hour, "model_score": .5, "model": {"id": model},
                "shap": {"features": [{"feature": "HR", "feature_value": 80, "shap_value": .2},
                                      {"feature": "Temp", "feature_value": 37, "shap_value": -.1}]}}

    def retrospective(self, patient_id, model="baseline"):
        self.calls.append(("review", patient_id, model))
        return {"mode": "retrospective_review", "timeline": [{"hour": 25, "label": 1}]}


@contextmanager
def running_server():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "index.html").write_text("<h1>Replay</h1>", encoding="utf-8")
        data = FakeData()
        server = DashboardServer(("127.0.0.1", 0), data, static_root=root)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        try:
            yield server, data
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def request(server, path, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


class DashboardServerTests(unittest.TestCase):
    def test_complete_csv_download_uses_snapshot_only(self):
        with running_server() as (server, data):
            status, headers, body = request(server, "/api/shap.csv?patient_id=p006266&hour=22&model=C")
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "text/csv; charset=utf-8")
            self.assertIn("attachment", headers["Content-Disposition"])
            rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
            self.assertEqual([row["feature"] for row in rows], ["HR", "Temp"])
            self.assertEqual(rows[1]["shap_value"], "-0.1")
            self.assertTrue(all(row["model"] == "C" for row in rows))
            self.assertEqual(data.calls, [("snapshot", "p006266", 22.0, "C")])

    def test_explicit_outcome_route_and_no_implicit_review(self):
        with running_server() as (server, data):
            status, headers, body = request(server, "/api/snapshot?patient_id=p006266&hour=22&model=C")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["hour"], 22)
            self.assertEqual(data.calls, [("snapshot", "p006266", 22.0, "C")])
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertNotIn("Access-Control-Allow-Origin", headers)
            status, _, body = request(server, "/api/review?patient_id=p006266")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["mode"], "retrospective_review")
            self.assertEqual(data.calls[-1], ("review", "p006266", "baseline"))

    def test_localhost_origin_boundary(self):
        with running_server() as (server, data):
            for headers in ({"Origin": "https://unrelated.example"}, {"Host": "rebind.example"},
                            {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}):
                with self.subTest(headers=headers):
                    status, _, _ = request(server, "/api/patients", headers)
                    self.assertEqual(status, 403)
            self.assertEqual(data.calls, [])
            status, _, _ = request(server, "/api/health", {"Origin": f"http://127.0.0.1:{server.server_port}"})
            self.assertEqual(status, 200)

    def test_only_whitelisted_static_files_are_served(self):
        with running_server() as (server, _):
            status, headers, body = request(server, "/")
            self.assertEqual(status, 200)
            self.assertIn("text/html", headers["Content-Type"])
            self.assertIn(b"Replay", body)
            for path in ("/../config.json", "/%2e%2e/config.json", "/processed_data/test.csv", "/src/common.py"):
                with self.subTest(path=path):
                    self.assertEqual(request(server, path)[0], 404)

    def test_input_errors_are_json_and_never_invoke_snapshot(self):
        with running_server() as (server, data):
            for path in ("/api/snapshot?patient_id=p006266", "/api/snapshot?patient_id=p006266&hour=bad",
                         "/api/snapshot?patient_id=p006266&hour=1&hour=2", "/api/patients?limit=x"):
                with self.subTest(path=path):
                    status, _, body = request(server, path)
                    self.assertEqual(status, 400)
                    self.assertTrue(json.loads(body)["error"])
            self.assertEqual(data.calls, [])
            self.assertEqual(request(server, "/api/patient?patient_id=unknown")[0], 404)
            self.assertEqual(request(server, "/api/patients?query=")[0], 200)


if __name__ == "__main__":
    unittest.main()
