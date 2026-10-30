import json
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.exporters.okahu import okahu_exporter
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter, _mask_api_key

API_KEY = "okh_abcdefgh_1234567890_wxyz"
ENDPOINT = "https://ingest-stage.okahu.co/api/v1/trace/ingest"


def _response(status_code, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def _span():
    span = MagicMock()
    span.parent = None
    span.attributes = {"span.type": "workflow"}
    return span


class TestOkahuExporter(unittest.TestCase):
    def setUp(self):
        # Isolate each test from ambient env vars and .env file lookups to prevent config leakage.
        self._env_patcher = patch.dict(
            os.environ, {"OKAHU_API_KEY": API_KEY, "OKAHU_INGESTION_ENDPOINT": ENDPOINT}, clear=True
        )
        self._env_patcher.start()
        self._env_value_patcher = patch.object(
            okahu_exporter, "get_monocle_env_value", side_effect=lambda key: os.environ.get(key)
        )
        self._env_value_patcher.start()
        self._env_value_source_patcher = patch.object(
            okahu_exporter,
            "get_monocle_env_value_with_source",
            side_effect=lambda key: (os.environ.get(key), "environment" if os.environ.get(key) else None),
        )
        self._env_value_source_patcher.start()
        self._serialize_patcher = patch.object(okahu_exporter, "serialize_span", return_value={})
        self._serialize_patcher.start()

    def tearDown(self):
        self._serialize_patcher.stop()
        self._env_value_source_patcher.stop()
        self._env_value_patcher.stop()
        self._env_patcher.stop()

    def _exporter(self, *responses, evaluate=False):
        session = MagicMock()
        session.headers = {}
        session.post.side_effect = list(responses)
        exporter = OkahuSpanExporter(session=session, evaluate=evaluate)
        exporter.skip_export = lambda span: False
        return exporter, session

    def test_mask_api_key(self):
        self.assertEqual(_mask_api_key(API_KEY), "okh_...wxyz")
        self.assertEqual(_mask_api_key("short"), "****")

    def test_logs_destination_with_masked_key(self):
        with self.assertLogs(okahu_exporter.logger, level=logging.INFO) as logs:
            self._exporter()
        output = "\n".join(logs.output)
        self.assertIn(ENDPOINT, output)
        self.assertIn("okh_...wxyz (from environment)", output)
        self.assertNotIn(API_KEY, output)

    def test_auth_failure_logged_once_and_later_uploads_skipped(self):
        exporter, session = self._exporter(_response(401, "Authentication Failed"))
        with self.assertLogs(okahu_exporter.logger, level=logging.ERROR) as logs:
            first = exporter.export([_span()])
            second = exporter.export([_span()])
        self.assertEqual(first, SpanExportResult.FAILURE)
        self.assertEqual(second, SpanExportResult.FAILURE)
        self.assertEqual(session.post.call_count, 1)
        self.assertEqual(exporter.last_status_code, 401)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("okh_...wxyz", logs.output[0])
        self.assertIn(ENDPOINT, logs.output[0])
        self.assertIn("staging and production keys differ", logs.output[0])

    def test_other_failures_keep_retrying(self):
        exporter, session = self._exporter(_response(500, "oops"), _response(204))
        with self.assertLogs(okahu_exporter.logger, level=logging.ERROR):
            self.assertEqual(exporter.export([_span()]), SpanExportResult.FAILURE)
        self.assertEqual(exporter.export([_span()]), SpanExportResult.SUCCESS)
        self.assertEqual(session.post.call_count, 2)

    def test_no_verification_by_default(self):
        _, session = self._exporter()
        session.post.assert_not_called()

    def test_verify_rejects_bad_key_at_startup(self):
        os.environ["MONOCLE_OKAHU_VERIFY"] = "true"
        with self.assertLogs(okahu_exporter.logger, level=logging.ERROR) as logs:
            exporter, session = self._exporter(_response(401, "Authentication Failed"))
        self.assertEqual(json.loads(session.post.call_args.kwargs["data"]), {"batch": []})
        self.assertIn("rejected OKAHU_API_KEY", logs.output[0])
        self.assertEqual(exporter.export([_span()]), SpanExportResult.FAILURE)
        self.assertEqual(session.post.call_count, 1)

    def test_verify_accepts_good_key(self):
        os.environ["MONOCLE_OKAHU_VERIFY"] = "true"
        with self.assertLogs(okahu_exporter.logger, level=logging.INFO) as logs:
            exporter, session = self._exporter(_response(204), _response(204))
        self.assertTrue(any("accepted" in line for line in logs.output))
        self.assertEqual(exporter.export([_span()]), SpanExportResult.SUCCESS)

    def test_verify_skipped_for_eval_exporter(self):
        os.environ["MONOCLE_OKAHU_VERIFY"] = "true"
        _, session = self._exporter(evaluate=True)
        session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
