import json
import logging
from typing import Callable, Optional, Sequence
import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, ConsoleSpanExporter
from requests.exceptions import ReadTimeout
from monocle_apptrace.exporters.base_exporter import SpanExporterBase, serialize_span
from monocle_apptrace.exporters.exporter_processor import ExportTaskProcessor
from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value, get_monocle_env_value_with_source

REQUESTS_SUCCESS_STATUS_CODES = (200, 202, 204)
REQUESTS_AUTH_FAILURE_STATUS_CODES = (401, 403)
OKAHU_PROD_INGEST_ENDPOINT = "https://ingest.okahu.co/api/v1/trace/ingest"

logger = logging.getLogger(__name__)


def _get_monocle_exporter() -> Optional[str]:
    return get_monocle_env_value("MONOCLE_EXPORTER")


def _mask_api_key(api_key: str) -> str:
    """Show just enough of the key to tell two keys apart in logs."""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}...{api_key[-4:]}"


def _verify_enabled() -> bool:
    return (get_monocle_env_value("MONOCLE_OKAHU_VERIFY") or "").lower() in ("1", "true", "yes")


class OkahuSpanExporter(SpanExporterBase):
    def __init__(
            self,
            endpoint: Optional[str] = None,
            timeout: Optional[int] = None,
            session: Optional[requests.Session] = None,
            task_processor: ExportTaskProcessor = None,
            evaluate: Optional[bool] = False
    ):
        """Okahu exporter."""
        super().__init__()
        okahu_endpoint: str = get_monocle_env_value("OKAHU_INGESTION_ENDPOINT") or OKAHU_PROD_INGEST_ENDPOINT
        if evaluate:
            okahu_endpoint = okahu_endpoint.replace("/trace/ingest", "/eval/ingest")
        self.endpoint = endpoint or okahu_endpoint
        api_key, api_key_source = get_monocle_env_value_with_source("OKAHU_API_KEY")
        self._closed = False
        # HTTP status of the most recent synchronous upload, so a caller that
        # gets SpanExportResult.FAILURE can say why (None: no response, e.g. timeout).
        self.last_status_code: Optional[int] = None
        # Set to the HTTP status once Okahu rejects the key; later uploads are skipped
        # since the key can't change for the life of this exporter.
        self._auth_failure_status: Optional[int] = None
        if not api_key:
            raise ValueError("OKAHU_API_KEY not set.")
        self._api_key_description = f"{_mask_api_key(api_key)} (from {api_key_source})"
        logger.info("Okahu exporter sending traces to %s with OKAHU_API_KEY %s",
                    self.endpoint, self._api_key_description)
        self.timeout = timeout or 15
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "x-api-key": api_key}
        )

        self.task_processor = task_processor or None
        if task_processor is not None:
            task_processor.start()

        # Eval ingestion has no empty-batch contract, so only trace exporters verify.
        if _verify_enabled() and not evaluate:
            self.verify_credentials()

    def verify_credentials(self) -> bool:
        """Send an empty batch to check the endpoint accepts the API key, so a bad
        key is reported at startup rather than on the first real upload."""
        try:
            result = self.session.post(url=self.endpoint, data=json.dumps({"batch": []}), timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.warning("Could not reach Okahu at %s to verify OKAHU_API_KEY: %s", self.endpoint, str(e))
            return False
        if result.status_code in REQUESTS_AUTH_FAILURE_STATUS_CODES:
            self._report_auth_failure(result.status_code, result.text)
            return False
        if result.status_code not in REQUESTS_SUCCESS_STATUS_CODES:
            logger.warning("Okahu credential check at %s returned status code %s: %s",
                           self.endpoint, result.status_code, result.text)
            return False
        logger.info("Okahu accepted OKAHU_API_KEY %s at %s", self._api_key_description, self.endpoint)
        return True

    def _report_auth_failure(self, status_code: int, message: str) -> None:
        if self._auth_failure_status is not None:
            return
        self._auth_failure_status = status_code
        logger.error(
            "Okahu rejected OKAHU_API_KEY %s at %s; status code: %s, message %s. "
            "Check that the key was issued for this Okahu environment (staging and production keys differ, "
            "see OKAHU_INGESTION_ENDPOINT) and that an OKAHU_API_KEY already set in the shell is not "
            "overriding the one in your .env file. Further trace uploads from this exporter are skipped.",
            self._api_key_description, self.endpoint, status_code, message,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        # After the call to Shutdown subsequent calls to Export are
        # not allowed and should return a Failure result
        if not hasattr(self, 'session'):
            return self.exporter.export(spans)
        
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring batch")
            return SpanExportResult.FAILURE
        
        span_list = {
            "batch": []
        }

        # append the batch object with all the spans object
        for span in spans:
            if self.skip_export(span):
                continue
            span_list["batch"].append(serialize_span(span))

        # if there are no spans to export after filtering, then return
        if len(span_list["batch"]) == 0:
            return
        
        # Calculate is_root_span by checking if any span has no parent
        is_root_span = any(OkahuSpanExporter._is_root_span(span) for span in spans)

        def send_spans_to_okahu(span_list_local=None, is_root=False):
            if self._auth_failure_status is not None:
                self.last_status_code = self._auth_failure_status
                logger.debug("Skipping Okahu upload; API key was rejected earlier")
                return SpanExportResult.FAILURE
            try:
                self.last_status_code = None
                result = self.session.post(
                    url=self.endpoint,
                    data=json.dumps(span_list_local),
                    timeout=self.timeout,
                )
                self.last_status_code = result.status_code
                if result.status_code in REQUESTS_AUTH_FAILURE_STATUS_CODES:
                    self._report_auth_failure(result.status_code, result.text)
                    return SpanExportResult.FAILURE
                if result.status_code not in REQUESTS_SUCCESS_STATUS_CODES:
                    logger.error(
                        "Traces cannot be uploaded; status code: %s, message %s",
                        result.status_code,
                        result.text,
                    )
                    return SpanExportResult.FAILURE
                logger.debug("spans successfully exported to okahu. Is root span: %s", is_root)
                return SpanExportResult.SUCCESS
            except ReadTimeout as e:
                logger.warning("Trace export timed out: %s", str(e))
                return SpanExportResult.FAILURE

        # if async task function is present, then push the request to asnc task
        if self.task_processor is not None and callable(self.task_processor.queue_task):
            self.task_processor.queue_task(
                send_spans_to_okahu,
                kwargs={'span_list_local': span_list, 'is_root': is_root_span},
                is_root_span=is_root_span
            )
            return SpanExportResult.SUCCESS
        return send_spans_to_okahu(span_list, is_root_span)

    @staticmethod
    def _is_root_span(span: ReadableSpan) -> bool:
        """Return True if the span has no parent (i.e. it is a root span)."""
        return (not span.parent) or (span.attributes.get("span.type") == "workflow")

    def shutdown(self) -> None:
        if self._closed:
            logger.warning("Exporter already shutdown, ignoring call")
            return
        if hasattr(self, 'session'):
            self.session.close()
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
    