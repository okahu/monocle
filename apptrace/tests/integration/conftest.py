"""Shared fixtures for integration tests.

Monocle telemetry is process-global: ``setup_monocle_telemetry`` installs a single
instrumentor plus an OpenTelemetry ``TracerProvider`` and, since PR #478, refuses to
re-initialize while an instrumentor already exists (see ``check_duplicate_setup``).
OpenTelemetry itself only honors the *first* ``set_tracer_provider`` call per process.

When the whole suite runs in one pytest process, state therefore leaks across modules:
a module that does not fully tear telemetry down leaves its instrumentor and its
provider (with its ``service.name``) in place, so the next module's ``setup_monocle_telemetry``
becomes a no-op and its spans/service name never take effect.

The autouse fixture below gives every test module a clean slate by fully resetting the
global telemetry state before it runs, restoring per-module isolation.
"""

import pytest
from opentelemetry import trace
from opentelemetry.util._once import Once


def _reset_global_telemetry():
    """Fully reset Monocle + OpenTelemetry global tracing state."""
    from monocle_apptrace.instrumentation.common.instrumentor import (
        get_monocle_instrumentor,
        set_monocle_instrumentor,
        set_monocle_setup_signature,
        set_monocle_span_processor,
        set_tracer_provider,
    )

    instrumentor = get_monocle_instrumentor()
    if instrumentor is not None:
        try:
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()
        except Exception:
            pass

    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)
    set_monocle_span_processor(None)
    set_tracer_provider(None)

    # OpenTelemetry only allows set_tracer_provider() to succeed once per process; reset
    # the guard so the next module can install its own provider (and thus its service.name).
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


@pytest.fixture(scope="module", autouse=True)
def reset_monocle_telemetry():
    _reset_global_telemetry()
    yield
    _reset_global_telemetry()
