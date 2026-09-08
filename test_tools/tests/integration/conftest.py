# load the plugin for local test runs without installing the package
# pytest_plugins = ["monocle_test_tools.pytest_plugin"]
# Load the plugin manually only when the package is NOT installed (e.g. local
# runs that rely on `pythonpath = ["src", ...]`). When the package IS installed,
# its `pytest11` entry point already registers this plugin; registering it again
# here would raise "Plugin already registered under a different name".
from importlib.metadata import entry_points as _entry_points


def _plugin_registered_via_entrypoint() -> bool:
    eps = _entry_points()
    # Python 3.10+ exposes the selectable API; 3.8/3.9 return a dict.
    group = eps.select(group="pytest11") if hasattr(eps, "select") else eps.get("pytest11", [])
    return any(ep.value == "monocle_test_tools.pytest_plugin" for ep in group)


if not _plugin_registered_via_entrypoint():
    pytest_plugins = ["monocle_test_tools.pytest_plugin"]

# Trace-return must be enabled BEFORE the first setup_monocle_telemetry() call,
# because that is when the trace-return SpanProcessor is wired onto the tracer
# provider. Test modules that set this at their own import time lose the race:
# pytest imports modules alphabetically during collection, and an earlier module
# (e.g. test_crewai_simple_agent, whose @MonocleValidator().monocle_testcase
# decorator runs at import) constructs the singleton validator first -- so the
# flag arrives too late and the feature is silently dead for the whole session.
# conftest is imported before every test module in this directory, so setting it
# here wins that race.
#
# This is inert for tests that do not opt in: the retrieval KEYS
# (MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY / _KEY) are read live per request, so
# without them the server never authorizes a trailer and HttpRunner never
# injects the header. Only the modules that set those keys see any behaviour.
# setdefault so a caller can still force it off.
import os

os.environ.setdefault("MONOCLE_ENABLE_TRACE_RETURN", "true")
