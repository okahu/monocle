import os
import logging
import unittest
from common.dummy_class import DummyClass
from common.utils import verify_scope, verify_traceID, SCOPE_NAME, SCOPE_VALUE
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace import setup_monocle_telemetry, monocle_trace_scope, start_scope, stop_scope, monocle_trace_scope_method
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from opentelemetry.trace.status import StatusCode

logger = logging.getLogger(__name__)

class TestHandler(unittest.IsolatedAsyncioTestCase):
    instrumentor = None
    exporter = None
    dummy = None
    
    def setUp(self):
        if self.exporter:
            self.exporter.reset()
            # Reset span processors to reconnect with the exporter
            from monocle_apptrace.instrumentation.common.instrumentor import reset_span_processors
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            reset_span_processors([SimpleSpanProcessor(self.exporter)])

    def tearDown(self):
        return super().tearDown()
    
    @classmethod
    def tearDownClass(cls):
        if cls.instrumentor is not None:
            cls.instrumentor.uninstrument()
        cls.instrumentor = None
        cls.exporter = None
        super().tearDownClass()
    
    @classmethod
    def setUpClass(cls):
        # Clean up any existing instrumentation from previous tests
        from monocle_apptrace.instrumentation.common.instrumentor import (
            get_monocle_instrumentor,
            set_monocle_instrumentor,
            set_monocle_setup_signature,
            set_monocle_span_processor,
            set_tracer_provider
        )
        
        existing_instrumentor = get_monocle_instrumentor()
        if existing_instrumentor is not None:
            try:
                existing_instrumentor.uninstrument()
            except:
                pass
            # Manually clear global state
            set_monocle_instrumentor(None)
            set_monocle_setup_signature(None)
            set_monocle_span_processor(None)
            set_tracer_provider(None)
        
        cls.exporter = CustomConsoleSpanExporter()
        cls.dummy = DummyClass()
        cls.exporter.reset()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="async_test",
            span_processors=[
                    SimpleSpanProcessor(cls.exporter)
                ],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add1",
                    span_name="add1",
                    wrapper_method= atask_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add2",
                    span_name="add2",
                    wrapper_method= atask_wrapper
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add3", 
                    span_name="add3",
                    wrapper_method= atask_wrapper
                )                        
        ])
        return super().setUpClass()

    # verify nested async calls have same traceID
    async def test_nested_async_traceID(self):
        res = await self.dummy.add1(10)
        assert res == 16
        verify_traceID(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_API(self):
        token = start_scope(SCOPE_NAME, SCOPE_VALUE)
        res = await self.dummy.add1(10)
        stop_scope(token)
        assert res == 16
        verify_scope(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_wrapper(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = await self.dummy.add1(10)
        assert res == 16
        verify_scope(self.exporter, excepted_span_count=4)

    # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_wrapper_errors(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            try:
                res = await self.dummy.add1(10,raise_error=True)
                assert False
            except Exception as e:
                logger.info(f"Got exception {e}")
        self.exporter.force_flush()
        spans = self.exporter.captured_spans
        assert len(spans) == 4
        traceID = None
        for span in spans:
            assert span.attributes.get("scope."+SCOPE_NAME) == SCOPE_VALUE
            if span.attributes.get("span.type") != "workflow":
                assert span.status.status_code == StatusCode.ERROR
            if traceID == None:
                traceID = span.context.trace_id
            else:
                assert traceID == span.context.trace_id

    # verify nested async calls have same scope set using scope decorator
    async def test_nested_async_scopes_with_decorator(self):
        res = await self.dummy.scope_async_decorator_test_method()
        assert res == 16
        verify_scope(self.exporter, excepted_span_count=4)

     # verify nested async calls have same scope set using scope API
    async def test_nested_async_scopes_with_config(self):
        with monocle_trace_scope(SCOPE_NAME, SCOPE_VALUE):
            res = await self.dummy.scope_async_config_test_method()
        assert res == 16
        # Give time for async spans to be exported
        import asyncio
        await asyncio.sleep(0.1)
        verify_scope(self.exporter, excepted_span_count=4)

if __name__ == '__main__':
    unittest.main()