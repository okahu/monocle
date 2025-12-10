import logging
from opentelemetry.trace.status import StatusCode

logger  = logging.getLogger(__name__)

# Constants for scope name and value used across tests
SCOPE_NAME = "test_scope"
SCOPE_VALUE = "test_value"
# Multiple scopes for testing scope_values
MULTIPLE_SCOPES = {
    "test_scope1": "test_value1",
    "test_scope2": "test_value2",
    "test_scope3": "test_value3"
}

def verify_traceID(exporter, excepted_span_count=None):
    """
    Verify that all spans in the exporter have the same trace ID.
    
    Args:
        exporter: The span exporter containing captured spans.
        excepted_span_count: Optional expected number of spans.
    """
    exporter.force_flush()
    spans = exporter.captured_spans
    
    if excepted_span_count is not None:
        assert len(spans) == excepted_span_count, f"Expected {excepted_span_count} spans, got {len(spans)}"
    
    if not spans:
        return
    
    # Get the trace ID from the first span
    trace_id = spans[0].context.trace_id
    
    # Verify all spans have the same trace ID
    for span in spans:
        assert span.context.trace_id == trace_id, f"Span trace ID mismatch: {span.context.trace_id} != {trace_id}"

def verify_scope(exporter, excepted_span_count=None):
    """
    Verify that all spans in the exporter have the same scope attribute.
    
    Args:
        exporter: The span exporter containing captured spans.
        excepted_span_count: Optional expected number of spans.
    """
    exporter.force_flush()
    spans = exporter.captured_spans
    
    if excepted_span_count is not None:
        assert len(spans) == excepted_span_count, f"Expected {excepted_span_count} spans, got {len(spans)}"
    
    if not spans:
        return
    
    # Verify all spans have the correct scope attribute
    trace_id = None
    for span in spans:
        assert span.attributes.get(f"scope.{SCOPE_NAME}") == SCOPE_VALUE, f"Span is missing expected scope attribute"
        
        # Also verify trace ID consistency
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id, f"Span trace ID mismatch: {span.context.trace_id} != {trace_id}"

def verify_multiple_scopes(exporter, scopes_dict, excepted_span_count=None):
    """
    Verify that all spans in the exporter have multiple scope attributes.
    
    Args:
        exporter: The span exporter containing captured spans.
        scopes_dict: Dictionary of scope names to scope values.
        excepted_span_count: Optional expected number of spans.
    """
    exporter.force_flush()
    spans = exporter.captured_spans
    
    if excepted_span_count is not None:
        assert len(spans) == excepted_span_count, f"Expected {excepted_span_count} spans, got {len(spans)}"
    
    if not spans:
        return
    
    # Verify all spans have the correct scope attributes
    trace_id = None
    for span in spans:
        logger.info(f"Checking span: {span.name}")
        logger.info(f"Span attributes: {span.attributes}")
        for scope_name, scope_value in scopes_dict.items():
            scope_attr_name = f"scope.{scope_name}"
            assert span.attributes.get(scope_attr_name) == scope_value, \
                f"Span is missing expected scope attribute: {scope_attr_name}={scope_value}"
        
        # Also verify trace ID consistency
        if trace_id is None:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id, f"Span trace ID mismatch: {span.context.trace_id} != {trace_id}"

def get_scope_values_from_args(args, kwargs):
    """
    Extracts scope values from args and kwargs for testing dynamic scope values.
    For this test, we'll deliberately set user.id to the session_id value to match what we're seeing.
    """
    scopes = {
        "user_id": kwargs.get("user_id", args[0] if args else None),
        "session_id": kwargs.get("session_id", args[1] if len(args) > 1 else None),
    }
    
    
    
    return scopes

def verify_inference_span(span_attributes, span_events, provider_type, model_name, expected_endpoint=None):
    """
    Verify inference span attributes and events.
    
    Args:
        span_attributes: The span attributes dictionary.
        span_events: The span events list.
        provider_type: Expected provider type (e.g., "inference.gemini", "inference.openai").
        model_name: Expected model name (e.g., "gemini-2.0-flash", "gpt-4").
        expected_endpoint: Optional expected inference endpoint.
    
    Returns:
        bool: True if verification passes.
    """
    logger.info(f"---------------------- Verifying inference span ------------------------")
    logger.info(f"provider_type: {provider_type}, model: {model_name}")
    logger.info(f"span.type: {span_attributes['span.type']}")
    logger.info(f"entity.1.type: {span_attributes.get('entity.1.type')}")
    logger.info(f"entity.1.provider_name: {span_attributes.get('entity.1.provider_name')}")
    logger.info(f"entity.1.inference_endpoint: {span_attributes.get('entity.1.inference_endpoint')}")
    logger.info(f"entity.2.name: {span_attributes.get('entity.2.name')}")
    logger.info(f"entity.2.type: {span_attributes.get('entity.2.type')}")
    logger.info(f"entity.2.model_version: {span_attributes.get('entity.2.model_version')}")
    
    print(f"\n{'='*80}")
    print(f"INFERENCE SPAN ATTRIBUTES")
    print(f"{'='*80}")
    print(f"Expected Provider: {provider_type}, Model: {model_name}")
    print(f"span.type: {span_attributes.get('span.type')}")
    print(f"entity.1.type: {span_attributes.get('entity.1.type')}")
    print(f"entity.1.provider_name: {span_attributes.get('entity.1.provider_name')}")
    print(f"entity.1.inference_endpoint: {span_attributes.get('entity.1.inference_endpoint')}")
    print(f"entity.2.name: {span_attributes.get('entity.2.name')}")
    print(f"entity.2.type: {span_attributes.get('entity.2.type')}")
    print(f"entity.2.model_version: {span_attributes.get('entity.2.model_version')}")
    
    # span_input, span_output, span_metadata = span_events[0], span_events[1], span_events[2]
    span_input, span_output, span_metadata = span_events[0], span_events[1], span_events[2]
    
    print(f"\nEVENTS ({len(span_events)} found)")
    print(f"-" * 80)
    input_text = span_input.attributes.get('input', '')
    print(f"Input: {input_text[:100]}..." if len(input_text) > 100 else f"Input: {input_text}")
    response_text = span_output.attributes.get('response', '')
    print(f"Response: {response_text[:100]}..." if len(response_text) > 100 else f"Response: {response_text}")
    print(f"Token Metadata:")
    print(f"  - Prompt tokens: {span_metadata.attributes.get('prompt_tokens')}")
    print(f"  - Completion tokens: {span_metadata.attributes.get('completion_tokens')}")
    print(f"  - Total tokens: {span_metadata.attributes.get('total_tokens')}")
    if 'finish_type' in span_metadata.attributes:
        print(f"  - Finish type: {span_metadata.attributes['finish_type']}")
    if 'finish_reason' in span_metadata.attributes:
        print(f"  - Finish reason: {span_metadata.attributes['finish_reason']}")
    print(f"{'='*80}\n")
    logger.info(f"span_events: {len(span_events)} events found")
    logger.info(f"span_input.attributes[\"input\"]: {span_input.attributes['input']}")
    logger.info(f"span_input.attributes[\"response\"]: {span_output.attributes['response']}")
    logger.info(f"✓ Token metadata - prompt: {span_metadata.attributes['prompt_tokens']}, "
                f"completion: {span_metadata.attributes['completion_tokens']}, "
                f"total: {span_metadata.attributes['total_tokens']}")

    # Verify span type
    assert span_attributes["span.type"] in ["inference", "inference.framework"], \
        f"Expected span.type to be 'inference' or 'inference.framework', got {span_attributes.get('span.type')}"
    logger.info(f"✓ Span type verified: {span_attributes['span.type']}")
    
    # Verify entity.1 (inference provider)
    assert span_attributes["entity.1.type"] == provider_type, \
        f"Expected entity.1.type to be '{provider_type}', got {span_attributes.get('entity.1.type')}"
    logger.info(f"✓ Provider type verified: {provider_type}")
    
    assert "entity.1.provider_name" in span_attributes, \
        "Missing entity.1.provider_name attribute"
    logger.info(f"✓ Provider name found: {span_attributes['entity.1.provider_name']}")
    
    assert "entity.1.inference_endpoint" in span_attributes, \
        "Missing entity.1.inference_endpoint attribute"
    logger.info(f"✓ Inference endpoint found: {span_attributes['entity.1.inference_endpoint']}")
    
    # if expected_endpoint:
    #     assert span_attributes["entity.1.inference_endpoint"] == expected_endpoint, \
    #         f"Expected endpoint '{expected_endpoint}', got {span_attributes.get('entity.1.inference_endpoint')}"
    #     logger.info(f"✓ Endpoint matches expected: {expected_endpoint}")
    
    # Verify entity.2 (model)
    assert span_attributes["entity.2.name"] == model_name, \
        f"Expected entity.2.name to be '{model_name}', got {span_attributes.get('entity.2.name')}"
    logger.info(f"✓ Model name verified: {model_name}")
    
    expected_model_type = f"model.llm.{model_name}"
    assert span_attributes["entity.2.type"] == expected_model_type, \
        f"Expected entity.2.type to be '{expected_model_type}', got {span_attributes.get('entity.2.type')}"
    logger.info(f"✓ Model type verified: {expected_model_type}")
    
    # Verify events (input, output, metadata)
    assert len(span_events) >= 3, \
        f"Expected at least 3 events (input, output, metadata), got {len(span_events)}"
    
    # span_input, span_output, span_metadata = span_events[0], span_events[1], span_events[2]
    
    # Verify input event
    assert "input" in span_input.attributes, "Missing 'input' in span input event"
    assert span_input.attributes["input"] is not None and span_input.attributes["input"] != "", \
        "Input attribute is empty or None"
    logger.info(f"✓ Input event verified with {len(span_input.attributes['input'])} characters")
    
    # Verify output event
    assert "response" in span_output.attributes, "Missing 'response' in span output event"
    assert span_output.attributes["response"] is not None and span_output.attributes["response"] != "", \
        "Response attribute is empty or None"
    logger.info(f"✓ Output event verified with {len(span_output.attributes['response'])} characters")
    
    # Verify metadata event
    assert "completion_tokens" in span_metadata.attributes, "Missing 'completion_tokens' in metadata"
    assert "prompt_tokens" in span_metadata.attributes, "Missing 'prompt_tokens' in metadata"
    assert "total_tokens" in span_metadata.attributes, "Missing 'total_tokens' in metadata"
    logger.info(f"✓ Token metadata verified - prompt: {span_metadata.attributes['prompt_tokens']}, "
                f"completion: {span_metadata.attributes['completion_tokens']}, "
                f"total: {span_metadata.attributes['total_tokens']}")
    
    # Verify finish reason and type if present
    if "finish_type" in span_metadata.attributes:
        logger.info(f"✓ Finish type found: {span_metadata.attributes['finish_type']}")
    
    if "finish_reason" in span_metadata.attributes:
        logger.info(f"✓ Finish reason found: {span_metadata.attributes['finish_reason']}")
    
    logger.info("✓ All inference span verifications passed")
    return True