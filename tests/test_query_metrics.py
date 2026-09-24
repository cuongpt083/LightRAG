try:
    import pytest
except ImportError:
    class _RaisesContext:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return exc_type is not None and issubclass(exc_type, self.exc)
    class _PytestMock:
        @staticmethod
        def raises(exc):
            return _RaisesContext(exc)
    pytest = _PytestMock()
import time
from lightrag.query_metrics import (
    record_query_duration,
    record_query_total,
    record_llm_call,
    record_query_stage,
    QUERY_DURATION,
    QUERY_TOTAL,
    QUERY_STAGE_DURATION,
    LLM_CALL_DURATION,
)

def test_record_query_metrics():
    record_query_total("mix", "success")
    record_query_duration("mix", "success", 2.3)
    assert QUERY_TOTAL.labels(mode="mix", status="success")._value.get() >= 1
    assert QUERY_DURATION.labels(mode="mix", status="success")._sum.get() >= 2.3

def test_record_query_stage_context_manager():
    before = QUERY_STAGE_DURATION.labels(mode="mix", stage="extract_keywords_llm")._sum.get()
    with record_query_stage("mix", "extract_keywords_llm"):
        time.sleep(0.05)
    after = QUERY_STAGE_DURATION.labels(mode="mix", stage="extract_keywords_llm")._sum.get()
    assert after >= before + 0.04

def test_record_query_stage_catches_exception_safely():
    with pytest.raises(ValueError):
        with record_query_stage("mix", "perform_kg_search"):
            raise ValueError("Test error inside stage")
    # Verify observation was still recorded in finally block
    samples = QUERY_STAGE_DURATION.labels(mode="mix", stage="perform_kg_search")._samples()
    count = [s.value for s in samples if s.name.endswith("_count")][0]
    assert count >= 1

def test_record_llm_call():
    before = LLM_CALL_DURATION.labels(role="keyword")._sum.get()
    record_llm_call("keyword", 1.5)
    after = LLM_CALL_DURATION.labels(role="keyword")._sum.get()
    assert after >= before + 1.5

def test_init_default_metrics():
    from lightrag.query_metrics import init_default_metrics
    init_default_metrics()
    assert QUERY_TOTAL.labels(mode="mix", status="success") is not None
    assert LLM_CALL_DURATION.labels(role="keyword") is not None
