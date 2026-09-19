import pytest
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
