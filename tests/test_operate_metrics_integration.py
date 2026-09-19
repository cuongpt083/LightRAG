import pytest
from lightrag.query_metrics import QUERY_STAGE_DURATION, QUERY_DURATION, QUERY_TOTAL

def test_query_stage_definitions():
    stages = [
        "extract_keywords_llm",
        "perform_kg_search",
        "token_truncation",
        "merge_chunks",
        "vector_search_chunks",
    ]
    for s in stages:
        metric = QUERY_STAGE_DURATION.labels(mode="mix", stage=s)
        assert metric is not None
        samples = metric._samples()
        assert any(sample.name.endswith("_count") for sample in samples)

def test_query_route_metrics_definitions():
    total = QUERY_TOTAL.labels(mode="mix", status="success")
    assert total is not None
    duration = QUERY_DURATION.labels(mode="mix", status="success")
    assert duration is not None
