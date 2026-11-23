"""Actual pinned trained Self-RAG CPU gate, without paid providers or substitutes."""
import json
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rags.research.selfrag_adapter import search
from rags.research.selfrag_runtime import (
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    MODEL_BYTES,
    MODEL_SHA256,
    RUNTIME_REVISION,
    TOKEN_IDS,
)

pytestmark = [pytest.mark.integration, pytest.mark.external, pytest.mark.research_selfrag]


@pytest.mark.skipif(os.getenv("RAGS_TEST_SELFRAG") != "1", reason="Explicit actual trained Self-RAG CPU gate")
def test_actual_trained_selfrag_selects_original_source_and_preserves_its_generated_answer(tmp_path):
    started = time.monotonic()
    initial_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    result = search("What is the refund period?", [
        {"id": "authorized:refund", "text": "The refund period is 14 days after purchase."},
        {"id": "authorized:unrelated", "text": "Bananas are yellow fruit and dolphins swim in the ocean."},
    ], 2, tmp_path)
    assert result["grounded_answer"]["cited_ids"] == ["authorized:refund"]
    assert "14 days" in result["grounded_answer"]["text"]
    assert [hit["chunk_id"] for hit in result["hits"]] == ["authorized:refund"]
    assert result["diagnostics"]["reflection_calls"] == 2
    assert result["diagnostics"]["reflection_abstentions"] == 1
    assert result["diagnostics"]["reflection_support_tokens"] >= 1
    assert result["diagnostics"]["reflection_utility_tokens"] == 2
    assert result["usage"]["input_tokens"] > 0 and result["usage"]["output_tokens"] > 0
    assert list(tmp_path.iterdir()) == []
    report = {"schema": "selfrag-trained-cpu-v10", "status": "passed",
              "verified_at": datetime.now(UTC).isoformat(), "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "native_threads": int(os.getenv("RAGS_SELFRAG_THREADS", "4")),
              "major_page_faults": resource.getrusage(resource.RUSAGE_SELF).ru_majflt - initial_faults,
              "weight_loading": "owned_cpu_buffers",
              "kv_policy": "full_snapshot_max_prompt_plus_128_rounded_256_capped_1792",
              "checkpoint": {"repository": CHECKPOINT_REPOSITORY, "revision": CHECKPOINT_REVISION,
                             "bytes": MODEL_BYTES, "sha256": MODEL_SHA256, "quantization": "Q4_K_M"},
              "runtime_revision": RUNTIME_REVISION, "reflection_token_ids": TOKEN_IDS,
              "all_reflection_ids_are_non_eog": True,
              "evidence_kind": "actual_trained_checkpoint_with_two_synthetic_passages",
              "limitations": ["One positive and one negative fixture do not establish domain quality.",
                              "Quantized short-form CPU port; no full-precision or paper benchmark parity.",
                              "No ordinary generation provider or embedding provider is used."],
              "result": result}
    output = os.getenv("RAGS_SELFRAG_REPORT_PATH")
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "peak_process_rss_bytes": report["peak_process_rss_bytes"]}))
