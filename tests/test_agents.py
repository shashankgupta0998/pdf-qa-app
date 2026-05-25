from dataclasses import dataclass


def test_agent_result_success():
    """AgentResult can represent a successful result with data."""
    from app import AgentResult

    result = AgentResult(
        status="success",
        data="some answer text",
        error=None,
        is_retryable=False,
        metadata={"agent": "answer"},
    )
    assert result.status == "success"
    assert result.data == "some answer text"
    assert result.error is None
    assert result.is_retryable is False
    assert result.metadata == {"agent": "answer"}


def test_agent_result_failure():
    """AgentResult can represent a failure with error details."""
    from app import AgentResult

    result = AgentResult(
        status="error",
        data=None,
        error="ChromaDB connection failed",
        is_retryable=True,
        metadata={"agent": "retrieval"},
    )
    assert result.status == "error"
    assert result.data is None
    assert result.error == "ChromaDB connection failed"
    assert result.is_retryable is True


def test_agent_result_no_results():
    """AgentResult can represent a valid-empty state (no relevant chunks found)."""
    from app import AgentResult

    result = AgentResult(
        status="no_results",
        data=[],
        error=None,
        is_retryable=False,
        metadata={"agent": "retrieval", "reason": "below_threshold"},
    )
    assert result.status == "no_results"
    assert result.data == []
