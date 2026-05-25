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


from unittest.mock import MagicMock
from langchain_core.documents import Document


def test_retrieval_agent_returns_agent_result_with_scores():
    """retrieval_agent returns AgentResult with chunks and scores in metadata."""
    from app import retrieval_agent, AgentResult

    mock_vectorstore = MagicMock()
    doc = Document(page_content="Test content", metadata={"source": "test.pdf", "page": 0})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.85),
    ]

    result = retrieval_agent(mock_vectorstore, "What is test?")

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert len(result.data) == 1
    assert result.data[0][0].page_content == "Test content"
    assert result.data[0][1] == 0.85
    assert result.metadata["agent"] == "retrieval"


def test_retrieval_agent_low_scores_returns_no_results():
    """retrieval_agent returns no_results when all chunks score below threshold."""
    from app import retrieval_agent, AgentResult, SIMILARITY_THRESHOLD

    mock_vectorstore = MagicMock()
    doc = Document(page_content="Irrelevant content", metadata={"source": "test.pdf", "page": 0})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.15),
        (doc, 0.10),
    ]

    result = retrieval_agent(mock_vectorstore, "Completely unrelated question?")

    assert isinstance(result, AgentResult)
    assert result.status == "no_results"
    assert result.data == []
    assert result.metadata["reason"] == "below_threshold"


def test_retrieval_agent_filters_low_scores_keeps_high():
    """retrieval_agent keeps only chunks above the threshold."""
    from app import retrieval_agent, AgentResult, SIMILARITY_THRESHOLD

    mock_vectorstore = MagicMock()
    good_doc = Document(page_content="Relevant", metadata={"source": "test.pdf", "page": 1})
    bad_doc = Document(page_content="Irrelevant", metadata={"source": "test.pdf", "page": 5})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (good_doc, 0.82),
        (bad_doc, 0.15),
        (bad_doc, 0.08),
    ]

    result = retrieval_agent(mock_vectorstore, "Relevant question")

    assert result.status == "success"
    assert len(result.data) == 1
    assert result.data[0][0].page_content == "Relevant"


def test_retrieval_agent_handles_vectorstore_error():
    """retrieval_agent returns error AgentResult when ChromaDB raises."""
    from app import retrieval_agent, AgentResult

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_relevance_scores.side_effect = Exception("connection refused")

    result = retrieval_agent(mock_vectorstore, "Any question")

    assert result.status == "error"
    assert result.is_retryable is True
    assert "connection refused" in result.error
