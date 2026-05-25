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


from unittest.mock import patch


def test_answer_agent_returns_agent_result():
    """answer_agent returns AgentResult wrapping the LLM response."""
    from app import answer_agent, AgentResult

    doc = Document(page_content="Python was created by Guido van Rossum.", metadata={})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Python was created by Guido van Rossum."
        mock_model.invoke.return_value = mock_response

        result = answer_agent("Who created Python?", chunks, [])

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert result.data == "Python was created by Guido van Rossum."
    assert result.metadata["agent"] == "answer"


def test_answer_agent_handles_llm_error():
    """answer_agent returns error AgentResult when LLM call fails."""
    from app import answer_agent, AgentResult

    doc = Document(page_content="Some content", metadata={})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_model.invoke.side_effect = Exception("rate limit exceeded")

        result = answer_agent("question", chunks, [])

    assert result.status == "error"
    assert "rate limit" in result.error
    assert result.is_retryable is True


def test_critic_agent_returns_agent_result():
    """critic_agent returns AgentResult wrapping the critique."""
    from app import critic_agent, AgentResult

    doc = Document(page_content="Content here", metadata={})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "NO ISSUES"
        mock_model.invoke.return_value = mock_response

        result = critic_agent("question", chunks, "the answer")

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert result.data == "NO ISSUES"


def test_refiner_agent_returns_agent_result():
    """refiner_agent returns AgentResult wrapping the refined answer."""
    from app import refiner_agent, AgentResult

    doc = Document(page_content="Content here", metadata={})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Refined answer here."
        mock_model.invoke.return_value = mock_response

        result = refiner_agent("question", chunks, "initial answer", "NO ISSUES", [])

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert result.data == "Refined answer here."


def test_orchestrator_short_circuits_on_no_results():
    """orchestrator returns a friendly message when retrieval finds nothing relevant."""
    from app import orchestrator, AgentResult

    mock_vectorstore = MagicMock()
    doc = Document(page_content="Irrelevant", metadata={"source": "test.pdf", "page": 0})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.10),
    ]

    result = orchestrator(mock_vectorstore, "What is quantum computing?", [])

    assert isinstance(result, AgentResult)
    assert result.status == "no_results"
    assert "not found" in result.data.lower() or "no relevant" in result.data.lower() or "couldn't find" in result.data.lower()


def test_orchestrator_short_circuits_on_retrieval_error():
    """orchestrator returns error when retrieval fails."""
    from app import orchestrator, AgentResult

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_relevance_scores.side_effect = Exception("disk full")

    result = orchestrator(mock_vectorstore, "Any question", [])

    assert isinstance(result, AgentResult)
    assert result.status == "error"
    assert "disk full" in result.error


def test_orchestrator_runs_full_pipeline_on_success():
    """orchestrator runs all agents when retrieval returns good chunks."""
    from app import orchestrator, AgentResult

    mock_vectorstore = MagicMock()
    doc = Document(page_content="Python was created by Guido.", metadata={"source": "test.pdf", "page": 0})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.92),
    ]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Python was created by Guido van Rossum."
        mock_model.invoke.return_value = mock_response

        result = orchestrator(mock_vectorstore, "Who created Python?", [])

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert len(result.data) > 0
    # model.invoke called 3 times: answer, critic, refiner
    assert mock_model.invoke.call_count == 3


def test_orchestrator_degrades_gracefully_on_critic_failure():
    """orchestrator returns initial answer with degraded flag when critic fails."""
    from app import orchestrator, AgentResult

    mock_vectorstore = MagicMock()
    doc = Document(page_content="Python was created by Guido.", metadata={"source": "test.pdf", "page": 0})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.92),
    ]

    call_count = 0

    def invoke_side_effect(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            mock_resp = MagicMock()
            mock_resp.content = "Initial answer from answer agent."
            return mock_resp
        # Second call is critic — fail it
        raise Exception("API timeout")

    with patch("app.model") as mock_model:
        mock_model.invoke.side_effect = invoke_side_effect
        result = orchestrator(mock_vectorstore, "Who created Python?", [])

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert result.data == "Initial answer from answer agent."
    assert result.metadata.get("degraded") is True
    assert result.metadata.get("failed_stage") == "critic"
