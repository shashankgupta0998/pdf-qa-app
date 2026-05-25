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
    # With retry loop: 1 answer + (MAX_REFINE_RETRIES+1) critic + (MAX_REFINE_RETRIES+1) refiner
    from app import MAX_REFINE_RETRIES
    expected = 1 + (MAX_REFINE_RETRIES + 1) + (MAX_REFINE_RETRIES + 1)
    assert mock_model.invoke.call_count == expected


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


def test_build_context_sorts_by_score_and_labels():
    """build_context sorts chunks by score descending and labels them."""
    from app import build_context

    doc1 = Document(page_content="Low relevance", metadata={"source": "a.pdf", "page": 0})
    doc2 = Document(page_content="High relevance", metadata={"source": "b.pdf", "page": 1})
    doc3 = Document(page_content="Mid relevance", metadata={"source": "a.pdf", "page": 2})
    chunks = [(doc1, 0.40), (doc2, 0.95), (doc3, 0.60)]

    context = build_context(chunks)

    lines = context.split("\n")
    assert "[Most Relevant]" in context
    assert "[Supporting]" in context
    high_pos = context.index("High relevance")
    low_pos = context.index("Low relevance")
    assert high_pos < low_pos


def test_build_context_includes_source_metadata():
    """build_context includes source filename and page in each chunk header."""
    from app import build_context

    doc = Document(page_content="Some text", metadata={"source": "/path/to/report.pdf", "page": 3})
    chunks = [(doc, 0.85)]

    context = build_context(chunks)

    assert "report.pdf" in context
    assert "p.3" in context


def test_answer_agent_prompt_contains_few_shot_examples():
    """answer_agent system prompt includes few-shot examples for borderline cases."""
    from app import answer_agent, AgentResult

    doc = Document(page_content="Revenue grew 15% in Q3.", metadata={"source": "report.pdf", "page": 1})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Revenue grew 15%."
        mock_model.invoke.return_value = mock_response

        answer_agent("What was Q3 revenue?", chunks, [])

        call_args = mock_model.invoke.call_args[0][0]
        system_msg = call_args[0].content

        assert "Example" in system_msg or "example" in system_msg
        assert "Partially answerable" in system_msg or "partially" in system_msg.lower()


def test_answer_agent_prompt_requests_citations():
    """answer_agent system prompt instructs model to cite sources."""
    from app import answer_agent, AgentResult

    doc = Document(page_content="Python was created in 1991.", metadata={"source": "history.pdf", "page": 5})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Python was created in 1991. [Source: history.pdf, p.5]"
        mock_model.invoke.return_value = mock_response

        answer_agent("When was Python created?", chunks, [])

        call_args = mock_model.invoke.call_args[0][0]
        system_msg = call_args[0].content

        assert "cite" in system_msg.lower() or "[Source:" in system_msg


def test_refiner_agent_prompt_requests_citations():
    """refiner_agent system prompt instructs model to preserve source citations."""
    from app import refiner_agent, AgentResult

    doc = Document(page_content="Data here.", metadata={"source": "report.pdf", "page": 2})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Refined. [Source: report.pdf, p.2]"
        mock_model.invoke.return_value = mock_response

        refiner_agent("question", chunks, "initial", "NO ISSUES", [])

        call_args = mock_model.invoke.call_args[0][0]
        system_msg = call_args[0].content

        assert "cite" in system_msg.lower() or "citation" in system_msg.lower() or "[Source:" in system_msg


def test_answer_agent_prompt_has_anti_fabrication_instructions():
    """answer_agent prompt explicitly allows 'unclear' / 'not stated' outputs."""
    from app import answer_agent, AgentResult

    doc = Document(page_content="Some content.", metadata={"source": "doc.pdf", "page": 0})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "Answer."
        mock_model.invoke.return_value = mock_response

        answer_agent("question", chunks, [])

        call_args = mock_model.invoke.call_args[0][0]
        system_msg = call_args[0].content

        assert "unclear" in system_msg.lower() or "not stated" in system_msg.lower()
        assert "do not" in system_msg.lower() and ("guess" in system_msg.lower() or "fabricat" in system_msg.lower() or "supplement" in system_msg.lower())


def test_critic_agent_checks_for_fabrication():
    """critic_agent prompt specifically checks for fabricated/unsourced claims."""
    from app import critic_agent, AgentResult

    doc = Document(page_content="Content.", metadata={"source": "doc.pdf", "page": 0})
    chunks = [(doc, 0.9)]

    with patch("app.model") as mock_model:
        mock_response = MagicMock()
        mock_response.content = "NO ISSUES"
        mock_model.invoke.return_value = mock_response

        critic_agent("question", chunks, "answer")

        call_args = mock_model.invoke.call_args[0][0]
        system_msg = call_args[0].content

        assert "fabricat" in system_msg.lower() or "invented" in system_msg.lower() or "not in the source" in system_msg.lower()


def test_orchestrator_retries_when_critic_finds_issues():
    """orchestrator retries refiner when critic finds issues, up to MAX_REFINE_RETRIES."""
    from app import orchestrator, AgentResult, MAX_REFINE_RETRIES

    mock_vectorstore = MagicMock()
    doc = Document(page_content="Python was created by Guido.", metadata={"source": "test.pdf", "page": 0})
    mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.92),
    ]

    call_count = 0

    def invoke_side_effect(messages):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        if call_count == 1:
            # answer agent
            mock_resp.content = "Initial answer."
        elif call_count % 2 == 0:
            # critic calls — always find issues (to trigger retries)
            mock_resp.content = "ISSUE: Missing detail about year."
        else:
            # refiner calls
            mock_resp.content = f"Refined answer attempt {call_count}."
        return mock_resp

    with patch("app.model") as mock_model:
        mock_model.invoke.side_effect = invoke_side_effect
        result = orchestrator(mock_vectorstore, "Who created Python?", [])

    assert isinstance(result, AgentResult)
    assert result.status == "success"
    # 1 answer + (MAX_REFINE_RETRIES + 1) critic + (MAX_REFINE_RETRIES + 1) refiner
    expected_calls = 1 + (MAX_REFINE_RETRIES + 1) + (MAX_REFINE_RETRIES + 1)
    assert mock_model.invoke.call_count == expected_calls
    assert result.metadata.get("retries", 0) == MAX_REFINE_RETRIES


def test_orchestrator_stops_retrying_when_critic_approves():
    """orchestrator stops retry loop when critic says NO ISSUES."""
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
        mock_resp = MagicMock()
        if call_count == 1:
            mock_resp.content = "Initial answer."
        elif call_count == 2:
            # first critic — finds issue
            mock_resp.content = "ISSUE: missing year."
        elif call_count == 3:
            # first refiner
            mock_resp.content = "Python was created by Guido in 1991."
        elif call_count == 4:
            # second critic — approves
            mock_resp.content = "NO ISSUES"
        return mock_resp

    with patch("app.model") as mock_model:
        mock_model.invoke.side_effect = invoke_side_effect
        result = orchestrator(mock_vectorstore, "Who created Python?", [])

    assert result.status == "success"
    assert result.data == "Python was created by Guido in 1991."
    # Should have stopped: 1 answer + 2 critics + 1 refiner = 4 calls
    assert mock_model.invoke.call_count == 4
    assert result.metadata.get("retries", 0) == 1
