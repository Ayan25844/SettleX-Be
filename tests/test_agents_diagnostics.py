import pytest
import json
from unittest.mock import MagicMock, patch
import openai

from models.schemas import BorrowerProfile, LenderProfile
from services.agents import (
    borrower_agent,
    lender_agent,
    parse_agent_json_response,
    get_openai_client,
)
from services.negotiation_graph import (
    negotiation_graph,
    _sanitize_error_message,
    NegotiationState,
)


# ===========================================================================
# 1. Response Parsing Robustness & Isolation
# ===========================================================================

def test_parse_agent_json_response_clean_json():
    raw = '{"position": "accept", "reason": "Looks good", "target_interest_rate": 11.5, "target_tenure_months": 36}'
    parsed = parse_agent_json_response(raw)
    assert parsed["position"] == "accept"
    assert parsed["target_interest_rate"] == 11.5
    assert parsed["target_tenure_months"] == 36


def test_parse_agent_json_response_markdown_code_blocks():
    # LLMs frequently output ```json ... ``` blocks
    raw = "```json\n{\n  \"position\": \"counter\",\n  \"reason\": \"Need higher rate\",\n  \"target_interest_rate\": 12.5,\n  \"target_tenure_months\": 42\n}\n```"
    parsed = parse_agent_json_response(raw)
    assert parsed["position"] == "counter"
    assert parsed["target_interest_rate"] == 12.5
    assert parsed["target_tenure_months"] == 42


def test_parse_agent_json_response_generic_code_blocks():
    # Plain ``` without json tag
    raw = "```\n{\"position\": \"reject\", \"reason\": \"Unacceptable terms\"}\n```"
    parsed = parse_agent_json_response(raw)
    assert parsed["position"] == "reject"
    assert parsed["reason"] == "Unacceptable terms"


def test_parse_agent_json_response_conversational_text_wrapping():
    # LLMs occasionally output conversational text around JSON
    raw = "Here is the negotiation response:\n{\"position\": \"accept\", \"reason\": \"Mutual benefit reached\"}\nHope this helps!"
    parsed = parse_agent_json_response(raw)
    assert parsed["position"] == "accept"
    assert parsed["reason"] == "Mutual benefit reached"


def test_parse_agent_json_response_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_agent_json_response("This is plain text with no json structure at all.")


def test_parse_agent_json_response_empty():
    with pytest.raises(ValueError, match="Empty response"):
        parse_agent_json_response("")


# ===========================================================================
# 2. OpenAI Request Failure Distinctions
# ===========================================================================

def test_openai_request_failure_quota_rate_limit():
    """
    Distinguishes 429 quota exhaustion / rate limit errors from model or parsing errors.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 429
    err = openai.RateLimitError(
        message="You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.",
        response=mock_response,
        body={"error": {"code": "credit_balance_exhausted"}}
    )
    mock_client.chat.completions.create.side_effect = err

    bp = BorrowerProfile(
        loan_amount=500000.0,
        monthly_income=100000.0,
        monthly_expenses=30000.0,
        existing_emi=5000.0,
        max_emi=35000.0,
        max_interest_rate=14.0,
        preferred_tenure=36,
        max_tenure=48,
    )

    with patch("services.agents.get_openai_client", return_value=mock_client):
        with pytest.raises(openai.RateLimitError) as exc_info:
            borrower_agent(bp, current_offer=None)

        assert "credits remaining" in str(exc_info.value)
        assert isinstance(exc_info.value, openai.RateLimitError)


def test_openai_authentication_failure():
    """
    Distinguishes 401 invalid API key / authentication errors.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    err = openai.AuthenticationError(
        message="Incorrect API key provided: sk-proj-12345***.",
        response=mock_response,
        body={"error": {"code": "invalid_api_key"}}
    )
    mock_client.chat.completions.create.side_effect = err

    lp = LenderProfile(
        max_loan_amount=1000000.0,
        min_interest_rate=10.0,
        max_tenure=60,
        min_expected_return=8.0,
    )

    with patch("services.agents.get_openai_client", return_value=mock_client):
        with pytest.raises(openai.AuthenticationError) as exc_info:
            lender_agent(lp, current_offer=None)

        assert isinstance(exc_info.value, openai.AuthenticationError)


def test_openai_model_access_failure():
    """
    Distinguishes 404 model not found or forbidden access errors.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    err = openai.NotFoundError(
        message="The model 'gpt-4o-mini' does not exist or you do not have access to it.",
        response=mock_response,
        body={"error": {"code": "model_not_found"}}
    )
    mock_client.chat.completions.create.side_effect = err

    bp = BorrowerProfile(
        loan_amount=500000.0,
        monthly_income=100000.0,
        monthly_expenses=30000.0,
        existing_emi=5000.0,
        max_emi=35000.0,
        max_interest_rate=14.0,
        preferred_tenure=36,
        max_tenure=48,
    )

    with patch("services.agents.get_openai_client", return_value=mock_client):
        with pytest.raises(openai.NotFoundError) as exc_info:
            borrower_agent(bp, current_offer=None)

        assert isinstance(exc_info.value, openai.NotFoundError)


# ===========================================================================
# 3. Sanitized Error Logging (Zero Secret Leakage)
# ===========================================================================

def test_sanitize_error_message():
    raw_error = Exception("Failed request with key sk-proj-1234567890abcdef1234567890 and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz")
    sanitized = _sanitize_error_message(raw_error)

    assert "sk-proj-1234567890abcdef1234567890" not in sanitized
    assert "sk-***[REDACTED]***" in sanitized
    assert "Bearer [REDACTED]" in sanitized


# ===========================================================================
# 4. LangGraph Fallback Isolation
# ===========================================================================

def test_langgraph_failure_isolation_rate_limit():
    """
    When an OpenAI 429 quota exhaustion occurs during graph execution,
    the graph must gracefully isolate the failure to fallback counters,
    prevent crashes, and terminate safely without infinite loops or invalid agreements.
    """
    mock_response = MagicMock()
    mock_response.status_code = 429
    quota_err = openai.RateLimitError(
        message="You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.",
        response=mock_response,
        body={"error": {"code": "credit_balance_exhausted"}}
    )

    def failing_agent(*args, **kwargs):
        raise quota_err

    with patch("services.negotiation_graph.borrower_agent", side_effect=failing_agent), \
         patch("services.negotiation_graph.lender_agent", side_effect=failing_agent):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "max_rounds": 2,
        }

        result = negotiation_graph.invoke(initial_state)

        # Graph completes safely without unhandled exception
        assert result["agreement_found"] is False
        assert result["status"] == "no_agreement"
        # History contains the deterministic fallback events
        assert len(result["negotiation_history"]) >= 2
        for evt in result["negotiation_history"]:
            assert evt["action"] == "counter"
            assert "fallback" in evt["reason"].lower()
