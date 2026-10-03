import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

from models.schemas import BorrowerProfile, LenderProfile

# Ensure environment variables from .env are loaded
load_dotenv()

logger = logging.getLogger(__name__)

# Safely log whether the key is present without exposing its value
has_key = bool(os.getenv("OPENAI_API_KEY"))
logger.info(f"OPENAI_API_KEY configured: {has_key}")


def get_openai_client() -> OpenAI:
    """
    Initializes and returns an OpenAI client using the current environment.
    Loads .env dynamically to ensure any updated runtime keys are available.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not configured or is empty.")
    return OpenAI(api_key=api_key)


# Module-level client instance for backward compatibility
try:
    client = get_openai_client()
except Exception:
    client = None


def parse_agent_json_response(content: str) -> dict:
    """
    Robustly parses JSON responses from LLM, stripping markdown code blocks,
    leading/trailing whitespace, or conversational wrappers if present.
    """
    if not content:
        raise ValueError("Empty response received from LLM.")

    cleaned = content.strip()

    # Strip markdown code blocks (e.g., ```json ... ```)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Direct JSON parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find outer-most JSON object bounds
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def borrower_agent(
    borrower: BorrowerProfile,
    current_offer: dict | None = None
) -> dict:
    """
    Borrower Advocate AI:
    Evaluates current offer and returns structured JSON position (accept, counter, reject).
    """
    active_client = get_openai_client()

    prompt = f"""
You are the Borrower Advocate AI for a financial negotiation.

Your job is to protect the borrower's interests.

Borrower information:
- Loan amount: {borrower.loan_amount}
- Maximum affordable EMI: {borrower.max_emi}
- Maximum acceptable interest rate: {borrower.max_interest_rate}%
- Preferred tenure: {borrower.preferred_tenure} months
- Maximum tenure: {borrower.max_tenure} months

Current offer:
{json.dumps(current_offer)}

Analyze the current offer from the borrower's perspective.

Return a valid JSON object matching this schema:
{{
    "position": "accept|counter|reject",
    "reason": "short explanation",
    "target_interest_rate": number,
    "target_tenure_months": number
}}

Do not calculate or invent financial facts.
Do not reveal private borrower limits unnecessarily.
"""

    response = active_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful financial negotiation agent. Respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    raw_content = response.choices[0].message.content
    return parse_agent_json_response(raw_content)


def lender_agent(
    lender: LenderProfile,
    current_offer: dict | None = None
) -> dict:
    """
    Lender Advocate AI:
    Evaluates current offer and returns structured JSON position (accept, counter, reject).
    """
    active_client = get_openai_client()

    prompt = f"""
You are the Lender Advocate AI for a financial negotiation.

Your job is to protect the lender's interests.

Lender information:
- Maximum loan amount: {lender.max_loan_amount}
- Minimum interest rate: {lender.min_interest_rate}%
- Maximum tenure: {lender.max_tenure} months
- Minimum expected return: {lender.min_expected_return}

Current offer:
{json.dumps(current_offer)}

Analyze the current offer from the lender's perspective.

Return a valid JSON object matching this schema:
{{
    "position": "accept|counter|reject",
    "reason": "short explanation",
    "target_interest_rate": number,
    "target_tenure_months": number
}}

Do not calculate or invent financial facts.
Do not reveal private lender limits unnecessarily.
"""

    response = active_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful financial negotiation agent. Respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    raw_content = response.choices[0].message.content
    return parse_agent_json_response(raw_content)