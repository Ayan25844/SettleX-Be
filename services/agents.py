import os
import json
import time
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

from models.schemas import BorrowerProfile, LenderProfile

# Ensure environment variables from .env are loaded
load_dotenv()

logger = logging.getLogger(__name__)

# Safely log whether the key is present without exposing its value
has_key = bool(os.getenv("GEMINI_API_KEY"))
logger.info(f"GEMINI_API_KEY configured: {has_key}")

# Configured Gemini model constant (flash-tier)
GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]



def get_gemini_client() -> genai.Client:
    """
    Initializes and returns a Google Gemini client using the current environment.
    Loads .env dynamically to ensure any updated runtime keys are available.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not configured or is empty.")
    return genai.Client(api_key=api_key)


# Module-level client instance for backward compatibility
try:
    client = get_gemini_client()
except Exception:
    client = None

# Backward compatibility alias
get_openai_client = get_gemini_client


def _generate_gemini_content(active_client: genai.Client, prompt: str) -> str:
    """
    Calls Gemini API with structured JSON response config.
    Falls back gracefully to active models if the configured model is unavailable,
    and retries on temporary 503 service spikes.
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )

    models_to_try = [GEMINI_MODEL]
    for model_name in FALLBACK_GEMINI_MODELS:
        if model_name not in models_to_try:
            models_to_try.append(model_name)

    last_error = None
    for _ in range(2):
        for model_name in models_to_try:
            try:
                response = active_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                if "not_found" in err_msg or "404" in err_msg or "no longer available" in err_msg:
                    # Model not available for new users, try next model immediately
                    continue
                if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                    # Quota exhausted on this model, try next model immediately
                    continue
                if "503" in err_msg or "unavailable" in err_msg or "high demand" in err_msg:
                    # Temporary Google API spike, brief sleep and try next model / next attempt
                    time.sleep(1.0)
                    continue

                raise e

    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate content from Gemini API.")


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
    Aggressively advocates for borrower's interest minimization and affordability.
    """
    cur_rate_str = f"{current_offer.get('interest_rate')}%" if current_offer and current_offer.get('interest_rate') is not None else "the current rate"
    prompt = f"""You are the Borrower Advocate AI for a structured bilateral financial negotiation.
Your primary fiduciary duty is to advocate for the borrower's financial interests and long-term affordability.

Borrower Profile & Hard Constraints:
- Loan amount: ₹{borrower.loan_amount:,.2f}
- Maximum affordable EMI limit: ₹{borrower.max_emi:,.2f}/month
- Maximum acceptable interest rate ceiling: {borrower.max_interest_rate}%
- Preferred loan tenure: {borrower.preferred_tenure} months
- Maximum allowable tenure: {borrower.max_tenure} months

Current offer under evaluation:
{json.dumps(current_offer, indent=2)}

Borrower Strategic Objectives & Decision Logic:
1. Interest Rate Minimization: Your core objective is to minimize borrowing cost. If the offered interest rate exceeds your ceiling ({borrower.max_interest_rate}%), you must REJECT or COUNTER.
2. Tenure Alignment: Prioritize the borrower's preferred tenure ({borrower.preferred_tenure} months). Never exceed the maximum tenure ({borrower.max_tenure} months).
3. Affordability & EMI: Ensure the monthly payment remains comfortably within the borrower's maximum affordable EMI limit.
4. Position Selection & Deal Closing:
   - "accept": Choose this if the offer provides an attractive, competitive interest rate (at least 1.0% below your ceiling of {borrower.max_interest_rate}%) and aligns with your preferred tenure ({borrower.preferred_tenure} months) and EMI capacity. Do not counter perpetually if the terms are already favorable and affordable.
   - "counter": Choose this if the offered rate is high (near your ceiling of {borrower.max_interest_rate}%) or if tenure requires adjustment, and there is meaningful room to negotiate lower borrowing costs. CRITICAL: When countering, you MUST specify a concrete new `target_interest_rate` that is strictly LOWER than {cur_rate_str} (e.g. 0.5% to 1.5% lower). Never echo or return the current offer rate when countering.
   - "reject": Choose this ONLY if the terms fundamentally violate constraints or cannot be reconciled.

Return a valid JSON object matching this schema:
{{
    "position": "accept|counter|reject",
    "reason": "Borrower-centric explanation focusing on borrowing cost, EMI burden, and affordability",
    "target_interest_rate": <number, strictly lower than current rate if countering>,
    "target_tenure_months": <integer>
}}

Important constraints:
- Do not invent financial facts.
- Do not disclose private limits (such as exact max EMI or rate ceiling) to the counterparty in your reason.
- Keep the reason concise, professional, and strictly from the borrower advocate perspective.
"""
    try:
        active_client = get_gemini_client()
        raw_content = _generate_gemini_content(active_client, prompt)
        parsed = parse_agent_json_response(raw_content)
        logger.info("[Borrower Advocate] Gemini call: PASS")
        return parsed
    except Exception as e:
        logger.error(f"[Borrower Advocate] Gemini call: FAIL | Exception class: {type(e).__name__}")
        raise


def lender_agent(
    lender: LenderProfile,
    current_offer: dict | None = None
) -> dict:
    """
    Lender Advocate AI:
    Evaluates current offer and returns structured JSON position (accept, counter, reject).
    Protects lender's capital, return hurdles, and underwriting constraints.
    """
    cur_rate_str = f"{current_offer.get('interest_rate')}%" if current_offer and current_offer.get('interest_rate') is not None else "the current rate"
    prompt = f"""You are the Lender Advocate AI for a structured bilateral financial negotiation.
Your primary fiduciary duty is to protect the lending institution's capital, enforce risk-adjusted return hurdles, and optimize profitability.

Lender Criteria & Underwriting Constraints:
- Maximum loan amount limit: ₹{lender.max_loan_amount:,.2f}
- Minimum acceptable interest rate (hurdle rate): {lender.min_interest_rate}%
- Maximum allowable tenure: {lender.max_tenure} months
- Minimum expected return metric: {lender.min_expected_return}
- Collateral required: {lender.collateral_required}

Current offer under evaluation:
{json.dumps(current_offer, indent=2)}

Lender Strategic Objectives & Decision Logic:
1. Return Protection & Margin: Your core objective is to protect lender return and risk-adjusted yield. Never accept an interest rate below your minimum hurdle rate ({lender.min_interest_rate}%).
2. Tenure & Duration Risk: Keep loan tenure within {lender.max_tenure} months to manage duration risk and maintain capital velocity.
3. Capacity & Collateral Adherence: Ensure the loan amount does not exceed lending capacity and collateral rules are respected.
4. Position Selection & Deal Closing:
   - "accept": Choose this if the offer provides an adequate return meeting or exceeding your required return criteria (at least 0.75% above your minimum hurdle rate of {lender.min_interest_rate}%), and the tenure/principal terms are acceptable. Do not counter perpetually if the return is satisfactory.
   - "counter": Choose this if the proposed rate is at or near the minimum hurdle ({lender.min_interest_rate}%) and leaves inadequate margin, or if tenure requires adjustment. CRITICAL: When countering, you MUST specify a concrete new `target_interest_rate` that is strictly HIGHER than {cur_rate_str} (e.g. 0.5% to 1.5% higher). Never echo or return the current offer rate when countering.
   - "reject": Choose this ONLY if the proposal breaches non-negotiable risk or capacity constraints.

Return a valid JSON object matching this schema:
{{
    "position": "accept|counter|reject",
    "reason": "Lender-centric explanation focusing on capital preservation, risk-adjusted yield, and portfolio criteria",
    "target_interest_rate": <number, strictly higher than current rate if countering>,
    "target_tenure_months": <integer>
}}

Important constraints:
- Do not invent financial facts.
- Do not disclose private reservation hurdles (such as exact minimum interest rate) to the counterparty in your reason.
- Keep the reason concise, professional, and strictly from the lender risk/return perspective.
"""
    try:
        active_client = get_gemini_client()
        raw_content = _generate_gemini_content(active_client, prompt)
        parsed = parse_agent_json_response(raw_content)
        logger.info("[Lender Advocate] Gemini call: PASS")
        return parsed
    except Exception as e:
        logger.error(f"[Lender Advocate] Gemini call: FAIL | Exception class: {type(e).__name__}")
        raise