import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from models.schemas import BorrowerProfile, LenderProfile

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def borrower_agent(
    borrower: BorrowerProfile,
    current_offer: dict | None = None
):

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

Return JSON only:

{{
    "position": "accept|counter|reject",
    "reason": "short explanation",
    "target_interest_rate": number,
    "target_tenure_months": number
}}

Do not calculate or invent financial facts.
Do not reveal private borrower limits unnecessarily.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful financial negotiation agent."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return json.loads(
        response.choices[0].message.content
    )


def lender_agent(
    lender: LenderProfile,
    current_offer: dict | None = None
):

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

Return JSON only:

{{
    "position": "accept|counter|reject",
    "reason": "short explanation",
    "target_interest_rate": number,
    "target_tenure_months": number
}}

Do not calculate or invent financial facts.
Do not reveal private lender limits unnecessarily.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful financial negotiation agent."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return json.loads(
        response.choices[0].message.content
    )