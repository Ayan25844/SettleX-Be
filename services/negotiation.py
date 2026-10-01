from models.schemas import (
    BorrowerProfile,
    LenderProfile,
    LoanProposal
)

from services.verifier import verify_proposal


def find_best_deal(
    borrower: BorrowerProfile,
    lender: LenderProfile
):
    candidates = []

    # Try interest rates from lender minimum
    # up to borrower's maximum
    start_rate = lender.min_interest_rate
    end_rate = borrower.max_interest_rate

    rate = start_rate

    while rate <= end_rate:
        # Try different tenure values
        for tenure in range(
            borrower.preferred_tenure,
            min(
                borrower.max_tenure,
                lender.max_tenure
            ) + 1,
            6
        ):

            proposal = LoanProposal(
                amount=borrower.loan_amount,
                interest_rate=rate,
                tenure_months=tenure,
                upfront_payment=0
            )

            result = verify_proposal(
                borrower,
                lender,
                proposal
            )

            if result["valid"]:

                borrower_utility = result["borrower_utility"]
                lender_utility = result["lender_utility"]

                nash_product = (
                    borrower_utility *
                    lender_utility
                )

                candidates.append({
                    "proposal": proposal.model_dump(),
                    "emi": result["emi"],
                    "borrower_utility": borrower_utility,
                    "lender_utility": lender_utility,
                    "nash_product": round(nash_product, 4)
                })

        rate += 0.5

    if not candidates:
        return {
            "agreement_found": False,
            "message": "No mutually feasible agreement found.",
            "candidates_checked": 0
        }

    best_deal = max(
        candidates,
        key=lambda x: x["nash_product"]
    )

    return {
        "agreement_found": True,
        "best_deal": best_deal,
        "candidates_checked": len(candidates)
    }