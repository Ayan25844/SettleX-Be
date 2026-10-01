from models.schemas import (
    BorrowerProfile,
    LenderProfile,
    LoanProposal
)

from services.financial import (
    calculate_emi,
    calculate_borrower_utility,
    calculate_lender_utility
)


def verify_proposal(
    borrower: BorrowerProfile,
    lender: LenderProfile,
    proposal: LoanProposal
):

    violations = []

    # -------------------------
    # Borrower constraints
    # -------------------------

    if proposal.amount != borrower.loan_amount:
        violations.append(
            "Proposal amount does not match requested loan amount."
        )

    if proposal.interest_rate > borrower.max_interest_rate:
        violations.append(
            "Interest rate exceeds borrower's maximum."
        )

    # -------------------------
    # Lender constraints
    # -------------------------

    if proposal.amount > lender.max_loan_amount:
        violations.append(
            "Loan amount exceeds lender's maximum."
        )

    if proposal.interest_rate < lender.min_interest_rate:
        violations.append(
            "Interest rate is below lender's minimum."
        )

    if proposal.tenure_months > lender.max_tenure:
        violations.append(
            "Tenure exceeds lender's maximum."
        )

    # -------------------------
    # EMI calculation
    # -------------------------

    emi = calculate_emi(
        proposal.amount,
        proposal.interest_rate,
        proposal.tenure_months
    )

    if emi > borrower.max_emi:
        violations.append(
            "EMI exceeds borrower's maximum affordable EMI."
        )

    # -------------------------
    # Utilities
    # -------------------------

    borrower_utility = calculate_borrower_utility(
        emi,
        proposal.interest_rate,
        borrower.max_emi,
        borrower.max_interest_rate
    )

    lender_utility = calculate_lender_utility(
        proposal.interest_rate,
        lender.min_interest_rate
    )

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "emi": emi,
        "borrower_utility": borrower_utility,
        "lender_utility": lender_utility
    }