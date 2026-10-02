"""
Deterministic Matching Engine for SettleX.

Pipeline:
Borrower Profile
       ↓
Hard Filtering (Loan Amount, Interest Rate, Tenure, Capacity, Collateral)
       ↓
Compatible Lenders
       ↓
Deterministic Multi-factor Compatibility Scoring (0.0 to 1.0)
       ↓
Ranking
       ↓
Top Matches
"""

from typing import Any, Dict, List, Tuple


# Weights for the transparent compatibility score
WEIGHT_INTEREST = 0.30
WEIGHT_LOAN_AMOUNT = 0.25
WEIGHT_TENURE = 0.20
WEIGHT_COLLATERAL = 0.15
WEIGHT_CAPACITY = 0.10


def evaluate_hard_filters(borrower: Any, lender: Any) -> Tuple[bool, str | None]:
    """
    Applies strict financial and operational boundary conditions.
    Returns (True, None) if compatible, or (False, reason) if any constraint fails.
    """
    # 1. Loan amount compatibility: Borrower request cannot exceed lender max ticket size
    if borrower.loan_amount > lender.max_loan_amount:
        return (
            False,
            f"Loan amount ({borrower.loan_amount}) exceeds lender max ticket size ({lender.max_loan_amount})"
        )

    # 2. Interest rate overlap: There must be a feasible interest-rate interval
    if lender.min_interest_rate > borrower.max_interest_rate:
        return (
            False,
            f"Lender minimum rate ({lender.min_interest_rate}%) exceeds borrower maximum rate ({borrower.max_interest_rate}%)"
        )

    # 3. Tenure compatibility: Overlapping tenure range
    if borrower.preferred_tenure > lender.max_tenure:
        return (
            False,
            f"Borrower preferred tenure ({borrower.preferred_tenure}m) exceeds lender maximum tenure ({lender.max_tenure}m)"
        )

    # 4. Lender capacity: Available capacity must cover the loan amount
    if borrower.loan_amount > lender.available_capacity:
        return (
            False,
            f"Borrower loan amount ({borrower.loan_amount}) exceeds lender available capacity ({lender.available_capacity})"
        )

    # 5. Collateral constraint: Lender requiring collateral cannot match with borrower unable to provide it
    if lender.collateral_required and not borrower.collateral_required:
        return (
            False,
            "Lender mandates collateral, but borrower profile indicates no collateral available"
        )

    return (True, None)


def score_interest_rate(borrower: Any, lender: Any) -> float:
    """
    Calculates 0.0 to 1.0 score based on interest rate negotiation headroom.
    Guaranteed lender.min_interest_rate <= borrower.max_interest_rate by hard filters.
    """
    if borrower.max_interest_rate <= 0:
        return 0.5
    rate_ratio = lender.min_interest_rate / borrower.max_interest_rate
    score = 1.0 - (rate_ratio * 0.5)
    return round(min(1.0, max(0.0, score)), 4)


def score_loan_amount(borrower: Any, lender: Any) -> float:
    """
    Calculates 0.0 to 1.0 score based on ticket-size match.
    Guaranteed borrower.loan_amount <= lender.max_loan_amount by hard filters.
    """
    if lender.max_loan_amount <= 0:
        return 0.5
    amount_ratio = borrower.loan_amount / lender.max_loan_amount
    score = 1.0 - (0.4 * (1.0 - amount_ratio))
    return round(min(1.0, max(0.0, score)), 4)


def score_tenure(borrower: Any, lender: Any) -> float:
    """
    Calculates 0.0 to 1.0 score based on tenure accommodation.
    Guaranteed borrower.preferred_tenure <= lender.max_tenure by hard filters.
    """
    if lender.max_tenure >= borrower.max_tenure:
        return 1.0

    tenure_spread = max(1, borrower.max_tenure - borrower.preferred_tenure)
    coverage = (lender.max_tenure - borrower.preferred_tenure) / tenure_spread
    score = 0.7 + (0.3 * coverage)
    return round(min(1.0, max(0.0, score)), 4)


def score_collateral(borrower: Any, lender: Any) -> float:
    """
    Calculates 0.0 to 1.0 score based on collateral alignment.
    """
    if lender.collateral_required == borrower.collateral_required:
        return 1.0
    if not lender.collateral_required and borrower.collateral_required:
        return 0.95
    return 0.0


def score_capacity(borrower: Any, lender: Any) -> float:
    """
    Calculates 0.0 to 1.0 score based on remaining lender liquidity risk.
    Guaranteed borrower.loan_amount <= lender.available_capacity by hard filters.
    """
    if lender.available_capacity <= 0:
        return 0.5
    utilization = borrower.loan_amount / lender.available_capacity
    score = 0.6 + (0.4 * (1.0 - utilization))
    return round(min(1.0, max(0.0, score)), 4)


def calculate_match_score(borrower: Any, lender: Any) -> Tuple[float, Dict[str, float]]:
    """
    Computes deterministic multi-factor match score and detailed breakdown.
    Formula:
      30% interest rate
      25% loan amount
      20% tenure
      15% collateral
      10% lender capacity
    """
    s_interest = score_interest_rate(borrower, lender)
    s_amount = score_loan_amount(borrower, lender)
    s_tenure = score_tenure(borrower, lender)
    s_collateral = score_collateral(borrower, lender)
    s_capacity = score_capacity(borrower, lender)

    composite_score = (
        WEIGHT_INTEREST * s_interest +
        WEIGHT_LOAN_AMOUNT * s_amount +
        WEIGHT_TENURE * s_tenure +
        WEIGHT_COLLATERAL * s_collateral +
        WEIGHT_CAPACITY * s_capacity
    )

    breakdown = {
        "interest": s_interest,
        "loan_amount": s_amount,
        "tenure": s_tenure,
        "collateral": s_collateral,
        "capacity": s_capacity,
    }

    return round(composite_score, 4), breakdown


def find_and_rank_matches(
    borrower: Any,
    lenders: List[Any]
) -> List[Dict[str, Any]]:
    """
    Filters and ranks candidate lenders for a given borrower.
    Returns ranked list of compatible lenders with match scores and breakdowns.
    """
    candidates = []

    for lender in lenders:
        is_compatible, _ = evaluate_hard_filters(borrower, lender)
        if not is_compatible:
            continue

        score, breakdown = calculate_match_score(borrower, lender)
        candidates.append({
            "lender": lender,
            "match_score": score,
            "score_breakdown": breakdown,
        })

    # Rank descending by match_score
    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates
