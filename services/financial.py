def calculate_emi(
    principal: float,
    annual_interest_rate: float,
    tenure_months: int
) -> float:

    monthly_rate = annual_interest_rate / 12 / 100

    if monthly_rate == 0:
        return round(principal / tenure_months, 2)

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** tenure_months
        / ((1 + monthly_rate) ** tenure_months - 1)
    )

    return round(emi, 2)


def calculate_total_repayment(
    emi: float,
    tenure_months: int
) -> float:

    return round(emi * tenure_months, 2)

def calculate_borrower_utility(
    emi: float,
    interest_rate: float,
    max_emi: float,
    max_interest_rate: float
) -> float:

    emi_score = max(
        0,
        1 - (emi / max_emi)
    )

    rate_score = max(
        0,
        1 - (interest_rate / max_interest_rate)
    )

    utility = (
        0.6 * emi_score +
        0.4 * rate_score
    )

    return round(
        min(utility, 1),
        3
    )

def calculate_lender_utility(
    interest_rate: float,
    min_interest_rate: float
) -> float:

    if interest_rate < min_interest_rate:
        return 0.0

    utility = (
        interest_rate / min_interest_rate
    ) / 1.5

    return round(
        min(utility, 1),
        3
    )