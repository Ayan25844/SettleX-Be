from pydantic import BaseModel, Field


class BorrowerProfile(BaseModel):
    loan_amount: float = Field(gt=0)
    monthly_income: float = Field(gt=0)
    monthly_expenses: float = Field(ge=0)
    existing_emi: float = Field(ge=0)

    max_emi: float = Field(gt=0)
    max_interest_rate: float = Field(gt=0)

    preferred_tenure: int = Field(gt=0)
    max_tenure: int = Field(gt=0)

    collateral_required: bool = False


class LenderProfile(BaseModel):
    max_loan_amount: float = Field(gt=0)
    min_interest_rate: float = Field(gt=0)
    max_tenure: int = Field(gt=0)

    min_expected_return: float = Field(gt=0)

    collateral_required: bool = False


class LoanProposal(BaseModel):
    amount: float = Field(gt=0)
    interest_rate: float = Field(gt=0)
    tenure_months: int = Field(gt=0)
    upfront_payment: float = Field(ge=0, default=0)


class EvaluationRequest(BaseModel):
    borrower: BorrowerProfile
    lender: LenderProfile
    proposal: LoanProposal