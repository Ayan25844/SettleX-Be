from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BorrowerProfileCreate(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested loan principal amount")
    monthly_income: float = Field(..., gt=0, description="Borrower monthly gross income")
    monthly_expenses: float = Field(default=0.0, ge=0, description="Monthly fixed expenses")
    existing_emi: float = Field(default=0.0, ge=0, description="Current existing monthly EMI obligations")
    max_emi: float = Field(..., gt=0, description="Maximum affordable EMI")
    max_interest_rate: float = Field(..., gt=0, description="Maximum acceptable annual interest rate (%)")
    preferred_tenure: int = Field(..., gt=0, description="Preferred tenure in months")
    max_tenure: int = Field(..., gt=0, description="Maximum acceptable tenure in months")
    collateral_required: bool = Field(default=False, description="Whether borrower is offering collateral")

    @model_validator(mode="after")
    def validate_tenure(self) -> "BorrowerProfileCreate":
        if self.max_tenure < self.preferred_tenure:
            raise ValueError("max_tenure must be greater than or equal to preferred_tenure")
        return self


class BorrowerProfileUpdate(BaseModel):
    loan_amount: Optional[float] = Field(default=None, gt=0)
    monthly_income: Optional[float] = Field(default=None, gt=0)
    monthly_expenses: Optional[float] = Field(default=None, ge=0)
    existing_emi: Optional[float] = Field(default=None, ge=0)
    max_emi: Optional[float] = Field(default=None, gt=0)
    max_interest_rate: Optional[float] = Field(default=None, gt=0)
    preferred_tenure: Optional[int] = Field(default=None, gt=0)
    max_tenure: Optional[int] = Field(default=None, gt=0)
    collateral_required: Optional[bool] = None

    @model_validator(mode="after")
    def validate_tenure(self) -> "BorrowerProfileUpdate":
        if self.max_tenure is not None and self.preferred_tenure is not None:
            if self.max_tenure < self.preferred_tenure:
                raise ValueError("max_tenure must be greater than or equal to preferred_tenure")
        return self


class BorrowerProfileResponse(BaseModel):
    id: int
    user_id: int
    loan_amount: float
    monthly_income: float
    monthly_expenses: float
    existing_emi: float
    max_emi: float
    max_interest_rate: float
    preferred_tenure: int
    max_tenure: int
    collateral_required: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class LenderProfileCreate(BaseModel):
    max_loan_amount: float = Field(..., gt=0, description="Maximum loan amount the lender can disburse")
    min_interest_rate: float = Field(..., gt=0, description="Minimum acceptable annual interest rate (%)")
    max_tenure: int = Field(..., gt=0, description="Maximum tenure allowed in months")
    min_expected_return: float = Field(..., gt=0, description="Minimum expected overall return multiplier")
    collateral_required: bool = Field(default=False, description="Whether collateral is mandatory")
    available_capacity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Available lending capacity in pool (defaults to max_loan_amount if omitted)"
    )

    @model_validator(mode="after")
    def set_default_capacity(self) -> "LenderProfileCreate":
        if self.available_capacity is None:
            self.available_capacity = self.max_loan_amount
        return self


class LenderProfileUpdate(BaseModel):
    max_loan_amount: Optional[float] = Field(default=None, gt=0)
    min_interest_rate: Optional[float] = Field(default=None, gt=0)
    max_tenure: Optional[int] = Field(default=None, gt=0)
    min_expected_return: Optional[float] = Field(default=None, gt=0)
    collateral_required: Optional[bool] = None
    available_capacity: Optional[float] = Field(default=None, ge=0)


class LenderProfileResponse(BaseModel):
    id: int
    user_id: int
    max_loan_amount: float
    min_interest_rate: float
    max_tenure: int
    min_expected_return: float
    collateral_required: bool
    available_capacity: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
