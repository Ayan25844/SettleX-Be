from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CreateSessionRequest(BaseModel):
    match_id: int = Field(..., description="ID of the accepted match to start negotiation for")


class OfferDetail(BaseModel):
    amount: float
    interest_rate: float
    tenure_months: int
    upfront_payment: float = 0.0


class OfferResponse(BaseModel):
    id: int
    session_id: int
    round_number: int
    agent_type: str
    amount: float
    interest_rate: float
    tenure_months: int
    upfront_payment: float
    position: str
    reason: Optional[str] = None
    is_valid: bool
    verification_result: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HistoryEvent(BaseModel):
    round: int
    agent: str
    action: str
    reason: Optional[str] = None
    offer: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None


class NegotiationSessionResponse(BaseModel):
    session_id: int
    match_id: int
    borrower_id: int
    lender_id: int
    status: str
    round_number: int
    max_rounds: int
    agreement_found: bool
    current_offer: Optional[Dict[str, Any]] = None
    final_proposal: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
