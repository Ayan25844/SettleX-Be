from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ScoreBreakdown(BaseModel):
    interest: float = Field(..., ge=0, le=1.0)
    loan_amount: float = Field(..., ge=0, le=1.0)
    tenure: float = Field(..., ge=0, le=1.0)
    collateral: float = Field(..., ge=0, le=1.0)
    capacity: float = Field(..., ge=0, le=1.0)


class MatchCandidate(BaseModel):
    match_id: int
    lender_id: int
    match_score: float
    status: str
    score_breakdown: ScoreBreakdown
    lender_summary: Optional[Dict[str, Any]] = None


class FindMatchesResponse(BaseModel):
    matches: List[MatchCandidate]
    message: Optional[str] = None


class MatchDetailResponse(BaseModel):
    id: int
    borrower_id: int
    lender_id: int
    match_score: float
    status: str
    score_breakdown: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AdminMatchResponse(BaseModel):
    id: int
    borrower_id: int
    lender_id: int
    match_score: float
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
