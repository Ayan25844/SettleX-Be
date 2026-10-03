from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from dependencies.auth import get_current_user
from models.match import Match, MatchStatus
from models.negotiation import NegotiationOffer, NegotiationSession, SessionStatus
from models.negotiation_schemas import (
    CreateSessionRequest,
    NegotiationSessionResponse,
    OfferResponse,
)
from models.profile import BorrowerProfile, LenderProfile
from models.user import User, UserRole
from services.negotiation_graph import run_negotiation_session

router = APIRouter(prefix="/api/negotiations", tags=["negotiations"])


def _is_session_participant(session: NegotiationSession, user: User) -> bool:
    return (
        user.role == UserRole.ADMIN
        or session.borrower.user_id == user.id
        or session.lender.user_id == user.id
    )


def _session_response(
    session: NegotiationSession,
    history: Optional[List[Dict[str, Any]]] = None,
    verification: Optional[Dict[str, Any]] = None,
    demo_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    from services.demo_negotiation import is_demo_mode

    if demo_mode is None:
        demo_mode = is_demo_mode()

    if history is None:
        history = [
            {
                "round": offer.round_number,
                "agent": offer.agent_type,
                "action": offer.position,
                "reason": offer.reason,
                "offer": {
                    "amount": offer.amount,
                    "interest_rate": offer.interest_rate,
                    "tenure_months": offer.tenure_months,
                    "upfront_payment": offer.upfront_payment,
                },
                "verification": offer.verification_result,
            }
            for offer in session.offers
        ]

    if verification is None:
        verification = next(
            (
                offer.verification_result
                for offer in reversed(session.offers)
                if offer.verification_result is not None
            ),
            None,
        )

    return {
        "session_id": session.id,
        "match_id": session.match_id,
        "borrower_id": session.borrower_id,
        "lender_id": session.lender_id,
        "status": session.status,
        "round_number": session.current_round,
        "max_rounds": session.max_rounds,
        "agreement_found": session.agreement_found,
        "current_offer": session.current_offer,
        "final_proposal": session.final_proposal,
        "verification": verification,
        "history": history,
        "demo_mode": demo_mode,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@router.post(
    "/session",
    response_model=NegotiationSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a negotiation session for an accepted match",
)
def create_negotiation_session(
    request: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    borrower = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
    if not borrower:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the borrower in a match can create its negotiation session.",
        )

    match = db.query(Match).filter_by(id=request.match_id).first()
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found.",
        )
    if match.borrower_id != borrower.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to create a session for this match.",
        )

    existing = (
        db.query(NegotiationSession)
        .filter_by(match_id=match.id)
        .first()
    )
    if existing:
        return _session_response(existing)

    if match.status != MatchStatus.ACCEPTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Match must be 'accepted' before a negotiation session can be created.",
        )

    session = NegotiationSession(
        match_id=match.id,
        borrower_id=match.borrower_id,
        lender_id=match.lender_id,
        status=SessionStatus.PENDING.value,
        current_round=1,
        max_rounds=6,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_response(session, history=[])


@router.post(
    "/{session_id}/start",
    response_model=NegotiationSessionResponse,
    summary="Start a negotiation session",
)
def start_negotiation_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = db.query(NegotiationSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Negotiation session not found.",
        )
    if not _is_session_participant(session, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to access this negotiation session.",
        )

    try:
        result = run_negotiation_session(session.id, db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    db.refresh(session)
    return _session_response(
        session,
        history=result.get("negotiation_history", []),
        verification=result.get("verification_result"),
        demo_mode=result.get("demo_mode"),
    )


@router.get(
    "/{session_id}",
    response_model=NegotiationSessionResponse,
    summary="Get a negotiation session",
)
def get_negotiation_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = db.query(NegotiationSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Negotiation session not found.",
        )
    if not _is_session_participant(session, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to access this negotiation session.",
        )
    return _session_response(session)


@router.get(
    "/{session_id}/offers",
    response_model=List[OfferResponse],
    summary="List the offers in a negotiation session",
)
def get_negotiation_offers(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = db.query(NegotiationSession).filter_by(id=session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Negotiation session not found.",
        )
    if not _is_session_participant(session, current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to access this negotiation session.",
        )
    return (
        db.query(NegotiationOffer)
        .filter_by(session_id=session.id)
        .order_by(NegotiationOffer.id)
        .all()
    )