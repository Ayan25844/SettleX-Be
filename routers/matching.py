from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db, is_sqlite
from dependencies.auth import get_current_user, require_borrower
from models.match import Match, MatchStatus
from models.match_schemas import (
    FindMatchesResponse,
    MatchCandidate,
    MatchDetailResponse,
    ScoreBreakdown,
)
from models.profile import BorrowerProfile, LenderProfile
from models.user import User, UserRole
from services.matching import find_and_rank_matches

router = APIRouter(prefix="/api/matching", tags=["matching"])


@router.post(
    "/find",
    response_model=FindMatchesResponse,
    summary="Find and rank compatible lenders for borrower"
)
def find_matches(
    current_user: User = Depends(require_borrower),
    db: Session = Depends(get_db)
):
    """
    Finds and ranks compatible lenders for an authenticated borrower.
    1. Loads borrower profile
    2. Retrieves active lenders
    3. Runs deterministic hard filters
    4. Calculates multi-factor match scores
    5. Ranks and records pending matches in database (preventing duplicates)
    """
    borrower = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
    if not borrower:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Borrower profile not found. Please create your profile before finding matches."
        )

    # Retrieve all lender profiles
    lenders = db.query(LenderProfile).all()

    # Run deterministic matching engine
    ranked_candidates = find_and_rank_matches(borrower, lenders)

    if not ranked_candidates:
        return FindMatchesResponse(
            matches=[],
            message="No compatible lenders found matching your criteria."
        )

    results: List[MatchCandidate] = []

    # Batch retrieve existing active matches for this borrower to eliminate N+1 network queries
    existing_matches = db.query(Match).filter(
        Match.borrower_id == borrower.id,
        Match.status.in_([MatchStatus.PENDING.value, MatchStatus.ACCEPTED.value])
    ).all()
    existing_match_map = {}
    for m in existing_matches:
        if m.lender_id not in existing_match_map:
            existing_match_map[m.lender_id] = m

    new_matches = []
    for candidate in ranked_candidates:
        lender: LenderProfile = candidate["lender"]
        score: float = candidate["match_score"]
        breakdown: dict = candidate["score_breakdown"]

        existing_match = existing_match_map.get(lender.id)
        if existing_match:
            existing_match.match_score = score
            existing_match.score_breakdown = breakdown
            existing_match.updated_at = datetime.now(timezone.utc)
        else:
            match_record = Match(
                borrower_id=borrower.id,
                lender_id=lender.id,
                match_score=score,
                score_breakdown=breakdown,
                status=MatchStatus.PENDING.value
            )
            db.add(match_record)
            existing_match_map[lender.id] = match_record
            new_matches.append(match_record)

    # Perform a single batch flush to populate IDs for newly created matches
    if new_matches:
        db.flush()

    for candidate in ranked_candidates:
        lender: LenderProfile = candidate["lender"]
        score: float = candidate["match_score"]
        breakdown: dict = candidate["score_breakdown"]
        match_record = existing_match_map[lender.id]

        results.append(
            MatchCandidate(
                match_id=match_record.id,
                lender_id=lender.id,
                match_score=score,
                status=match_record.status,
                score_breakdown=ScoreBreakdown(**breakdown),
                lender_summary={
                    "max_loan_amount": lender.max_loan_amount,
                    "min_interest_rate": lender.min_interest_rate,
                    "max_tenure": lender.max_tenure,
                    "available_capacity": lender.available_capacity,
                    "collateral_required": lender.collateral_required,
                }
            )
        )

    db.commit()

    return FindMatchesResponse(
        matches=results,
        message=f"Found {len(results)} compatible lenders."
    )


@router.get(
    "/my-matches",
    response_model=List[MatchDetailResponse],
    summary="List current user's matches"
)
def get_my_matches(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns matches relevant to the authenticated user:
    - Borrowers see matches for their borrower profile
    - Lenders see matches for their lender profile
    - Admins see all matches across the system
    """
    if current_user.role == UserRole.BORROWER:
        profile = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
        if not profile:
            return []
        matches = db.query(Match).filter_by(borrower_id=profile.id).order_by(Match.created_at.desc()).all()
        return matches

    elif current_user.role == UserRole.LENDER:
        profile = db.query(LenderProfile).filter_by(user_id=current_user.id).first()
        if not profile:
            return []
        matches = db.query(Match).filter_by(lender_id=profile.id).order_by(Match.created_at.desc()).all()
        return matches

    elif current_user.role == UserRole.ADMIN:
        matches = db.query(Match).order_by(Match.created_at.desc()).all()
        return matches

    return []


@router.post(
    "/{match_id}/accept",
    response_model=MatchDetailResponse,
    summary="Accept a pending match with capacity safety"
)
def accept_match(
    match_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Accepts a pending match.
    Transaction-safe capacity enforcement:
    - Verifies available_capacity >= borrower.loan_amount
    - Deducts borrower.loan_amount from lender.available_capacity
    - Updates match status from 'pending' to 'accepted'
    """
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found"
        )

    # Authorization verification
    is_owner = False
    if current_user.role == UserRole.BORROWER:
        b_profile = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
        if b_profile and b_profile.id == match.borrower_id:
            is_owner = True
    elif current_user.role == UserRole.LENDER:
        l_profile = db.query(LenderProfile).filter_by(user_id=current_user.id).first()
        if l_profile and l_profile.id == match.lender_id:
            is_owner = True
    elif current_user.role == UserRole.ADMIN:
        is_owner = True

    if not is_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to accept this match."
        )

    if match.status == MatchStatus.ACCEPTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Match is already accepted."
        )

    if match.status != MatchStatus.PENDING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot accept match with status '{match.status}'."
        )

    # Capacity safety check with row-level locking for PostgreSQL
    lender_query = db.query(LenderProfile).filter_by(id=match.lender_id)
    if not is_sqlite:
        lender_query = lender_query.with_for_update()
    lender = lender_query.first()

    borrower = db.query(BorrowerProfile).filter_by(id=match.borrower_id).first()

    if not lender or not borrower:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Associated borrower or lender profile not found."
        )

    # Enforce capacity safety constraint
    if lender.available_capacity < borrower.loan_amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lender capacity ({lender.available_capacity}) is insufficient for requested loan amount ({borrower.loan_amount})."
        )

    # Deduct allocated loan amount from lender available capacity
    lender.available_capacity -= borrower.loan_amount
    lender.updated_at = datetime.now(timezone.utc)

    # Transition match to accepted
    match.status = MatchStatus.ACCEPTED.value
    match.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(match)
    return match


@router.post(
    "/{match_id}/reject",
    response_model=MatchDetailResponse,
    summary="Reject a match"
)
def reject_match(
    match_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Rejects a match.
    If the match was previously accepted, restores the allocated capacity to the lender.
    """
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found"
        )

    is_owner = False
    if current_user.role == UserRole.BORROWER:
        b_profile = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
        if b_profile and b_profile.id == match.borrower_id:
            is_owner = True
    elif current_user.role == UserRole.LENDER:
        l_profile = db.query(LenderProfile).filter_by(user_id=current_user.id).first()
        if l_profile and l_profile.id == match.lender_id:
            is_owner = True
    elif current_user.role == UserRole.ADMIN:
        is_owner = True

    if not is_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to modify this match."
        )

    if match.status == MatchStatus.REJECTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Match is already rejected."
        )

    # If was accepted, restore lender capacity
    if match.status == MatchStatus.ACCEPTED.value:
        lender = db.query(LenderProfile).filter_by(id=match.lender_id).first()
        borrower = db.query(BorrowerProfile).filter_by(id=match.borrower_id).first()
        if lender and borrower:
            lender.available_capacity += borrower.loan_amount
            lender.updated_at = datetime.now(timezone.utc)

    match.status = MatchStatus.REJECTED.value
    match.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(match)
    return match
