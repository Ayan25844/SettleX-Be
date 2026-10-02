from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from dependencies.auth import require_borrower, require_lender
from models.profile import BorrowerProfile, LenderProfile
from models.profile_schemas import (
    BorrowerProfileCreate,
    BorrowerProfileResponse,
    BorrowerProfileUpdate,
    LenderProfileCreate,
    LenderProfileResponse,
    LenderProfileUpdate,
)
from models.user import User

borrower_router = APIRouter(prefix="/api/borrower", tags=["borrower-profile"])
lender_router = APIRouter(prefix="/api/lender", tags=["lender-profile"])


# ==============================================================================
# Borrower Profile Endpoints
# ==============================================================================

@borrower_router.post(
    "/profile",
    response_model=BorrowerProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create borrower profile"
)
def create_borrower_profile(
    payload: BorrowerProfileCreate,
    current_user: User = Depends(require_borrower),
    db: Session = Depends(get_db)
):
    """Creates a persistent borrower profile for the authenticated borrower."""
    existing = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Borrower profile already exists. Use PUT to update."
        )

    profile = BorrowerProfile(
        user_id=current_user.id,
        loan_amount=payload.loan_amount,
        monthly_income=payload.monthly_income,
        monthly_expenses=payload.monthly_expenses,
        existing_emi=payload.existing_emi,
        max_emi=payload.max_emi,
        max_interest_rate=payload.max_interest_rate,
        preferred_tenure=payload.preferred_tenure,
        max_tenure=payload.max_tenure,
        collateral_required=payload.collateral_required
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


@borrower_router.get(
    "/profile",
    response_model=BorrowerProfileResponse,
    summary="Get borrower profile"
)
def get_borrower_profile(
    current_user: User = Depends(require_borrower),
    db: Session = Depends(get_db)
):
    """Retrieves the profile of the authenticated borrower."""
    profile = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Borrower profile not found"
        )
    return profile


@borrower_router.put(
    "/profile",
    response_model=BorrowerProfileResponse,
    summary="Update borrower profile"
)
def update_borrower_profile(
    payload: BorrowerProfileUpdate,
    current_user: User = Depends(require_borrower),
    db: Session = Depends(get_db)
):
    """Updates the profile of the authenticated borrower."""
    profile = db.query(BorrowerProfile).filter_by(user_id=current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Borrower profile not found"
        )

    update_data = payload.model_dump(exclude_unset=True)

    # Check tenure consistency if tenure fields are being updated
    pref_t = update_data.get("preferred_tenure", profile.preferred_tenure)
    max_t = update_data.get("max_tenure", profile.max_tenure)
    if max_t < pref_t:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_tenure must be greater than or equal to preferred_tenure"
        )

    for field, value in update_data.items():
        setattr(profile, field, value)

    profile.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(profile)
    return profile


# ==============================================================================
# Lender Profile Endpoints
# ==============================================================================

@lender_router.post(
    "/profile",
    response_model=LenderProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create lender profile"
)
def create_lender_profile(
    payload: LenderProfileCreate,
    current_user: User = Depends(require_lender),
    db: Session = Depends(get_db)
):
    """Creates a persistent lender profile for the authenticated lender."""
    existing = db.query(LenderProfile).filter_by(user_id=current_user.id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lender profile already exists. Use PUT to update."
        )

    profile = LenderProfile(
        user_id=current_user.id,
        max_loan_amount=payload.max_loan_amount,
        min_interest_rate=payload.min_interest_rate,
        max_tenure=payload.max_tenure,
        min_expected_return=payload.min_expected_return,
        collateral_required=payload.collateral_required,
        available_capacity=payload.available_capacity
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


@lender_router.get(
    "/profile",
    response_model=LenderProfileResponse,
    summary="Get lender profile"
)
def get_lender_profile(
    current_user: User = Depends(require_lender),
    db: Session = Depends(get_db)
):
    """Retrieves the profile of the authenticated lender."""
    profile = db.query(LenderProfile).filter_by(user_id=current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lender profile not found"
        )
    return profile


@lender_router.put(
    "/profile",
    response_model=LenderProfileResponse,
    summary="Update lender profile"
)
def update_lender_profile(
    payload: LenderProfileUpdate,
    current_user: User = Depends(require_lender),
    db: Session = Depends(get_db)
):
    """Updates the profile of the authenticated lender."""
    profile = db.query(LenderProfile).filter_by(user_id=current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lender profile not found"
        )

    update_data = payload.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(profile, field, value)

    profile.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(profile)
    return profile
