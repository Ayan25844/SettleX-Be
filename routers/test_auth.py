from fastapi import APIRouter, Depends

from dependencies.auth import require_admin, require_borrower, require_lender
from models.auth_schemas import MessageResponse
from models.user import User

router = APIRouter()


@router.get(
    "/borrower",
    response_model=MessageResponse,
    summary="Borrower access test endpoint"
)
def verify_borrower_access(
    current_user: User = Depends(require_borrower)
):
    """Accessible only to authenticated users with the 'borrower' role."""
    return MessageResponse(message="Borrower access granted")


@router.get(
    "/lender",
    response_model=MessageResponse,
    summary="Lender access test endpoint"
)
def verify_lender_access(
    current_user: User = Depends(require_lender)
):
    """Accessible only to authenticated users with the 'lender' role."""
    return MessageResponse(message="Lender access granted")


@router.get(
    "/admin",
    response_model=MessageResponse,
    summary="Admin access test endpoint"
)
def verify_admin_access(
    current_user: User = Depends(require_admin)
):
    """Accessible only to authenticated users with the 'admin' role."""
    return MessageResponse(message="Admin access granted")

