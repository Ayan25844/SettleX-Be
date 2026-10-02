from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from dependencies.auth import require_admin
from models.auth_schemas import RoleUpdateRequest, UserResponse
from models.user import User, UserRole

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="List all users"
)
def list_users(
    db: Session = Depends(get_db)
):
    """Retrieve all users across the system. Password hashes are never returned."""
    users = db.query(User).order_by(User.id).all()
    return users


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get user details"
)
def get_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Retrieve details for a specific user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.patch(
    "/users/{user_id}/activate",
    response_model=UserResponse,
    summary="Activate user account"
)
def activate_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Activate a deactivated user account."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user.is_active = True
    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


@router.patch(
    "/users/{user_id}/deactivate",
    response_model=UserResponse,
    summary="Deactivate user account"
)
def deactivate_user(
    user_id: int,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Deactivate a user account.
    Prevents an admin from deactivating themselves or deactivating the only active admin.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin cannot deactivate their own account"
        )

    if user.role == UserRole.ADMIN:
        other_active_admins = db.query(User).filter(
            User.role == UserRole.ADMIN,
            User.is_active == True,
            User.id != user_id
        ).count()
        if other_active_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate the only active admin account"
            )

    user.is_active = False
    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Change user role"
)
def update_user_role(
    user_id: int,
    role_update: RoleUpdateRequest,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Update a user's role.
    Prevents demoting the only active admin.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    new_role = UserRole(role_update.role.strip().lower())

    # Protect against demoting the sole active admin
    if user.role == UserRole.ADMIN and new_role != UserRole.ADMIN:
        other_active_admins = db.query(User).filter(
            User.role == UserRole.ADMIN,
            User.is_active == True,
            User.id != user_id
        ).count()
        if other_active_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the only active admin"
            )

    user.role = new_role
    user.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user
