from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.orm import Session

from database import get_db
from models.user import User, UserRole
from services.auth import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=True
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Validates the JWT access token and loads the authoritative user from the database.
    Rejects missing, invalid, or expired tokens with 401 Unauthorized.
    Rejects deactivated accounts with 403 Forbidden.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise credentials_exception

    sub: str = payload.get("sub")
    email: str = payload.get("email")

    if sub is None and email is None:
        raise credentials_exception

    user = None
    if sub is not None:
        try:
            user_id = int(sub)
            user = db.query(User).filter(User.id == user_id).first()
        except (ValueError, TypeError):
            pass

    if user is None and email is not None:
        user = db.query(User).filter(User.email == email.strip().lower()).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    return user


def require_authenticated_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency that guarantees an active, authenticated user."""
    return current_user


def require_borrower(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency that restricts access to users with the 'borrower' role."""
    if current_user.role != UserRole.BORROWER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not permitted: Borrower role required"
        )
    return current_user


def require_lender(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency that restricts access to users with the 'lender' role."""
    if current_user.role != UserRole.LENDER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not permitted: Lender role required"
        )
    return current_user


def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency that restricts access to users with the 'admin' role."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not permitted: Admin role required"
        )
    return current_user
