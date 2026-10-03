from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from database import get_db
from dependencies.auth import get_current_user
from models.auth_schemas import LoginResponse, MessageResponse, RegisterRequest, UserResponse
from models.user import User, UserRole
from services.auth import create_access_token, hash_password, verify_password

router = APIRouter()


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserResponse,
    summary="Register a new borrower or lender"
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db)
):
    """
    Public registration endpoint.
    Only 'borrower' and 'lender' roles can be created publicly.
    Admin registration is strictly rejected.
    """
    role_clean = request.role.strip().lower()

    # Reject public admin registration
    if role_clean == "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin registration is not allowed"
        )

    if role_clean not in (UserRole.BORROWER.value, UserRole.LENDER.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'borrower' or 'lender'"
        )

    normalized_email = request.email.strip().lower()

    # Check for duplicate email
    existing_user = db.query(User).filter(User.email == normalized_email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    # Hash the password
    password_hash = hash_password(request.password)

    # Create and persist the user
    new_user = User(
        full_name=request.full_name.strip(),
        email=normalized_email,
        password_hash=password_hash,
        role=UserRole(role_clean),
        is_active=True
    )

    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user due to a database error."
        )

    return new_user


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User Login",
    description="OAuth2 password flow compatible login. Accepts form-urlencoded credentials or JSON payload with email/username and password."
)
async def login(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Authenticates user with username (email) and password.
    Returns a signed JWT access token and safe user details.
    """
    username = None
    password = None

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                username = body.get("username") or body.get("email")
                password = body.get("password")
        except Exception:
            pass
    else:
        try:
            form = await request.form()
            username = form.get("username") or form.get("email")
            password = form.get("password")
        except Exception:
            pass

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    normalized_email = str(username).strip().lower()

    # Look up user in database off the event loop
    user = await run_in_threadpool(
        lambda: db.query(User).filter(User.email == normalized_email).first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password with Argon2 off the event loop
    password_valid = await run_in_threadpool(
        lambda: verify_password(str(password), user.password_hash)
    )

    if not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    # Generate access token
    role_value = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": role_value
        }
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Current User"
)
def get_me(
    current_user: User = Depends(get_current_user)
):
    """Returns the currently authenticated user profile."""
    return current_user


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout User"
)
def logout():
    """
    Because JWT access tokens are stateless, do NOT falsely claim that this endpoint
    invalidates an already-issued access token.
    Production logout would normally use:
    - refresh-token revocation
    - token blacklist
    - or short-lived access tokens
    For this hackathon, the frontend will clear its local auth state.
    """
    return MessageResponse(message="Logged out successfully")
