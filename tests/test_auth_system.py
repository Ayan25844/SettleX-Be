import pytest
from datetime import timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app
from models.user import User, UserRole
from services.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
import jwt

# Create in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. Password Hashing and JWT Service Tests
# ==============================================================================

def test_password_hashing_and_verification():
    raw_pass = "SecurePass123"
    hashed = hash_password(raw_pass)
    assert hashed != raw_pass
    assert hashed.startswith("$argon2")
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False
    assert verify_password("", hashed) is False


def test_password_hashing_min_length():
    with pytest.raises(ValueError, match="at least 8 characters"):
        hash_password("short")


def test_jwt_creation_and_decoding():
    payload = {"sub": "42", "email": "alice@example.com", "role": "borrower"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=15))
    decoded = decode_access_token(token)
    assert decoded["sub"] == "42"
    assert decoded["email"] == "alice@example.com"
    assert decoded["role"] == "borrower"
    assert "exp" in decoded


def test_jwt_expired_token():
    payload = {"sub": "42", "email": "alice@example.com", "role": "borrower"}
    expired_token = create_access_token(payload, expires_delta=timedelta(seconds=-10))
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)


def test_jwt_invalid_token():
    with pytest.raises(jwt.PyJWTError):
        decode_access_token("not-a-valid-jwt-token")


# ==============================================================================
# 2. Registration Tests (/api/auth/register)
# ==============================================================================

def test_register_borrower_success(client):
    res = client.post(
        "/api/auth/register",
        json={
            "full_name": "  Alice Borrower  ",
            "email": " Alice@Example.Com ",
            "password": "password123",
            "role": "borrower",
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["id"] is not None
    assert data["full_name"] == "Alice Borrower"
    assert data["email"] == "alice@example.com"
    assert data["role"] == "borrower"
    assert data["is_active"] is True
    assert "created_at" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_lender_success(client):
    res = client.post(
        "/api/auth/register",
        json={
            "full_name": "Bob Lender",
            "email": "bob@example.com",
            "password": "password123",
            "role": "lender",
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["role"] == "lender"


def test_register_admin_rejected(client):
    res = client.post(
        "/api/auth/register",
        json={
            "full_name": "Eve Admin",
            "email": "eve@example.com",
            "password": "password123",
            "role": "admin",
        },
    )
    assert res.status_code == 400
    assert "Admin registration is not allowed" in res.json()["detail"]


def test_register_duplicate_email(client):
    payload = {
        "full_name": "Alice One",
        "email": "alice@example.com",
        "password": "password123",
        "role": "borrower",
    }
    r1 = client.post("/api/auth/register", json=payload)
    assert r1.status_code == 201

    r2 = client.post("/api/auth/register", json=payload)
    assert r2.status_code == 409
    assert "Email already registered" in r2.json()["detail"]


def test_register_short_password(client):
    res = client.post(
        "/api/auth/register",
        json={
            "full_name": "Alice Short",
            "email": "short@example.com",
            "password": "short",
            "role": "borrower",
        },
    )
    assert res.status_code == 422


# ==============================================================================
# 3. Login Tests (/api/auth/login)
# ==============================================================================

def test_login_json_success(client):
    # Register first
    client.post(
        "/api/auth/register",
        json={
            "full_name": "Charlie Borrower",
            "email": "charlie@example.com",
            "password": "password123",
            "role": "borrower",
        },
    )

    # Login with JSON
    res = client.post(
        "/api/auth/login",
        json={"email": "charlie@example.com", "password": "password123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "charlie@example.com"
    assert data["user"]["role"] == "borrower"


def test_login_oauth2_form_success(client):
    client.post(
        "/api/auth/register",
        json={
            "full_name": "Dave Lender",
            "email": "dave@example.com",
            "password": "password123",
            "role": "lender",
        },
    )

    # Login using OAuth2 form-data
    res = client.post(
        "/api/auth/login",
        data={"username": "dave@example.com", "password": "password123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["user"]["email"] == "dave@example.com"


def test_login_invalid_password(client):
    client.post(
        "/api/auth/register",
        json={
            "full_name": "Eve User",
            "email": "eve@example.com",
            "password": "password123",
            "role": "borrower",
        },
    )
    res = client.post(
        "/api/auth/login",
        json={"email": "eve@example.com", "password": "wrongpassword"},
    )
    assert res.status_code == 401
    assert "Invalid email or password" in res.json()["detail"]


def test_login_nonexistent_user(client):
    res = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "password123"},
    )
    assert res.status_code == 401


# ==============================================================================
# 4. Protected /api/auth/me and Logout Tests
# ==============================================================================

def test_auth_me_authenticated(client):
    client.post(
        "/api/auth/register",
        json={
            "full_name": "Me User",
            "email": "me@example.com",
            "password": "password123",
            "role": "borrower",
        },
    )
    login_res = client.post(
        "/api/auth/login",
        json={"email": "me@example.com", "password": "password123"},
    )
    token = login_res.json()["access_token"]

    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "me@example.com"


def test_auth_me_unauthenticated(client):
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_auth_logout(client):
    res = client.post("/api/auth/logout")
    assert res.status_code == 200
    assert res.json()["message"] == "Logged out successfully"


# ==============================================================================
# 5. Role-Based Access Tests (/api/test/*)
# ==============================================================================

def test_role_based_access(client):
    # Setup borrower
    client.post(
        "/api/auth/register",
        json={
            "full_name": "Role Borrower",
            "email": "role_borrower@example.com",
            "password": "password123",
            "role": "borrower",
        },
    )
    b_token = client.post(
        "/api/auth/login",
        json={"email": "role_borrower@example.com", "password": "password123"},
    ).json()["access_token"]

    # Setup lender
    client.post(
        "/api/auth/register",
        json={
            "full_name": "Role Lender",
            "email": "role_lender@example.com",
            "password": "password123",
            "role": "lender",
        },
    )
    l_token = client.post(
        "/api/auth/login",
        json={"email": "role_lender@example.com", "password": "password123"},
    ).json()["access_token"]

    # Setup admin directly in DB
    db = TestingSessionLocal()
    admin_user = User(
        full_name="System Admin",
        email="admin@example.com",
        password_hash=hash_password("adminpass123"),
        role=UserRole.ADMIN,
        is_active=True,
    )
    db.add(admin_user)
    db.commit()
    db.close()

    a_token = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "adminpass123"},
    ).json()["access_token"]

    # Test /api/test/borrower
    assert client.get("/api/test/borrower", headers={"Authorization": f"Bearer {b_token}"}).status_code == 200
    assert client.get("/api/test/borrower", headers={"Authorization": f"Bearer {l_token}"}).status_code == 403
    assert client.get("/api/test/borrower", headers={"Authorization": f"Bearer {a_token}"}).status_code == 403
    assert client.get("/api/test/borrower").status_code == 401

    # Test /api/test/lender
    assert client.get("/api/test/lender", headers={"Authorization": f"Bearer {l_token}"}).status_code == 200
    assert client.get("/api/test/lender", headers={"Authorization": f"Bearer {b_token}"}).status_code == 403
    assert client.get("/api/test/lender", headers={"Authorization": f"Bearer {a_token}"}).status_code == 403

    # Test /api/test/admin
    assert client.get("/api/test/admin", headers={"Authorization": f"Bearer {a_token}"}).status_code == 200
    assert client.get("/api/test/admin", headers={"Authorization": f"Bearer {b_token}"}).status_code == 403
    assert client.get("/api/test/admin", headers={"Authorization": f"Bearer {l_token}"}).status_code == 403


# ==============================================================================
# 6. Admin User Management Tests (/api/admin/*)
# ==============================================================================

def test_admin_user_management(client):
    db = TestingSessionLocal()
    admin = User(
        full_name="Root Admin",
        email="root@example.com",
        password_hash=hash_password("adminpass123"),
        role=UserRole.ADMIN,
        is_active=True,
    )
    db.add(admin)
    db.commit()
    db.close()

    admin_token = client.post(
        "/api/auth/login",
        json={"email": "root@example.com", "password": "adminpass123"},
    ).json()["access_token"]

    # Register a borrower
    reg_res = client.post(
        "/api/auth/register",
        json={
            "full_name": "Target User",
            "email": "target@example.com",
            "password": "password123",
            "role": "borrower",
        },
    )
    target_id = reg_res.json()["id"]

    # Admin list users
    users_res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert users_res.status_code == 200
    assert len(users_res.json()) >= 2

    # Admin get user by id
    user_res = client.get(f"/api/admin/users/{target_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert user_res.status_code == 200
    assert user_res.json()["email"] == "target@example.com"

    # Admin deactivate user
    deact_res = client.patch(f"/api/admin/users/{target_id}/deactivate", headers={"Authorization": f"Bearer {admin_token}"})
    assert deact_res.status_code == 200
    assert deact_res.json()["is_active"] is False

    # Deactivated user cannot login
    login_attempt = client.post(
        "/api/auth/login",
        json={"email": "target@example.com", "password": "password123"},
    )
    assert login_attempt.status_code == 403
    assert "deactivated" in login_attempt.json()["detail"].lower()

    # Admin reactivates user
    act_res = client.patch(f"/api/admin/users/{target_id}/activate", headers={"Authorization": f"Bearer {admin_token}"})
    assert act_res.status_code == 200
    assert act_res.json()["is_active"] is True

    # Reactivated user can now login
    login_success = client.post(
        "/api/auth/login",
        json={"email": "target@example.com", "password": "password123"},
    )
    assert login_success.status_code == 200

    # Admin changes user role to lender
    role_res = client.patch(
        f"/api/admin/users/{target_id}/role",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"role": "lender"},
    )
    assert role_res.status_code == 200
    assert role_res.json()["role"] == "lender"

    # Verify admin cannot deactivate own account
    admin_id = client.get("/api/auth/me", headers={"Authorization": f"Bearer {admin_token}"}).json()["id"]
    self_deact = client.patch(f"/api/admin/users/{admin_id}/deactivate", headers={"Authorization": f"Bearer {admin_token}"})
    assert self_deact.status_code == 400


# ==============================================================================
# 7. Verification that existing endpoints remain intact
# ==============================================================================

def test_existing_endpoints_preserved(client):
    # Health check
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    # Loan evaluate
    eval_payload = {
        "borrower": {
            "loan_amount": 100000.0,
            "monthly_income": 80000.0,
            "monthly_expenses": 30000.0,
            "existing_emi": 5000.0,
            "max_emi": 25000.0,
            "max_interest_rate": 15.0,
            "preferred_tenure": 24,
            "max_tenure": 48,
            "collateral_required": False,
        },
        "lender": {
            "max_loan_amount": 200000.0,
            "min_interest_rate": 10.0,
            "max_tenure": 60,
            "min_expected_return": 1.1,
            "collateral_required": False,
        },
        "proposal": {
            "amount": 100000.0,
            "interest_rate": 12.0,
            "tenure_months": 24,
            "upfront_payment": 0.0,
        },
    }
    eval_res = client.post("/api/loan/evaluate", json=eval_payload)
    assert eval_res.status_code == 200
    assert "valid" in eval_res.json()
    assert eval_res.json()["valid"] is True

    # Negotiation start
    nego_res = client.post(
        "/api/negotiation/start",
        json={
            "borrower": eval_payload["borrower"],
            "lender": eval_payload["lender"],
        },
    )
    assert nego_res.status_code == 200
    assert "agreement_found" in nego_res.json()
