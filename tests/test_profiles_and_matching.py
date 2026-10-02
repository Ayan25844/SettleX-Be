import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app
from models.match import Match, MatchStatus
from models.profile import BorrowerProfile, LenderProfile
from models.user import User, UserRole
from services.auth import hash_password
from services.matching import (
    calculate_match_score,
    evaluate_hard_filters,
    find_and_rank_matches,
)

# In-memory SQLite for test execution
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


def register_and_login(client, name: str, email: str, role: str) -> str:
    """Helper to register and login a user, returning their access token."""
    client.post(
        "/api/auth/register",
        json={
            "full_name": name,
            "email": email,
            "password": "password123",
            "role": role,
        },
    )
    res = client.post(
        "/api/auth/login",
        json={"email": email, "password": "password123"},
    )
    return res.json()["access_token"]


# ==============================================================================
# 1. Profile Tests (Borrower & Lender)
# ==============================================================================

def test_borrower_profile_crud(client):
    b_token = register_and_login(client, "Bob Borrower", "bob.borrower@example.com", "borrower")
    headers = {"Authorization": f"Bearer {b_token}"}

    # 1. Initially, profile should not exist
    get_res = client.get("/api/borrower/profile", headers=headers)
    assert get_res.status_code == 404

    # 2. Create profile
    profile_data = {
        "loan_amount": 100000.0,
        "monthly_income": 75000.0,
        "monthly_expenses": 25000.0,
        "existing_emi": 5000.0,
        "max_emi": 20000.0,
        "max_interest_rate": 14.5,
        "preferred_tenure": 24,
        "max_tenure": 48,
        "collateral_required": False,
    }
    create_res = client.post("/api/borrower/profile", headers=headers, json=profile_data)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["loan_amount"] == 100000.0
    assert created["max_interest_rate"] == 14.5
    assert created["preferred_tenure"] == 24

    # 3. Duplicate creation rejected
    dup_res = client.post("/api/borrower/profile", headers=headers, json=profile_data)
    assert dup_res.status_code == 400
    assert "already exists" in dup_res.json()["detail"]

    # 4. Get profile
    read_res = client.get("/api/borrower/profile", headers=headers)
    assert read_res.status_code == 200
    assert read_res.json()["loan_amount"] == 100000.0

    # 5. Update profile
    update_res = client.put(
        "/api/borrower/profile",
        headers=headers,
        json={"loan_amount": 120000.0, "max_interest_rate": 15.0},
    )
    assert update_res.status_code == 200
    assert update_res.json()["loan_amount"] == 120000.0
    assert update_res.json()["max_interest_rate"] == 15.0


def test_borrower_profile_invalid_tenure(client):
    b_token = register_and_login(client, "Tenure Borrower", "tenure.b@example.com", "borrower")
    headers = {"Authorization": f"Bearer {b_token}"}

    # max_tenure < preferred_tenure should be rejected
    invalid_data = {
        "loan_amount": 50000.0,
        "monthly_income": 40000.0,
        "monthly_expenses": 10000.0,
        "existing_emi": 0.0,
        "max_emi": 15000.0,
        "max_interest_rate": 12.0,
        "preferred_tenure": 36,
        "max_tenure": 24,  # invalid
        "collateral_required": False,
    }
    res = client.post("/api/borrower/profile", headers=headers, json=invalid_data)
    assert res.status_code == 422


def test_lender_profile_crud(client):
    l_token = register_and_login(client, "Leo Lender", "leo.lender@example.com", "lender")
    headers = {"Authorization": f"Bearer {l_token}"}

    # 1. Create lender profile
    lender_data = {
        "max_loan_amount": 500000.0,
        "min_interest_rate": 10.5,
        "max_tenure": 60,
        "min_expected_return": 1.15,
        "collateral_required": False,
        "available_capacity": 500000.0,
    }
    create_res = client.post("/api/lender/profile", headers=headers, json=lender_data)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["max_loan_amount"] == 500000.0
    assert created["available_capacity"] == 500000.0

    # 2. Duplicate profile rejected
    dup_res = client.post("/api/lender/profile", headers=headers, json=lender_data)
    assert dup_res.status_code == 400

    # 3. Get profile
    get_res = client.get("/api/lender/profile", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["min_interest_rate"] == 10.5

    # 4. Update profile
    up_res = client.put(
        "/api/lender/profile",
        headers=headers,
        json={"available_capacity": 450000.0, "min_interest_rate": 9.5},
    )
    assert up_res.status_code == 200
    assert up_res.json()["available_capacity"] == 450000.0
    assert up_res.json()["min_interest_rate"] == 9.5


def test_profile_role_isolation(client):
    b_token = register_and_login(client, "User B", "user.b@example.com", "borrower")
    l_token = register_and_login(client, "User L", "user.l@example.com", "lender")

    b_headers = {"Authorization": f"Bearer {b_token}"}
    l_headers = {"Authorization": f"Bearer {l_token}"}

    # Borrower cannot access lender endpoints
    assert client.get("/api/lender/profile", headers=b_headers).status_code == 403
    assert client.post("/api/lender/profile", headers=b_headers, json={}).status_code == 403

    # Lender cannot access borrower endpoints
    assert client.get("/api/borrower/profile", headers=l_headers).status_code == 403
    assert client.post("/api/borrower/profile", headers=l_headers, json={}).status_code == 403


# ==============================================================================
# 2. Matching Engine Unit Tests (Hard filters & Scoring)
# ==============================================================================

class MockBorrower:
    def __init__(self, amount=100000, max_rate=15.0, pref_t=24, max_t=48, collat=False):
        self.loan_amount = amount
        self.max_interest_rate = max_rate
        self.preferred_tenure = pref_t
        self.max_tenure = max_t
        self.collateral_required = collat


class MockLender:
    def __init__(self, max_amount=200000, min_rate=10.0, max_t=60, capacity=200000, collat=False):
        self.id = 1
        self.max_loan_amount = max_amount
        self.min_interest_rate = min_rate
        self.max_tenure = max_t
        self.available_capacity = capacity
        self.collateral_required = collat


def test_hard_filters_loan_amount():
    b = MockBorrower(amount=300000)
    l = MockLender(max_amount=200000)
    ok, reason = evaluate_hard_filters(b, l)
    assert ok is False
    assert "exceeds lender max ticket size" in reason


def test_hard_filters_interest_rate():
    b = MockBorrower(max_rate=11.0)
    l = MockLender(min_rate=12.0)
    ok, reason = evaluate_hard_filters(b, l)
    assert ok is False
    assert "exceeds borrower maximum rate" in reason


def test_hard_filters_tenure():
    b = MockBorrower(pref_t=48)
    l = MockLender(max_t=36)
    ok, reason = evaluate_hard_filters(b, l)
    assert ok is False
    assert "exceeds lender maximum tenure" in reason


def test_hard_filters_capacity():
    b = MockBorrower(amount=100000)
    l = MockLender(max_amount=200000, capacity=50000)
    ok, reason = evaluate_hard_filters(b, l)
    assert ok is False
    assert "exceeds lender available capacity" in reason


def test_hard_filters_collateral_mismatch():
    b = MockBorrower(collat=False)
    l = MockLender(collat=True)
    ok, reason = evaluate_hard_filters(b, l)
    assert ok is False
    assert "mandates collateral" in reason


def test_hard_filters_compatible():
    b = MockBorrower(amount=100000, max_rate=15.0, pref_t=24, collat=False)
    l = MockLender(max_amount=200000, min_rate=10.0, max_t=60, capacity=200000, collat=False)
    ok, reason = evaluate_hard_filters(b, l)
    assert ok is True
    assert reason is None


def test_scoring_components_and_ranking():
    b = MockBorrower(amount=100000, max_rate=15.0, pref_t=24, max_t=48, collat=False)

    l1 = MockLender(max_amount=150000, min_rate=10.0, max_t=60, capacity=150000, collat=False)
    l1.id = 1
    l2 = MockLender(max_amount=500000, min_rate=14.0, max_t=36, capacity=105000, collat=False)
    l2.id = 2

    ranked = find_and_rank_matches(b, [l1, l2])
    assert len(ranked) == 2
    # L1 should have higher score than L2 because lower min interest rate, higher tenure accommodation, better ticket size
    assert ranked[0]["lender"].id == 1
    assert ranked[0]["match_score"] > ranked[1]["match_score"]

    score, breakdown = calculate_match_score(b, l1)
    assert 0.0 <= score <= 1.0
    for comp in ["interest", "loan_amount", "tenure", "collateral", "capacity"]:
        assert 0.0 <= breakdown[comp] <= 1.0


# ==============================================================================
# 3. Matching Endpoints & Capacity Management Integration Tests
# ==============================================================================

def test_matching_flow_find_accept_reject(client):
    # Setup Borrower
    b_token = register_and_login(client, "Flow Borrower", "flow.b@example.com", "borrower")
    b_headers = {"Authorization": f"Bearer {b_token}"}
    client.post(
        "/api/borrower/profile",
        headers=b_headers,
        json={
            "loan_amount": 100000.0,
            "monthly_income": 80000.0,
            "monthly_expenses": 20000.0,
            "existing_emi": 0.0,
            "max_emi": 30000.0,
            "max_interest_rate": 15.0,
            "preferred_tenure": 24,
            "max_tenure": 48,
            "collateral_required": False,
        },
    )

    # Setup Lender 1 (Compatible)
    l1_token = register_and_login(client, "Lender One", "lender1@example.com", "lender")
    l1_headers = {"Authorization": f"Bearer {l1_token}"}
    client.post(
        "/api/lender/profile",
        headers=l1_headers,
        json={
            "max_loan_amount": 200000.0,
            "min_interest_rate": 10.0,
            "max_tenure": 60,
            "min_expected_return": 1.1,
            "collateral_required": False,
            "available_capacity": 200000.0,
        },
    )

    # Setup Lender 2 (Incompatible due to interest rate)
    l2_token = register_and_login(client, "Lender Two", "lender2@example.com", "lender")
    l2_headers = {"Authorization": f"Bearer {l2_token}"}
    client.post(
        "/api/lender/profile",
        headers=l2_headers,
        json={
            "max_loan_amount": 200000.0,
            "min_interest_rate": 18.0,  # exceeds borrower max 15.0
            "max_tenure": 60,
            "min_expected_return": 1.1,
            "collateral_required": False,
            "available_capacity": 200000.0,
        },
    )

    # Setup Lender 3 (Incompatible due to insufficient capacity)
    l3_token = register_and_login(client, "Lender Three", "lender3@example.com", "lender")
    l3_headers = {"Authorization": f"Bearer {l3_token}"}
    client.post(
        "/api/lender/profile",
        headers=l3_headers,
        json={
            "max_loan_amount": 200000.0,
            "min_interest_rate": 10.0,
            "max_tenure": 60,
            "min_expected_return": 1.1,
            "collateral_required": False,
            "available_capacity": 50000.0,  # less than loan_amount 100000
        },
    )

    # 1. Borrower searches for matches
    find_res = client.post("/api/matching/find", headers=b_headers)
    assert find_res.status_code == 200
    data = find_res.json()
    matches = data["matches"]
    # Only Lender 1 is compatible!
    assert len(matches) == 1
    match_item = matches[0]
    match_id = match_item["match_id"]
    assert match_item["status"] == "pending"
    assert match_item["score_breakdown"]["interest"] > 0

    # 2. View my-matches
    my_matches = client.get("/api/matching/my-matches", headers=b_headers).json()
    assert len(my_matches) == 1
    assert my_matches[0]["id"] == match_id

    # 3. Accept match -> Capacity deduction
    accept_res = client.post(f"/api/matching/{match_id}/accept", headers=b_headers)
    assert accept_res.status_code == 200
    assert accept_res.json()["status"] == "accepted"

    # Verify Lender 1 capacity decreased from 200000 to 100000
    l1_prof = client.get("/api/lender/profile", headers=l1_headers).json()
    assert l1_prof["available_capacity"] == 100000.0

    # 4. Cannot accept an already accepted match
    dup_accept = client.post(f"/api/matching/{match_id}/accept", headers=b_headers)
    assert dup_accept.status_code == 400

    # 5. Reject match -> Restores capacity
    reject_res = client.post(f"/api/matching/{match_id}/reject", headers=b_headers)
    assert reject_res.status_code == 200
    assert reject_res.json()["status"] == "rejected"

    # Verify Lender 1 capacity restored back to 200000
    l1_prof_restored = client.get("/api/lender/profile", headers=l1_headers).json()
    assert l1_prof_restored["available_capacity"] == 200000.0


def test_capacity_safety_rejection(client):
    # Setup Borrower
    b_token = register_and_login(client, "Cap Borrower", "cap.b@example.com", "borrower")
    b_headers = {"Authorization": f"Bearer {b_token}"}
    client.post(
        "/api/borrower/profile",
        headers=b_headers,
        json={
            "loan_amount": 100000.0,
            "monthly_income": 80000.0,
            "monthly_expenses": 20000.0,
            "existing_emi": 0.0,
            "max_emi": 30000.0,
            "max_interest_rate": 15.0,
            "preferred_tenure": 24,
            "max_tenure": 48,
            "collateral_required": False,
        },
    )

    # Setup Lender with capacity = 100000
    l_token = register_and_login(client, "Cap Lender", "cap.l@example.com", "lender")
    l_headers = {"Authorization": f"Bearer {l_token}"}
    client.post(
        "/api/lender/profile",
        headers=l_headers,
        json={
            "max_loan_amount": 100000.0,
            "min_interest_rate": 10.0,
            "max_tenure": 48,
            "min_expected_return": 1.1,
            "collateral_required": False,
            "available_capacity": 100000.0,
        },
    )

    # Find match
    find_res = client.post("/api/matching/find", headers=b_headers)
    match_id = find_res.json()["matches"][0]["match_id"]

    # Manually drain lender capacity via PUT to simulate concurrent drawdown
    client.put("/api/lender/profile", headers=l_headers, json={"available_capacity": 40000.0})

    # Attempt to accept match should fail due to insufficient capacity
    accept_res = client.post(f"/api/matching/{match_id}/accept", headers=b_headers)
    assert accept_res.status_code == 400
    assert "insufficient" in accept_res.json()["detail"].lower()


def test_admin_inspect_matches(client):
    # Setup Admin in DB
    db = TestingSessionLocal()
    admin = User(
        full_name="Match Admin",
        email="match.admin@example.com",
        password_hash=hash_password("adminpassword123"),
        role=UserRole.ADMIN,
        is_active=True,
    )
    db.add(admin)
    db.commit()
    db.close()

    admin_token = client.post(
        "/api/auth/login",
        json={"email": "match.admin@example.com", "password": "adminpassword123"},
    ).json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Inspect matches endpoint
    res = client.get("/api/admin/matches", headers=admin_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_no_matches_found_response(client):
    b_token = register_and_login(client, "Lonely Borrower", "lonely.b@example.com", "borrower")
    b_headers = {"Authorization": f"Bearer {b_token}"}
    client.post(
        "/api/borrower/profile",
        headers=b_headers,
        json={
            "loan_amount": 5000000.0,  # huge amount
            "monthly_income": 500000.0,
            "monthly_expenses": 50000.0,
            "existing_emi": 0.0,
            "max_emi": 150000.0,
            "max_interest_rate": 5.0,   # tiny rate
            "preferred_tenure": 24,
            "max_tenure": 48,
            "collateral_required": False,
        },
    )

    # Search for matches when no lenders exist for these criteria
    find_res = client.post("/api/matching/find", headers=b_headers)
    assert find_res.status_code == 200
    data = find_res.json()
    assert data["matches"] == []
    assert "No compatible lenders" in data["message"]
