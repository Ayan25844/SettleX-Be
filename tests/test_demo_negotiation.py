"""
Tests for SettleX Demo Simulation Mode.

Verifies:
1. Demo mode configuration toggle via SETTLEX_DEMO_MODE.
2. Demo mode produces the same deterministic result every time (reproducibility).
3. Round 1 borrower counter changes the proposal (11.80%).
4. Lender sees the changed proposal and counters (11.95%).
5. Round 2 reaches agreement at 11.95% · 42 months.
6. Verifier dynamically calculates EMI, total repayment, total interest, and validity.
7. Database persistence for session and offers in demo mode.
8. API response contract with demo_mode=True flag.
9. Live mode remains untouched when SETTLEX_DEMO_MODE=false.
"""

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app
from models.match import Match, MatchStatus
from models.negotiation import AgentType, NegotiationOffer, NegotiationSession, SessionStatus
from models.profile import BorrowerProfile, LenderProfile
from models.user import User, UserRole
from services.auth import hash_password
from services.demo_negotiation import is_demo_mode, run_demo_negotiation_session
from services.financial import calculate_emi, calculate_total_repayment
from services.negotiation_graph import run_negotiation_session

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
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def create_sample_entities(db):
    b_user = User(
        email="demo_borrower@settlex.local",
        password_hash=hash_password("password123"),
        full_name="Demo Borrower",
        role=UserRole.BORROWER,
        is_active=True,
    )
    l_user = User(
        email="demo_lender@settlex.local",
        password_hash=hash_password("password123"),
        full_name="Demo Lender",
        role=UserRole.LENDER,
        is_active=True,
    )
    db.add_all([b_user, l_user])
    db.commit()

    b_prof = BorrowerProfile(
        user_id=b_user.id,
        loan_amount=500000.0,
        monthly_income=100000.0,
        monthly_expenses=30000.0,
        existing_emi=5000.0,
        max_emi=35000.0,
        max_interest_rate=14.0,
        preferred_tenure=42,
        max_tenure=60,
        collateral_required=False,
    )
    l_prof = LenderProfile(
        user_id=l_user.id,
        max_loan_amount=1000000.0,
        min_interest_rate=10.0,
        max_tenure=60,
        min_expected_return=5.0,
        collateral_required=False,
        available_capacity=1000000.0,
    )
    db.add_all([b_prof, l_prof])
    db.commit()

    match = Match(
        borrower_id=b_prof.id,
        lender_id=l_prof.id,
        match_score=90.0,
        score_breakdown={"rate_score": 90.0},
        status=MatchStatus.ACCEPTED.value,
    )
    db.add(match)
    db.commit()

    return b_user, l_user, b_prof, l_prof, match


def get_auth_token(client, email, password="password123"):
    res = client.post("/api/auth/login", json={"email": email, "password": password})
    return res.json()["access_token"]


def test_demo_mode_configuration_toggle():
    """Verify is_demo_mode reads SETTLEX_DEMO_MODE correctly."""
    with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "true"}):
        assert is_demo_mode() is True

    with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "1"}):
        assert is_demo_mode() is True

    with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "YES"}):
        assert is_demo_mode() is True

    with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "false"}):
        assert is_demo_mode() is False

    with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "0"}):
        assert is_demo_mode() is False

    with patch.dict(os.environ, {}, clear=True):
        assert is_demo_mode() is False


def test_demo_mode_deterministic_execution_and_reproducibility():
    """Verify demo mode produces the exact same deterministic result every time."""
    db = TestingSessionLocal()
    try:
        b_user, l_user, b_prof, l_prof, match = create_sample_entities(db)

        session1 = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
        )
        db.add(session1)
        db.commit()
        db.refresh(session1)

        res1 = run_demo_negotiation_session(session1.id, db)

        # Create second session and verify identical results
        session2 = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
        )
        db.add(session2)
        db.commit()
        db.refresh(session2)

        res2 = run_demo_negotiation_session(session2.id, db)

        assert res1["status"] == res2["status"] == SessionStatus.AGREEMENT_REACHED.value
        assert res1["agreement_found"] == res2["agreement_found"] is True
        assert res1["round_number"] == res2["round_number"] == 2
        assert res1["final_proposal"] == res2["final_proposal"]
        assert res1["final_proposal"]["interest_rate"] == 11.95
        assert res1["final_proposal"]["tenure_months"] == 42
        assert res1["final_proposal"]["amount"] == 500000.0
        assert len(res1["negotiation_history"]) == len(res2["negotiation_history"]) == 4

    finally:
        db.close()


def test_demo_mode_multi_round_timeline_and_counter_propagation():
    """
    Verify the exact demo flow:
    - Round 1 Borrower: counter to 11.80%, 42m
    - Round 1 Lender: counter to 11.95%, 42m
    - Round 2 Borrower: accept 11.95%, 42m
    - Round 2 Lender: accept 11.95%, 42m
    - Final Agreement: 11.95% · 42m
    """
    db = TestingSessionLocal()
    try:
        b_user, l_user, b_prof, l_prof, match = create_sample_entities(db)

        session = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        res = run_demo_negotiation_session(session.id, db)

        history = res["negotiation_history"]
        assert len(history) == 4

        # Round 1 Borrower Counter
        b1 = history[0]
        assert b1["round"] == 1
        assert b1["agent"] == AgentType.BORROWER.value
        assert b1["action"] == "counter"
        assert b1["offer"]["interest_rate"] == 11.80
        assert b1["offer"]["tenure_months"] == 42
        assert b1["verification"]["valid"] is True

        # Round 1 Lender Counter
        l1 = history[1]
        assert l1["round"] == 1
        assert l1["agent"] == AgentType.LENDER.value
        assert l1["action"] == "counter"
        assert l1["offer"]["interest_rate"] == 11.95
        assert l1["offer"]["tenure_months"] == 42
        assert l1["verification"]["valid"] is True

        # Round 2 Borrower Accept
        b2 = history[2]
        assert b2["round"] == 2
        assert b2["agent"] == AgentType.BORROWER.value
        assert b2["action"] == "accept"
        assert b2["offer"]["interest_rate"] == 11.95
        assert b2["offer"]["tenure_months"] == 42

        # Round 2 Lender Accept
        l2 = history[3]
        assert l2["round"] == 2
        assert l2["agent"] == AgentType.LENDER.value
        assert l2["action"] == "accept"
        assert l2["offer"]["interest_rate"] == 11.95
        assert l2["offer"]["tenure_months"] == 42

    finally:
        db.close()


def test_demo_mode_dynamic_verifier_calculations():
    """
    Verify EMI, total repayment, total interest, and validity are computed
    dynamically by the financial verifier rather than hardcoded.
    """
    db = TestingSessionLocal()
    try:
        b_user, l_user, b_prof, l_prof, match = create_sample_entities(db)

        session = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        res = run_demo_negotiation_session(session.id, db)
        v = res["verification_result"]

        # Expected calculation for ₹500,000 at 11.95% for 42 months
        expected_emi = calculate_emi(500000.0, 11.95, 42)
        expected_total = calculate_total_repayment(expected_emi, 42)
        expected_interest = round(expected_total - 500000.0, 2)

        assert v["valid"] is True
        assert v["emi"] == expected_emi
        assert v["total_repayment"] == expected_total
        assert v["total_interest"] == expected_interest
        assert v["borrower_utility"] > 0
        assert v["lender_utility"] > 0

    finally:
        db.close()


def test_demo_mode_api_endpoint_integration(client: TestClient):
    """
    Test full API integration when SETTLEX_DEMO_MODE=true:
    - POST /api/negotiations/{session_id}/start triggers demo mode
    - Returns demo_mode=True and agreement_reached status
    - Database records are persisted and retrievable via GET
    """
    db = TestingSessionLocal()
    try:
        b_user, l_user, b_prof, l_prof, match = create_sample_entities(db)

        session = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id
        token = get_auth_token(client, b_user.email)
    finally:
        db.close()

    with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "true"}):
        res = client.post(
            f"/api/negotiations/{session_id}/start",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["demo_mode"] is True
        assert data["status"] == SessionStatus.AGREEMENT_REACHED.value
        assert data["agreement_found"] is True
        assert data["round_number"] == 2
        assert data["final_proposal"]["interest_rate"] == 11.95
        assert data["final_proposal"]["tenure_months"] == 42
        assert data["final_proposal"]["amount"] == 500000.0
        assert len(data["history"]) == 4

        # Verify GET returns same state with demo_mode=True
        get_res = client.get(
            f"/api/negotiations/{session_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["demo_mode"] is True
        assert get_data["agreement_found"] is True
        assert get_data["final_proposal"]["interest_rate"] == 11.95

        # Check DB records
        check_db = TestingSessionLocal()
        try:
            persisted_offers = check_db.query(NegotiationOffer).filter_by(session_id=session_id).order_by(NegotiationOffer.id.asc()).all()
            assert len(persisted_offers) == 4
            assert persisted_offers[0].position == "counter"
            assert persisted_offers[0].interest_rate == 11.80
            assert persisted_offers[1].position == "counter"
            assert persisted_offers[1].interest_rate == 11.95
            assert persisted_offers[2].position == "accept"
            assert persisted_offers[2].interest_rate == 11.95
            assert persisted_offers[3].position == "accept"
            assert persisted_offers[3].interest_rate == 11.95
        finally:
            check_db.close()


def test_live_mode_remains_available():
    """
    Verify that when SETTLEX_DEMO_MODE=false, run_negotiation_session
    routes to the live LangGraph execution path rather than demo simulation.
    """
    db = TestingSessionLocal()
    try:
        b_user, l_user, b_prof, l_prof, match = create_sample_entities(db)

        session = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Mock agents for live LangGraph path
        with patch.dict(os.environ, {"SETTLEX_DEMO_MODE": "false"}):
            with patch("services.negotiation_graph.borrower_agent", return_value={"position": "accept", "reason": "Live agree"}), \
                 patch("services.negotiation_graph.lender_agent", return_value={"position": "accept", "reason": "Live agree"}):

                res = run_negotiation_session(session.id, db)
                # Live mode executes LangGraph initialize node and reaches agreement in round 1
                assert res.get("demo_mode") is not True
                assert res["status"] == SessionStatus.AGREEMENT_REACHED.value
                assert res["agreement_found"] is True
                assert res["round_number"] == 1  # Live path finishes in Round 1 when both accept immediately
    finally:
        db.close()
