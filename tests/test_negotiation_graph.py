import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app
from models.match import Match, MatchStatus
from models.negotiation import NegotiationOffer, NegotiationSession, SessionStatus, AgentType
from models.profile import BorrowerProfile, LenderProfile
from models.user import User, UserRole
from services.auth import hash_password
from services.negotiation_graph import (
    negotiation_graph,
    run_negotiation_session,
    NegotiationState,
)


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


def register_and_login(client: TestClient, name: str, email: str, role: str) -> str:
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
        data={"username": email, "password": "password123"},
    )
    return res.json()["access_token"]


def create_sample_entities(db):
    borrower_user = User(
        email="borrower_test@settlex.local",
        password_hash=hash_password("password123"),
        full_name="Borrower Test",
        role=UserRole.BORROWER,
        is_active=True,
    )
    lender_user = User(
        email="lender_test@settlex.local",
        password_hash=hash_password("password123"),
        full_name="Lender Test",
        role=UserRole.LENDER,
        is_active=True,
    )
    other_user = User(
        email="other_test@settlex.local",
        password_hash=hash_password("password123"),
        full_name="Other User",
        role=UserRole.BORROWER,
        is_active=True,
    )
    db.add_all([borrower_user, lender_user, other_user])
    db.commit()

    borrower_profile = BorrowerProfile(
        user_id=borrower_user.id,
        loan_amount=500000.0,
        monthly_income=100000.0,
        monthly_expenses=30000.0,
        existing_emi=5000.0,
        max_emi=35000.0,
        max_interest_rate=14.0,
        preferred_tenure=36,
        max_tenure=48,
        collateral_required=False,
    )
    lender_profile = LenderProfile(
        user_id=lender_user.id,
        max_loan_amount=1000000.0,
        min_interest_rate=10.0,
        max_tenure=60,
        min_expected_return=8.0,
        collateral_required=False,
        available_capacity=1000000.0,
    )
    db.add_all([borrower_profile, lender_profile])
    db.commit()

    match = Match(
        borrower_id=borrower_profile.id,
        lender_id=lender_profile.id,
        match_score=85.0,
        score_breakdown={"rate_score": 85.0},
        status=MatchStatus.ACCEPTED.value,
    )
    db.add(match)
    db.commit()

    return borrower_user, lender_user, other_user, borrower_profile, lender_profile, match


# ===========================================================================
# 1. Targeted LangGraph Engine Unit Tests (Cases 1 - 7)
# ===========================================================================

def test_case_1_agreement_both_accept():
    """
    Case 1 — Agreement:
    Borrower accepts
    Lender independently accepts
    Verifier passes
    → agreement_reached
    """
    with patch("services.negotiation_graph.borrower_agent", return_value={"position": "accept", "reason": "Offer satisfies borrower"}), \
         patch("services.negotiation_graph.lender_agent", return_value={"position": "accept", "reason": "Offer satisfies lender"}):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "current_offer": {
                "amount": 500000.0,
                "interest_rate": 12.0,
                "tenure_months": 36,
                "upfront_payment": 0.0,
            },
            "max_rounds": 6,
        }

        result = negotiation_graph.invoke(initial_state)

        assert result["agreement_found"] is True
        assert result["status"] == SessionStatus.AGREEMENT_REACHED.value
        assert result["borrower_position"] == "accept"
        assert result["lender_position"] == "accept"
        assert result["verification_result"]["valid"] is True
        assert result["final_proposal"]["interest_rate"] == 12.0


def test_case_2_borrower_accepts_lender_rejects():
    """
    Case 2 — Borrower accepts but lender rejects:
    Borrower accepts
    Lender rejects
    → no agreement / rejected
    """
    with patch("services.negotiation_graph.borrower_agent", return_value={"position": "accept", "reason": "Borrower likes rate"}), \
         patch("services.negotiation_graph.lender_agent", return_value={"position": "reject", "reason": "Lender risk threshold exceeded"}):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "current_offer": {
                "amount": 500000.0,
                "interest_rate": 10.5,
                "tenure_months": 36,
                "upfront_payment": 0.0,
            },
            "max_rounds": 6,
        }

        result = negotiation_graph.invoke(initial_state)

        assert result["agreement_found"] is False
        assert result["status"] == SessionStatus.REJECTED.value
        assert result["decision"] == "reject"
        assert result["borrower_position"] == "accept"
        assert result["lender_position"] == "reject"


def test_case_3_borrower_accepts_lender_counters():
    """
    Case 3 — Borrower accepts but lender counters:
    Borrower accepts
    Lender counters
    → negotiation continues (does not finalize prematurely into agreement)
    """
    b_calls = [0]
    l_calls = [0]

    # In Round 1: Borrower accepts initial proposal, but Lender independently counters.
    # In Round 2: Borrower accepts Lender's counter, and Lender independently accepts.
    def mock_b(prof, current_offer=None):
        b_calls[0] += 1
        if b_calls[0] == 1:
            return {"position": "accept", "reason": "Round 1 accept"}
        return {"position": "accept", "reason": "Round 2 accept"}

    def mock_l(prof, current_offer=None):
        l_calls[0] += 1
        if l_calls[0] == 1:
            return {"position": "counter", "reason": "Lender wants 12.5%", "target_interest_rate": 12.5, "target_tenure_months": 36}
        return {"position": "accept", "reason": "Lender agrees to 12.5%"}

    with patch("services.negotiation_graph.borrower_agent", side_effect=mock_b), \
         patch("services.negotiation_graph.lender_agent", side_effect=mock_l):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "max_rounds": 6,
        }

        result = negotiation_graph.invoke(initial_state)

        # Confirm negotiation progressed beyond Round 1
        assert b_calls[0] == 2
        assert l_calls[0] == 2
        assert result["round_number"] == 2
        assert result["agreement_found"] is True
        assert result["status"] == SessionStatus.AGREEMENT_REACHED.value
        assert result["final_proposal"]["interest_rate"] == 12.5


def test_case_4_lender_accepts_borrower_counters():
    """
    Case 4 — Lender accepts but borrower counters:
    Borrower counters
    Lender accepts
    → negotiation continues (does not finalize prematurely into agreement)
    """
    b_calls = [0]
    l_calls = [0]

    # In Round 1: Borrower counters at 11%, Lender accepts. Since Borrower countered, negotiation continues!
    # In Round 2: Borrower confirms acceptance, Lender confirms acceptance.
    def mock_b(prof, current_offer=None):
        b_calls[0] += 1
        if b_calls[0] == 1:
            return {"position": "counter", "reason": "Borrower wants 11%", "target_interest_rate": 11.0, "target_tenure_months": 36}
        return {"position": "accept", "reason": "Borrower confirms 11%"}

    def mock_l(prof, current_offer=None):
        l_calls[0] += 1
        return {"position": "accept", "reason": "Lender accepts 11%"}

    with patch("services.negotiation_graph.borrower_agent", side_effect=mock_b), \
         patch("services.negotiation_graph.lender_agent", side_effect=mock_l):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "max_rounds": 6,
        }

        result = negotiation_graph.invoke(initial_state)

        # Confirm that Round 1 did not prematurely conclude: it looped to Round 2
        assert b_calls[0] == 2
        assert l_calls[0] == 2
        assert result["round_number"] == 2
        assert result["agreement_found"] is True
        assert result["final_proposal"]["interest_rate"] == 11.0


def test_case_5_invalid_final_offer():
    """
    Case 5 — Invalid final offer:
    Both agents accept
    Verifier rejects
    → no agreement / rejected
    """
    # 25% interest rate violates borrower's max rate of 14%
    with patch("services.negotiation_graph.borrower_agent", return_value={"position": "accept", "target_interest_rate": 25.0, "target_tenure_months": 36}), \
         patch("services.negotiation_graph.lender_agent", return_value={"position": "accept", "target_interest_rate": 25.0, "target_tenure_months": 36}):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "current_offer": {
                "amount": 500000.0,
                "interest_rate": 25.0,
                "tenure_months": 36,
                "upfront_payment": 0.0,
            },
            "max_rounds": 6,
        }

        result = negotiation_graph.invoke(initial_state)

        assert result["agreement_found"] is False
        assert result["status"] == SessionStatus.REJECTED.value
        assert result["verification_result"]["valid"] is False
        assert "Interest rate exceeds borrower's maximum." in result["verification_result"]["violations"]


def test_case_6_round_limit():
    """
    Case 6 — Round limit:
    Negotiation terminates at configured maximum rounds.
    """
    with patch("services.negotiation_graph.borrower_agent", return_value={"position": "counter", "target_interest_rate": 10.0, "target_tenure_months": 36}), \
         patch("services.negotiation_graph.lender_agent", return_value={"position": "counter", "target_interest_rate": 13.0, "target_tenure_months": 36}):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "max_rounds": 3,
        }

        result = negotiation_graph.invoke(initial_state)

        assert result["agreement_found"] is False
        assert result["status"] == SessionStatus.NO_AGREEMENT.value
        assert result["round_number"] == 3


def test_case_7_llm_failure_safe_fallback():
    """
    Case 7 — LLM failure:
    Fallback response does not create an invalid agreement.
    Deterministic counter is triggered and verified.
    """
    def broken_llm(*args, **kwargs):
        raise RuntimeError("OpenAI rate limit or network timeout")

    with patch("services.negotiation_graph.borrower_agent", side_effect=broken_llm), \
         patch("services.negotiation_graph.lender_agent", side_effect=broken_llm):

        initial_state: NegotiationState = {
            "session_id": 1,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "max_emi": 35000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 36,
                "max_tenure": 48,
            },
            "lender_profile": {
                "max_loan_amount": 1000000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
            },
            "max_rounds": 2,
        }

        result = negotiation_graph.invoke(initial_state)

        # Even with failed LLMs, no unhandled crash occurs and no accidental agreement is made
        assert result["agreement_found"] is False
        assert result["status"] == SessionStatus.NO_AGREEMENT.value
        # Negotiation history holds fallback counters with verified proposals
        assert len(result["negotiation_history"]) >= 2
        for evt in result["negotiation_history"]:
            assert evt["action"] == "counter"
            assert "verification" in evt
            assert evt["verification"]["valid"] is True


# ===========================================================================
# 2. Section 12 Deterministic End-to-End Scenario Test
# ===========================================================================

def test_section_12_deterministic_end_to_end_scenario():
    """
    Section 12 Scenario:
    Borrower:
      loan = ₹5,00,000
      max interest = 14%
      preferred tenure = 42
      max tenure = 60
      max EMI = ₹16,000

    Lender:
      max loan = ₹7,50,000
      min interest = 10%
      max tenure = 60
      capacity >= ₹5,00,000

    Trace:
      Round 1: Borrower counters at 10.5% -> Lender counters at 13.0%
      Round 2: Borrower counters at 11.5% -> Lender counters at 12.0%
      Round 3: Borrower accepts 12.0% -> Lender independently accepts 12.0%
      Verifier passes -> Agreement reached
    """
    borrower_responses = [
        {"position": "counter", "reason": "Requesting 10.5% starting rate", "target_interest_rate": 10.5, "target_tenure_months": 42},
        {"position": "counter", "reason": "Willing to counter at 11.5%", "target_interest_rate": 11.5, "target_tenure_months": 42},
        {"position": "accept", "reason": "12.0% is acceptable", "target_interest_rate": 12.0, "target_tenure_months": 42},
    ]
    lender_responses = [
        {"position": "counter", "reason": "Need 13.0% for risk spread", "target_interest_rate": 13.0, "target_tenure_months": 42},
        {"position": "counter", "reason": "Can lower to 12.0%", "target_interest_rate": 12.0, "target_tenure_months": 42},
        {"position": "accept", "reason": "Lender independently agrees to 12.0%", "target_interest_rate": 12.0, "target_tenure_months": 42},
    ]

    b_idx = [0]
    l_idx = [0]

    def mock_b(prof, current_offer=None):
        r = borrower_responses[min(b_idx[0], len(borrower_responses) - 1)]
        b_idx[0] += 1
        return r

    def mock_l(prof, current_offer=None):
        r = lender_responses[min(l_idx[0], len(lender_responses) - 1)]
        l_idx[0] += 1
        return r

    with patch("services.negotiation_graph.borrower_agent", side_effect=mock_b), \
         patch("services.negotiation_graph.lender_agent", side_effect=mock_l):

        initial_state: NegotiationState = {
            "session_id": 999,
            "borrower_id": 1,
            "lender_id": 1,
            "borrower_profile": {
                "loan_amount": 500000.0,
                "monthly_income": 100000.0,
                "max_emi": 16000.0,
                "max_interest_rate": 14.0,
                "preferred_tenure": 42,
                "max_tenure": 60,
                "collateral_required": False,
            },
            "lender_profile": {
                "max_loan_amount": 750000.0,
                "min_interest_rate": 10.0,
                "max_tenure": 60,
                "min_expected_return": 8.0,
                "collateral_required": False,
                "available_capacity": 600000.0,
            },
            "max_rounds": 6,
        }

        result = negotiation_graph.invoke(initial_state)

        assert result["agreement_found"] is True
        assert result["status"] == SessionStatus.AGREEMENT_REACHED.value
        assert result["round_number"] == 3
        assert result["final_proposal"]["interest_rate"] == 12.0
        assert result["final_proposal"]["tenure_months"] == 42
        assert result["final_proposal"]["amount"] == 500000.0

        # Verify mathematical EMI affordability constraint
        assert result["verification_result"]["valid"] is True
        assert result["verification_result"]["emi"] < 16000.0  # EMI at 12% for 42m is ~₹14,646
        assert len(result["negotiation_history"]) == 6

        # Check explicit offer ownership recorded
        actions = [(e["round"], e["agent"], e["action"]) for e in result["negotiation_history"]]
        assert actions == [
            (1, "borrower", "counter"),
            (1, "lender", "counter"),
            (2, "borrower", "counter"),
            (2, "lender", "counter"),
            (3, "borrower", "accept"),
            (3, "lender", "accept"),
        ]


# ===========================================================================
# 3. Database Persistence and Runner Tests
# ===========================================================================

def test_run_negotiation_session_db_persistence():
    """
    Verify run_negotiation_session loads data, invokes the graph,
    and accurately persists NegotiationSession and NegotiationOffer records to the database.
    """
    db = TestingSessionLocal()
    try:
        _, _, _, b_prof, l_prof, match = create_sample_entities(db)

        session = NegotiationSession(
            match_id=match.id,
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            status=SessionStatus.PENDING.value,
            current_round=1,
            max_rounds=6,
            agreement_found=False,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Mock agents to independently reach agreement in Round 1
        with patch("services.negotiation_graph.borrower_agent", return_value={"position": "accept", "reason": "Terms match"}), \
             patch("services.negotiation_graph.lender_agent", return_value={"position": "accept", "reason": "Terms match"}):

            res = run_negotiation_session(session.id, db)

            assert res["agreement_found"] is True
            assert res["status"] == SessionStatus.AGREEMENT_REACHED.value

            # Verify persisted session
            updated_session = db.query(NegotiationSession).filter_by(id=session.id).first()
            assert updated_session.status == SessionStatus.AGREEMENT_REACHED.value
            assert updated_session.agreement_found is True
            assert updated_session.final_proposal is not None

            # Verify persisted offers
            offers = db.query(NegotiationOffer).filter_by(session_id=session.id).all()
            assert len(offers) >= 2
            assert all(o.session_id == session.id for o in offers)
            assert any(o.agent_type == AgentType.BORROWER.value for o in offers)
            assert any(o.agent_type == AgentType.LENDER.value for o in offers)

            # Re-running an already completed session does not corrupt history
            rerun_res = run_negotiation_session(session.id, db)
            assert rerun_res["status"] == SessionStatus.AGREEMENT_REACHED.value
            offers_after_rerun = db.query(NegotiationOffer).filter_by(session_id=session.id).all()
            assert len(offers_after_rerun) == len(offers)
    finally:
        db.close()


# ===========================================================================
# 4. FastAPI Endpoint Integration Tests
# ===========================================================================

def test_create_session_endpoint_success(client: TestClient):
    db = TestingSessionLocal()
    try:
        b_user, _, _, _, _, match = create_sample_entities(db)
        match_id = match.id
        token = register_and_login(client, b_user.full_name, b_user.email, "borrower")
    finally:
        db.close()

    res = client.post(
        "/api/negotiations/session",
        json={"match_id": match_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["match_id"] == match_id
    assert data["status"] == SessionStatus.PENDING.value
    assert data["round_number"] == 1


def test_create_session_duplicate_prevention(client: TestClient):
    db = TestingSessionLocal()
    try:
        b_user, _, _, _, _, match = create_sample_entities(db)
        match_id = match.id
        token = register_and_login(client, b_user.full_name, b_user.email, "borrower")
    finally:
        db.close()

    # Create first
    res1 = client.post(
        "/api/negotiations/session",
        json={"match_id": match_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res1.status_code == 201
    session_id_1 = res1.json()["session_id"]

    # Attempt create again -> returns existing session
    res2 = client.post(
        "/api/negotiations/session",
        json={"match_id": match_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res2.status_code == 200 or res2.status_code == 201
    assert res2.json()["session_id"] == session_id_1


def test_create_session_match_not_accepted(client: TestClient):
    db = TestingSessionLocal()
    try:
        b_user, _, _, b_prof, l_prof, _ = create_sample_entities(db)
        pending_match = Match(
            borrower_id=b_prof.id,
            lender_id=l_prof.id,
            match_score=80.0,
            status=MatchStatus.PENDING.value,
        )
        db.add(pending_match)
        db.commit()
        db.refresh(pending_match)
        token = register_and_login(client, b_user.full_name, b_user.email, "borrower")
        m_id = pending_match.id
    finally:
        db.close()

    res = client.post(
        "/api/negotiations/session",
        json={"match_id": m_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 400
    assert "Match must be 'accepted'" in res.json()["detail"]


def test_start_and_get_negotiation_session_api(client: TestClient):
    db = TestingSessionLocal()
    try:
        b_user, l_user, _, b_prof, l_prof, match = create_sample_entities(db)
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
        token = register_and_login(client, b_user.full_name, b_user.email, "borrower")
    finally:
        db.close()

    with patch("services.negotiation_graph.borrower_agent", return_value={"position": "accept", "reason": "Agreed"}), \
         patch("services.negotiation_graph.lender_agent", return_value={"position": "accept", "reason": "Agreed"}):

        # Start session via API
        res = client.post(
            f"/api/negotiations/{session_id}/start",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == session_id
        assert data["status"] == SessionStatus.AGREEMENT_REACHED.value
        assert data["agreement_found"] is True
        assert data["verification"]["valid"] is True
        assert len(data["history"]) >= 2

        # Get session via API
        get_res = client.get(
            f"/api/negotiations/{session_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["session_id"] == session_id
        assert get_data["agreement_found"] is True
        assert len(get_data["history"]) >= 2

        # Get offers via API
        offers_res = client.get(
            f"/api/negotiations/{session_id}/offers",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert offers_res.status_code == 200
        offers_data = offers_res.json()
        assert len(offers_data) >= 2


def test_unauthorized_user_forbidden(client: TestClient):
    db = TestingSessionLocal()
    try:
        _, _, other_user, b_prof, l_prof, match = create_sample_entities(db)
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
        token = register_and_login(client, other_user.full_name, other_user.email, "borrower")
    finally:
        db.close()

    # Unauthorized access to get
    res = client.get(
        f"/api/negotiations/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403

    # Unauthorized access to start
    res_start = client.post(
        f"/api/negotiations/{session_id}/start",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_start.status_code == 403
