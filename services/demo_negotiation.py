"""
Demo Simulation Service for SettleX Hackathon Demonstrations.

Provides a deterministic, scripted multi-round borrower/lender negotiation
without invoking external LLMs (Google Gemini / OpenAI), while maintaining
full compatibility with:
- The authoritative deterministic financial verifier (services.verifier)
- Exact database persistence (NegotiationSession and NegotiationOffer tables)
- Negotiation timeline history structure
- Frontend API response format (with demo_mode=True flag)

NOTE: This is strictly an internal demo simulation mode for testing and hackathon
presentations. Live mode remains fully functional and accessible when
SETTLEX_DEMO_MODE=false.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from models.negotiation import AgentType, NegotiationOffer, NegotiationSession, SessionStatus
from models.profile import BorrowerProfile as BorrowerProfileDB, LenderProfile as LenderProfileDB
from models.schemas import BorrowerProfile as BorrowerProfileSchema, LenderProfile as LenderProfileSchema, LoanProposal
from services.verifier import verify_proposal

logger = logging.getLogger("settlex.demo_negotiation")


def is_demo_mode() -> bool:
    """
    Checks if the demo negotiation mode is enabled via the SETTLEX_DEMO_MODE environment variable.
    Accepts: 'true', '1', 'yes', 'on' (case-insensitive).
    """
    val = os.getenv("SETTLEX_DEMO_MODE", "false").lower().strip()
    return val in ("true", "1", "yes", "on")


def run_demo_negotiation_session(session_id: int, db: Session) -> Dict[str, Any]:
    """
    Executes a deterministic scripted multi-round negotiation simulation:
    - Initial indicative proposal: ₹5,00,000 · 12.00% · 42 months
    - Round 1 Borrower Advocate: Counter to 11.80% · 42 months
    - Round 1 Lender Advocate: Counter to 11.95% · 42 months
    - Round 2 Borrower Advocate: Accept 11.95% · 42 months
    - Round 2 Lender Advocate: Accept 11.95% · 42 months
    - Final Agreement: Agreement reached at 11.95% · 42 months

    All EMI, utility, repayment, and constraint calculations are executed
    via the authoritative deterministic verifier (verify_proposal), NOT hardcoded.
    """
    session = db.query(NegotiationSession).filter_by(id=session_id).first()
    if not session:
        raise ValueError(f"Negotiation session with ID {session_id} not found.")

    # Guard: If session is already finalized, return existing session state with demo flag
    if session.status in (
        SessionStatus.AGREEMENT_REACHED.value,
        SessionStatus.REJECTED.value,
        SessionStatus.NO_AGREEMENT.value,
    ):
        logger.info(
            f"[Demo Mode] Session {session.id} is already in terminal status '{session.status}'. Returning existing."
        )
        return {
            "session_id": session.id,
            "match_id": session.match_id,
            "borrower_id": session.borrower_id,
            "lender_id": session.lender_id,
            "round_number": session.current_round,
            "max_rounds": session.max_rounds or 6,
            "current_offer": session.current_offer,
            "final_proposal": session.final_proposal,
            "agreement_found": session.agreement_found,
            "status": session.status,
            "demo_mode": True,
            "negotiation_history": [
                {
                    "round": o.round_number,
                    "agent": o.agent_type,
                    "action": o.position,
                    "reason": o.reason,
                    "offer": {
                        "amount": o.amount,
                        "interest_rate": o.interest_rate,
                        "tenure_months": o.tenure_months,
                        "upfront_payment": o.upfront_payment,
                    },
                    "verification": o.verification_result,
                }
                for o in session.offers
            ],
        }

    borrower_profile = db.query(BorrowerProfileDB).filter_by(id=session.borrower_id).first()
    lender_profile = db.query(LenderProfileDB).filter_by(id=session.lender_id).first()

    # Standard demo parameters (500,000 loan, 42m tenure)
    demo_amount = 500000.0
    demo_tenure = 42

    # Construct robust schemas for verification
    borrower_schema = BorrowerProfileSchema(
        loan_amount=demo_amount,
        monthly_income=float(getattr(borrower_profile, "monthly_income", 100000.0) or 100000.0),
        monthly_expenses=float(getattr(borrower_profile, "monthly_expenses", 30000.0) or 30000.0),
        existing_emi=float(getattr(borrower_profile, "existing_emi", 5000.0) or 5000.0),
        max_emi=max(float(getattr(borrower_profile, "max_emi", 35000.0) or 35000.0), 30000.0),
        max_interest_rate=max(float(getattr(borrower_profile, "max_interest_rate", 14.0) or 14.0), 13.5),
        preferred_tenure=demo_tenure,
        max_tenure=max(int(getattr(borrower_profile, "max_tenure", 60) or 60), 48),
        collateral_required=False,
    )
    lender_schema = LenderProfileSchema(
        max_loan_amount=max(float(getattr(lender_profile, "max_loan_amount", 1000000.0) or 1000000.0), demo_amount),
        min_interest_rate=min(float(getattr(lender_profile, "min_interest_rate", 10.0) or 10.0), 11.0),
        max_tenure=max(int(getattr(lender_profile, "max_tenure", 60) or 60), 48),
        min_expected_return=float(getattr(lender_profile, "min_expected_return", 5.0) or 5.0),
        collateral_required=False,
    )

    logger.info(
        f"[Demo Mode] Starting deterministic simulation for session {session_id} "
        f"with amount ₹{demo_amount:,.2f} · {demo_tenure}m tenure"
    )

    # 1. Round 1: Borrower counters to 11.80%
    b1_offer = {
        "amount": demo_amount,
        "interest_rate": 11.80,
        "tenure_months": demo_tenure,
        "upfront_payment": 0.0,
    }
    b1_prop = LoanProposal(
        amount=b1_offer["amount"],
        interest_rate=b1_offer["interest_rate"],
        tenure_months=b1_offer["tenure_months"],
        upfront_payment=0.0,
    )
    b1_verification = verify_proposal(borrower_schema, lender_schema, b1_prop)
    b1_event = {
        "round": 1,
        "agent": AgentType.BORROWER.value,
        "action": "counter",
        "reason": "The borrower prefers a lower interest rate while retaining the preferred tenure.",
        "offer": b1_offer,
        "verification": b1_verification,
    }

    # 2. Round 1: Lender counters to 11.95%
    l1_offer = {
        "amount": demo_amount,
        "interest_rate": 11.95,
        "tenure_months": demo_tenure,
        "upfront_payment": 0.0,
    }
    l1_prop = LoanProposal(
        amount=l1_offer["amount"],
        interest_rate=l1_offer["interest_rate"],
        tenure_months=l1_offer["tenure_months"],
        upfront_payment=0.0,
    )
    l1_verification = verify_proposal(borrower_schema, lender_schema, l1_prop)
    l1_event = {
        "round": 1,
        "agent": AgentType.LENDER.value,
        "action": "counter",
        "reason": "The lender can improve the rate while maintaining an acceptable return.",
        "offer": l1_offer,
        "verification": l1_verification,
    }

    # 3. Round 2: Borrower accepts 11.95%
    b2_offer = dict(l1_offer)
    b2_verification = verify_proposal(borrower_schema, lender_schema, l1_prop)
    b2_event = {
        "round": 2,
        "agent": AgentType.BORROWER.value,
        "action": "accept",
        "reason": "The offered rate of 11.95% meets the borrower's affordability criteria and preferred 42-month tenure.",
        "offer": b2_offer,
        "verification": b2_verification,
    }

    # 4. Round 2: Lender accepts 11.95%
    l2_offer = dict(l1_offer)
    l2_verification = verify_proposal(borrower_schema, lender_schema, l1_prop)
    l2_event = {
        "round": 2,
        "agent": AgentType.LENDER.value,
        "action": "accept",
        "reason": "The rate of 11.95% achieves the lender's risk-adjusted return hurdles while maintaining optimal duration.",
        "offer": l2_offer,
        "verification": l2_verification,
    }

    history: List[Dict[str, Any]] = [b1_event, l1_event, b2_event, l2_event]
    final_proposal = dict(l1_offer)
    final_verification = l2_verification

    # Persist session state in DB
    session.status = SessionStatus.AGREEMENT_REACHED.value
    session.current_round = 2
    session.current_offer = final_proposal
    session.final_proposal = final_proposal
    session.agreement_found = True
    session.updated_at = datetime.now(timezone.utc)

    # Clean existing offers and persist simulated offers
    db.query(NegotiationOffer).filter_by(session_id=session.id).delete()

    for event in history:
        offer_data = event.get("offer") or {}
        verification = event.get("verification")
        offer_record = NegotiationOffer(
            session_id=session.id,
            round_number=event.get("round", 1),
            agent_type=event.get("agent", AgentType.SYSTEM.value),
            amount=float(offer_data.get("amount", demo_amount)),
            interest_rate=float(offer_data.get("interest_rate", 0.0)),
            tenure_months=int(offer_data.get("tenure_months", 0)),
            upfront_payment=float(offer_data.get("upfront_payment", 0.0)),
            position=event.get("action", "initial"),
            reason=event.get("reason"),
            is_valid=verification.get("valid", True) if verification else True,
            verification_result=verification,
        )
        db.add(offer_record)

    db.commit()
    db.refresh(session)

    logger.info(
        f"[Demo Mode] Simulation successfully completed and persisted for session {session_id}. "
        f"Agreed at ₹{final_proposal['amount']:,.2f} · {final_proposal['interest_rate']}% · "
        f"{final_proposal['tenure_months']}m (EMI: ₹{final_verification.get('emi')})"
    )

    return {
        "session_id": session.id,
        "match_id": session.match_id,
        "borrower_id": session.borrower_id,
        "lender_id": session.lender_id,
        "round_number": 2,
        "max_rounds": session.max_rounds or 6,
        "status": SessionStatus.AGREEMENT_REACHED.value,
        "agreement_found": True,
        "current_offer": final_proposal,
        "final_proposal": final_proposal,
        "verification_result": final_verification,
        "negotiation_history": history,
        "demo_mode": True,
    }
