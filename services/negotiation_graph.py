import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session

from models.schemas import (
    BorrowerProfile as BorrowerProfileSchema,
    LenderProfile as LenderProfileSchema,
    LoanProposal,
)
from models.negotiation import (
    NegotiationSession,
    NegotiationOffer,
    SessionStatus,
    AgentType,
)
from models.profile import (
    BorrowerProfile as BorrowerProfileDB,
    LenderProfile as LenderProfileDB,
)
from services.agents import borrower_agent, lender_agent
from services.verifier import verify_proposal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph State Definition
# ---------------------------------------------------------------------------

class NegotiationState(TypedDict, total=False):
    session_id: int
    borrower_id: int
    lender_id: int
    borrower_profile: Dict[str, Any]
    lender_profile: Dict[str, Any]
    round_number: int
    max_rounds: int
    current_offer: Optional[Dict[str, Any]]
    previous_offer: Optional[Dict[str, Any]]
    current_offer_by: Optional[str]  # "borrower" | "lender" | "system"
    borrower_position: Optional[str]
    lender_position: Optional[str]
    borrower_agent_response: Optional[Dict[str, Any]]
    lender_agent_response: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]
    agreement_found: bool
    final_proposal: Optional[Dict[str, Any]]
    status: str
    decision: Optional[str]
    negotiation_history: List[Dict[str, Any]]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helper Parsers & Converters
# ---------------------------------------------------------------------------

def _to_borrower_schema(data: Dict[str, Any]) -> BorrowerProfileSchema:
    return BorrowerProfileSchema(
        loan_amount=float(data["loan_amount"]),
        monthly_income=float(data.get("monthly_income", 50000.0)),
        monthly_expenses=float(data.get("monthly_expenses", 0.0)),
        existing_emi=float(data.get("existing_emi", 0.0)),
        max_emi=float(data["max_emi"]),
        max_interest_rate=float(data["max_interest_rate"]),
        preferred_tenure=int(data["preferred_tenure"]),
        max_tenure=int(data["max_tenure"]),
        collateral_required=bool(data.get("collateral_required", False)),
    )


def _to_lender_schema(data: Dict[str, Any]) -> LenderProfileSchema:
    return LenderProfileSchema(
        max_loan_amount=float(data["max_loan_amount"]),
        min_interest_rate=float(data["min_interest_rate"]),
        max_tenure=int(data["max_tenure"]),
        min_expected_return=float(data.get("min_expected_return", 5.0)),
        collateral_required=bool(data.get("collateral_required", False)),
    )


def _parse_agent_target_rate(raw_val: Any, default_val: float) -> float:
    if raw_val is None:
        return default_val
    try:
        clean = str(raw_val).replace("%", "").strip()
        return round(float(clean), 2)
    except (ValueError, TypeError):
        return default_val


def _parse_agent_target_tenure(raw_val: Any, default_val: int) -> int:
    if raw_val is None:
        return default_val
    try:
        clean = str(raw_val).replace("months", "").strip()
        return int(float(clean))
    except (ValueError, TypeError):
        return default_val


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def initialize_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Initializes round counter, max rounds, active status, history, and establishes
    the baseline initial proposal from the borrower's request if not present.
    """
    max_rounds = state.get("max_rounds") or int(os.getenv("MAX_NEGOTIATION_ROUNDS", "6"))
    round_number = state.get("round_number") or 1

    b_prof = state.get("borrower_profile", {})
    l_prof = state.get("lender_profile", {})

    current_offer = state.get("current_offer")
    current_offer_by = state.get("current_offer_by")

    if not current_offer:
        # Establish baseline starting proposal from borrower profile
        min_rate = float(l_prof.get("min_interest_rate", 10.0))
        max_rate = float(b_prof.get("max_interest_rate", 15.0))
        initial_rate = round((min_rate + max_rate) / 2.0, 2)

        current_offer = {
            "amount": float(b_prof.get("loan_amount", 100000.0)),
            "interest_rate": initial_rate,
            "tenure_months": int(b_prof.get("preferred_tenure", 36)),
            "upfront_payment": 0.0,
        }
        current_offer_by = AgentType.SYSTEM.value

    return {
        "round_number": round_number,
        "max_rounds": max_rounds,
        "status": SessionStatus.ACTIVE.value,
        "agreement_found": False,
        "current_offer": current_offer,
        "current_offer_by": current_offer_by,
        "previous_offer": None,
        "negotiation_history": state.get("negotiation_history") or [],
    }


def borrower_agent_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Borrower Advocate AI node:
    - Receives ONLY borrower-side profile and public current offer.
    - Evaluates offer and returns structured position (accept, counter, reject).
    - If counter: converts to LoanProposal and executes deterministic verification.
    - If LLM fails: deterministic fallback counter is generated and verified.
    """
    borrower_schema = _to_borrower_schema(state["borrower_profile"])
    lender_schema = _to_lender_schema(state["lender_profile"])
    current_offer = state.get("current_offer")

    try:
        response = borrower_agent(borrower_schema, current_offer=current_offer)
        if isinstance(response, str):
            response = json.loads(response)
    except Exception as e:
        logger.warning(
            f"Borrower agent LLM execution error: {type(e).__name__}. Falling back to deterministic counter."
        )
        response = {
            "position": "counter",
            "reason": "Deterministic fallback counter based on borrower profile preference.",
            "target_interest_rate": borrower_schema.max_interest_rate - 0.5,
            "target_tenure_months": borrower_schema.preferred_tenure,
        }

    pos = str(response.get("position", "counter")).lower().strip()
    history = list(state.get("negotiation_history", []))

    if pos == "accept":
        # Borrower accepts current offer: verify the accepted terms
        prop = LoanProposal(
            amount=float(current_offer["amount"]),
            interest_rate=float(current_offer["interest_rate"]),
            tenure_months=int(current_offer["tenure_months"]),
            upfront_payment=float(current_offer.get("upfront_payment", 0.0)),
        )
        verification = verify_proposal(borrower_schema, lender_schema, prop)

        event = {
            "round": state["round_number"],
            "agent": AgentType.BORROWER.value,
            "action": "accept",
            "reason": response.get("reason"),
            "offer": current_offer,
            "verification": verification,
        }
        return {
            "borrower_position": "accept",
            "borrower_agent_response": response,
            "verification_result": verification,
            "negotiation_history": history + [event],
        }

    elif pos == "reject":
        event = {
            "round": state["round_number"],
            "agent": AgentType.BORROWER.value,
            "action": "reject",
            "reason": response.get("reason"),
            "offer": current_offer,
            "verification": None,
        }
        return {
            "borrower_position": "reject",
            "borrower_agent_response": response,
            "negotiation_history": history + [event],
        }

    else:
        # Counteroffer
        cur_rate = current_offer["interest_rate"] if current_offer else borrower_schema.max_interest_rate
        cur_tenure = current_offer["tenure_months"] if current_offer else borrower_schema.preferred_tenure

        target_rate = _parse_agent_target_rate(response.get("target_interest_rate"), cur_rate)
        target_tenure = _parse_agent_target_tenure(response.get("target_tenure_months"), cur_tenure)

        candidate_offer = {
            "amount": borrower_schema.loan_amount,
            "interest_rate": target_rate,
            "tenure_months": target_tenure,
            "upfront_payment": 0.0,
        }

        # Authoritative deterministic verification of candidate counteroffer
        prop = LoanProposal(
            amount=candidate_offer["amount"],
            interest_rate=candidate_offer["interest_rate"],
            tenure_months=candidate_offer["tenure_months"],
            upfront_payment=0.0,
        )
        verification = verify_proposal(borrower_schema, lender_schema, prop)

        event = {
            "round": state["round_number"],
            "agent": AgentType.BORROWER.value,
            "action": "counter",
            "reason": response.get("reason"),
            "offer": candidate_offer,
            "verification": verification,
        }

        return {
            "borrower_position": "counter",
            "borrower_agent_response": response,
            "previous_offer": current_offer,
            "current_offer": candidate_offer,
            "current_offer_by": AgentType.BORROWER.value,
            "verification_result": verification,
            "negotiation_history": history + [event],
        }


def lender_agent_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Lender Advocate AI node:
    - Receives ONLY lender-side profile and public current offer.
    - INDEPENDENTLY evaluates the active proposal via lender_agent().
    - Never automatically accepts just because the borrower accepted.
    - If counter: converts to LoanProposal and executes deterministic verification.
    - If LLM fails: deterministic fallback counter is generated and verified.
    """
    history = list(state.get("negotiation_history", []))
    current_offer = state.get("current_offer")

    # If borrower already outright rejected, terminate negotiation
    if state.get("borrower_position") == "reject":
        return {
            "lender_position": "reject",
            "lender_agent_response": {"position": "reject", "reason": "Borrower rejected terms."},
            "negotiation_history": history,
        }

    lender_schema = _to_lender_schema(state["lender_profile"])
    borrower_schema = _to_borrower_schema(state["borrower_profile"])
    borrower_amount = float(state["borrower_profile"]["loan_amount"])

    try:
        response = lender_agent(lender_schema, current_offer=current_offer)
        if isinstance(response, str):
            response = json.loads(response)
    except Exception as e:
        logger.warning(
            f"Lender agent LLM execution error: {type(e).__name__}. Falling back to deterministic counter."
        )
        response = {
            "position": "counter",
            "reason": "Deterministic fallback counter based on lender minimum return criteria.",
            "target_interest_rate": lender_schema.min_interest_rate + 0.5,
            "target_tenure_months": lender_schema.max_tenure,
        }

    pos = str(response.get("position", "counter")).lower().strip()

    if pos == "accept":
        # Lender independently accepts current offer: verify accepted terms
        prop = LoanProposal(
            amount=float(current_offer["amount"]),
            interest_rate=float(current_offer["interest_rate"]),
            tenure_months=int(current_offer["tenure_months"]),
            upfront_payment=float(current_offer.get("upfront_payment", 0.0)),
        )
        verification = verify_proposal(borrower_schema, lender_schema, prop)

        event = {
            "round": state["round_number"],
            "agent": AgentType.LENDER.value,
            "action": "accept",
            "reason": response.get("reason"),
            "offer": current_offer,
            "verification": verification,
        }
        return {
            "lender_position": "accept",
            "lender_agent_response": response,
            "verification_result": verification,
            "negotiation_history": history + [event],
        }

    elif pos == "reject":
        event = {
            "round": state["round_number"],
            "agent": AgentType.LENDER.value,
            "action": "reject",
            "reason": response.get("reason"),
            "offer": current_offer,
            "verification": None,
        }
        return {
            "lender_position": "reject",
            "lender_agent_response": response,
            "negotiation_history": history + [event],
        }

    else:
        # Counteroffer
        cur_rate = current_offer["interest_rate"] if current_offer else lender_schema.min_interest_rate
        cur_tenure = current_offer["tenure_months"] if current_offer else lender_schema.max_tenure

        target_rate = _parse_agent_target_rate(response.get("target_interest_rate"), cur_rate)
        target_tenure = _parse_agent_target_tenure(response.get("target_tenure_months"), cur_tenure)

        candidate_offer = {
            "amount": borrower_amount,
            "interest_rate": target_rate,
            "tenure_months": target_tenure,
            "upfront_payment": 0.0,
        }

        # Authoritative deterministic verification of candidate counteroffer
        prop = LoanProposal(
            amount=candidate_offer["amount"],
            interest_rate=candidate_offer["interest_rate"],
            tenure_months=candidate_offer["tenure_months"],
            upfront_payment=0.0,
        )
        verification = verify_proposal(borrower_schema, lender_schema, prop)

        event = {
            "round": state["round_number"],
            "agent": AgentType.LENDER.value,
            "action": "counter",
            "reason": response.get("reason"),
            "offer": candidate_offer,
            "verification": verification,
        }

        return {
            "lender_position": "counter",
            "lender_agent_response": response,
            "previous_offer": current_offer,
            "current_offer": candidate_offer,
            "current_offer_by": AgentType.LENDER.value,
            "verification_result": verification,
            "negotiation_history": history + [event],
        }


def verify_offer_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Authoritative deterministic verifier node:
    Re-verifies current active offer across borrower and lender hard constraints.
    """
    borrower_schema = _to_borrower_schema(state["borrower_profile"])
    lender_schema = _to_lender_schema(state["lender_profile"])
    offer = state.get("current_offer") or {}

    proposal = LoanProposal(
        amount=float(offer.get("amount", borrower_schema.loan_amount)),
        interest_rate=float(offer.get("interest_rate", 12.0)),
        tenure_months=int(offer.get("tenure_months", 36)),
        upfront_payment=float(offer.get("upfront_payment", 0.0)),
    )

    verification = verify_proposal(borrower_schema, lender_schema, proposal)

    return {
        "verification_result": verification,
    }


def decide_next_step_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Deterministic control flow decision node:
    - Outright rejection by either party -> 'reject'
    - BOTH accept AND verifier.valid == True -> 'agreement'
    - BOTH accept BUT verifier.valid == False -> 'reject' (cannot agree on invalid terms)
    - Reached max rounds -> 'round_limit'
    - Otherwise -> 'counter' (increments round_number and loops)
    """
    b_pos = state.get("borrower_position")
    l_pos = state.get("lender_position")
    verification = state.get("verification_result") or {}
    is_valid = bool(verification.get("valid", False))
    current_round = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", 6)

    # 1. Outright rejection by either side
    if b_pos == "reject" or l_pos == "reject":
        return {
            "decision": "reject",
            "status": SessionStatus.REJECTED.value,
            "agreement_found": False,
        }

    # 2. Both agents accepted
    if b_pos == "accept" and l_pos == "accept":
        if is_valid:
            return {
                "decision": "agreement",
                "status": SessionStatus.AGREEMENT_REACHED.value,
                "agreement_found": True,
                "final_proposal": state.get("current_offer"),
            }
        else:
            # Both accepted, but offer violates hard constraints -> reject
            return {
                "decision": "reject",
                "status": SessionStatus.REJECTED.value,
                "agreement_found": False,
            }

    # 3. Round limit reached
    if current_round >= max_rounds:
        return {
            "decision": "round_limit",
            "status": SessionStatus.NO_AGREEMENT.value,
            "agreement_found": False,
        }

    # 4. Continue negotiation to next round
    return {
        "decision": "counter",
        "round_number": current_round + 1,
    }


def should_continue(state: NegotiationState) -> str:
    """
    Conditional edge router:
    If decision is agreement, reject, or round_limit, route to finalize.
    Otherwise loop back to borrower_agent for the next round.
    """
    decision = state.get("decision")
    if decision in ("agreement", "reject", "round_limit"):
        return "finalize"
    return "borrower_agent"


def finalize_node(state: NegotiationState) -> Dict[str, Any]:
    """
    Final node: guarantees final_proposal is accurately set if agreement was reached.
    """
    final_proposal = state.get("final_proposal")
    if state.get("agreement_found") and not final_proposal:
        final_proposal = state.get("current_offer")

    return {
        "final_proposal": final_proposal,
        "status": state.get("status", SessionStatus.NO_AGREEMENT.value),
    }


# ---------------------------------------------------------------------------
# LangGraph Assembly & Compilation
# ---------------------------------------------------------------------------

builder = StateGraph(NegotiationState)

builder.add_node("initialize", initialize_node)
builder.add_node("borrower_agent", borrower_agent_node)
builder.add_node("lender_agent", lender_agent_node)
builder.add_node("verify_offer", verify_offer_node)
builder.add_node("decide_next_step", decide_next_step_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "initialize")
builder.add_edge("initialize", "borrower_agent")
builder.add_edge("borrower_agent", "lender_agent")
builder.add_edge("lender_agent", "verify_offer")
builder.add_edge("verify_offer", "decide_next_step")
builder.add_conditional_edges(
    "decide_next_step",
    should_continue,
    {
        "finalize": "finalize",
        "borrower_agent": "borrower_agent",
    }
)
builder.add_edge("finalize", END)

negotiation_graph = builder.compile()


# ---------------------------------------------------------------------------
# Runner Function with Database Persistence
# ---------------------------------------------------------------------------

def run_negotiation_session(session_id: int, db: Session) -> Dict[str, Any]:
    """
    Orchestrates the complete multi-round negotiation session:
    1. Loads NegotiationSession and associated Borrower/Lender profiles from DB.
    2. Protects completed sessions from re-execution.
    3. Constructs initial NegotiationState.
    4. Executes the compiled LangGraph.
    5. Persists the final session status and individual offers into the database.
    6. Returns the final state dictionary.
    """
    session = db.query(NegotiationSession).filter_by(id=session_id).first()
    if not session:
        raise ValueError(f"Negotiation session with ID {session_id} not found.")

    # Guard: If session is already finalized, do not re-run or wipe history
    if session.status in (
        SessionStatus.AGREEMENT_REACHED.value,
        SessionStatus.REJECTED.value,
        SessionStatus.NO_AGREEMENT.value,
    ):
        logger.info(
            f"Negotiation session {session.id} is already in terminal status '{session.status}'. Skipping re-run."
        )
        return {
            "session_id": session.id,
            "borrower_id": session.borrower_id,
            "lender_id": session.lender_id,
            "round_number": session.current_round,
            "max_rounds": session.max_rounds,
            "current_offer": session.current_offer,
            "final_proposal": session.final_proposal,
            "agreement_found": session.agreement_found,
            "status": session.status,
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

    if not borrower_profile or not lender_profile:
        raise ValueError("Associated borrower or lender profile not found for this session.")

    b_dict = {
        "loan_amount": borrower_profile.loan_amount,
        "monthly_income": borrower_profile.monthly_income,
        "monthly_expenses": borrower_profile.monthly_expenses,
        "existing_emi": borrower_profile.existing_emi,
        "max_emi": borrower_profile.max_emi,
        "max_interest_rate": borrower_profile.max_interest_rate,
        "preferred_tenure": borrower_profile.preferred_tenure,
        "max_tenure": borrower_profile.max_tenure,
        "collateral_required": borrower_profile.collateral_required,
    }

    l_dict = {
        "max_loan_amount": lender_profile.max_loan_amount,
        "min_interest_rate": lender_profile.min_interest_rate,
        "max_tenure": lender_profile.max_tenure,
        "min_expected_return": lender_profile.min_expected_return,
        "collateral_required": lender_profile.collateral_required,
        "available_capacity": lender_profile.available_capacity,
    }

    max_rounds = session.max_rounds or int(os.getenv("MAX_NEGOTIATION_ROUNDS", "6"))

    initial_state: NegotiationState = {
        "session_id": session.id,
        "borrower_id": session.borrower_id,
        "lender_id": session.lender_id,
        "borrower_profile": b_dict,
        "lender_profile": l_dict,
        "round_number": 1,
        "max_rounds": max_rounds,
        "current_offer": session.current_offer,
        "current_offer_by": None,
        "previous_offer": None,
        "borrower_position": None,
        "lender_position": None,
        "borrower_agent_response": None,
        "lender_agent_response": None,
        "verification_result": None,
        "agreement_found": False,
        "final_proposal": None,
        "status": SessionStatus.ACTIVE.value,
        "decision": None,
        "negotiation_history": [],
        "error": None,
    }

    # Execute LangGraph
    final_state = negotiation_graph.invoke(initial_state)

    # Persist session state in DB
    session.status = final_state.get("status", SessionStatus.NO_AGREEMENT.value)
    session.current_round = final_state.get("round_number", 1)
    session.current_offer = final_state.get("current_offer")
    session.final_proposal = final_state.get("final_proposal")
    session.agreement_found = final_state.get("agreement_found", False)
    session.updated_at = datetime.now(timezone.utc)

    # Clean existing pending offers and persist finalized offer history
    db.query(NegotiationOffer).filter_by(session_id=session.id).delete()

    history = final_state.get("negotiation_history", [])
    for event in history:
        offer_data = event.get("offer") or {}
        verification = event.get("verification")
        offer_record = NegotiationOffer(
            session_id=session.id,
            round_number=event.get("round", 1),
            agent_type=event.get("agent", AgentType.SYSTEM.value),
            amount=float(offer_data.get("amount", b_dict["loan_amount"])),
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

    return final_state
