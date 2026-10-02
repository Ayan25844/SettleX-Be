from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import relationship

from database import Base


class SessionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    AGREEMENT_REACHED = "agreement_reached"
    REJECTED = "rejected"
    NO_AGREEMENT = "no_agreement"
    EXPIRED = "expired"


class AgentType(str, Enum):
    BORROWER = "borrower"
    LENDER = "lender"
    SYSTEM = "system"


class NegotiationSession(Base):
    __tablename__ = "negotiation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(
        Integer,
        ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    borrower_id = Column(
        Integer,
        ForeignKey("borrower_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    lender_id = Column(
        Integer,
        ForeignKey("lender_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    status = Column(
        String(50),
        nullable=False,
        default=SessionStatus.PENDING.value,
        index=True
    )
    current_round = Column(Integer, nullable=False, default=1)
    max_rounds = Column(Integer, nullable=False, default=6)
    current_offer = Column(JSON, nullable=True)
    final_proposal = Column(JSON, nullable=True)
    agreement_found = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    match = relationship("Match", foreign_keys=[match_id])
    borrower = relationship("BorrowerProfile", foreign_keys=[borrower_id])
    lender = relationship("LenderProfile", foreign_keys=[lender_id])
    offers = relationship(
        "NegotiationOffer",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="NegotiationOffer.id"
    )

    def __repr__(self) -> str:
        return f"<NegotiationSession id={self.id} match_id={self.match_id} status={self.status} round={self.current_round}>"


class NegotiationOffer(Base):
    __tablename__ = "negotiation_offers"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer,
        ForeignKey("negotiation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    round_number = Column(Integer, nullable=False)
    agent_type = Column(String(50), nullable=False)  # borrower, lender, system
    amount = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    upfront_payment = Column(Float, nullable=False, default=0.0)
    position = Column(String(50), nullable=False)  # initial, accept, counter, reject
    reason = Column(String(1000), nullable=True)
    is_valid = Column(Boolean, nullable=False, default=True)
    verification_result = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    session = relationship("NegotiationSession", back_populates="offers")

    def __repr__(self) -> str:
        return f"<NegotiationOffer id={self.id} session_id={self.session_id} agent={self.agent_type} round={self.round_number} pos={self.position}>"
