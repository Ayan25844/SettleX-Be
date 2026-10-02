from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from database import Base


class MatchStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
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
    match_score = Column(Float, nullable=False)
    score_breakdown = Column(JSON, nullable=True)
    status = Column(
        String(50),
        nullable=False,
        default=MatchStatus.PENDING.value,
        index=True
    )
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
    borrower = relationship(
        "BorrowerProfile",
        back_populates="matches",
        foreign_keys=[borrower_id]
    )
    lender = relationship(
        "LenderProfile",
        back_populates="matches",
        foreign_keys=[lender_id]
    )

    def __repr__(self) -> str:
        return f"<Match id={self.id} borrower_id={self.borrower_id} lender_id={self.lender_id} score={self.match_score} status={self.status}>"
