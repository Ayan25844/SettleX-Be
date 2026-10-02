from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
)
from sqlalchemy.orm import relationship

from database import Base


class BorrowerProfile(Base):
    __tablename__ = "borrower_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )
    loan_amount = Column(Float, nullable=False)
    monthly_income = Column(Float, nullable=False)
    monthly_expenses = Column(Float, nullable=False, default=0.0)
    existing_emi = Column(Float, nullable=False, default=0.0)
    max_emi = Column(Float, nullable=False)
    max_interest_rate = Column(Float, nullable=False)
    preferred_tenure = Column(Integer, nullable=False)
    max_tenure = Column(Integer, nullable=False)
    collateral_required = Column(Boolean, nullable=False, default=False)
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
    user = relationship("User", back_populates="borrower_profile")
    matches = relationship(
        "Match",
        back_populates="borrower",
        cascade="all, delete-orphan",
        foreign_keys="Match.borrower_id"
    )

    def __repr__(self) -> str:
        return f"<BorrowerProfile id={self.id} user_id={self.user_id} amount={self.loan_amount}>"


class LenderProfile(Base):
    __tablename__ = "lender_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True
    )
    max_loan_amount = Column(Float, nullable=False)
    min_interest_rate = Column(Float, nullable=False)
    max_tenure = Column(Integer, nullable=False)
    min_expected_return = Column(Float, nullable=False)
    collateral_required = Column(Boolean, nullable=False, default=False)
    available_capacity = Column(Float, nullable=False)
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
    user = relationship("User", back_populates="lender_profile")
    matches = relationship(
        "Match",
        back_populates="lender",
        cascade="all, delete-orphan",
        foreign_keys="Match.lender_id"
    )

    def __repr__(self) -> str:
        return f"<LenderProfile id={self.id} user_id={self.user_id} capacity={self.available_capacity}>"
