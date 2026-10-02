from datetime import datetime
import re
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.user import UserRole

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=1, description="Full name of the user")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (at least 8 characters)")
    role: str = Field(default="borrower", description="Role: 'borrower' or 'lender'")

    @field_validator("full_name")
    @classmethod
    def trim_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Full name cannot be empty")
        return v

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("role")
    @classmethod
    def normalize_role(cls, v: str) -> str:
        return v.strip().lower()


class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("role", mode="before")
    @classmethod
    def convert_role(cls, v):
        if isinstance(v, UserRole):
            return v.value
        return str(v)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class RoleUpdateRequest(BaseModel):
    role: str = Field(..., description="New role for the user: 'borrower', 'lender', or 'admin'")

    @field_validator("role")
    @classmethod
    def normalize_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("borrower", "lender", "admin"):
            raise ValueError("Role must be 'borrower', 'lender', or 'admin'")
        return v


class MessageResponse(BaseModel):
    message: str
