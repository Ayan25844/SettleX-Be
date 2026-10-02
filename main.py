from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from models.schemas import BorrowerProfile, EvaluationRequest, LenderProfile
from routers.admin import router as admin_router
from routers.auth import router as auth_router
from routers.matching import router as matching_router
from routers.profiles import borrower_router, lender_router
from routers.test_auth import router as test_auth_router
from services.agents import borrower_agent, lender_agent
from services.negotiation import find_best_deal
from services.verifier import verify_proposal


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: automatically initialize database tables on startup."""
    init_db()
    yield


app = FastAPI(
    title="SettleX API",
    description="AI-mediated financial negotiation backend",
    version="0.2.0",
    lifespan=lifespan
)

# Allow the Next.js frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication and Administration routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(test_auth_router, prefix="/api/test", tags=["test"])

# Profile management routers
app.include_router(borrower_router)
app.include_router(lender_router)

# Matching Engine router
app.include_router(matching_router)


@app.get("/")
def root():
    return {
        "message": "SettleX API is running",
        "status": "ok"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "SettleX backend"
    }


@app.post("/api/loan/evaluate")
def evaluate_loan(request: EvaluationRequest):

    result = verify_proposal(
        request.borrower,
        request.lender,
        request.proposal
    )

    return result


@app.post("/api/negotiation/start")
def start_negotiation(
    borrower: BorrowerProfile,
    lender: LenderProfile
):

    result = find_best_deal(
        borrower,
        lender
    )

    return result


@app.post("/api/agents/test")
def test_agents(
    borrower: BorrowerProfile,
    lender: LenderProfile
):

    current_offer = {
        "amount": borrower.loan_amount,
        "interest_rate": 12,
        "tenure_months": 42
    }

    borrower_response = borrower_agent(
        borrower,
        current_offer
    )

    lender_response = lender_agent(
        lender,
        current_offer
    )

    return {
        "current_offer": current_offer,
        "borrower_agent": borrower_response,
        "lender_agent": lender_response
    }
