from fastapi import FastAPI
from models.schemas import EvaluationRequest
from services.verifier import verify_proposal
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import BorrowerProfile, LenderProfile
from services.negotiation import find_best_deal
from services.agents import borrower_agent, lender_agent

app = FastAPI(
    title="SettleX API",
    description="AI-mediated financial negotiation backend",
    version="0.1.0"
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

