import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Reconfigure stdout/stderr for UTF-8 support on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from database import SessionLocal, engine, init_db
from models.profile import LenderProfile
from models.user import User, UserRole
from services.auth import hash_password
from services.matching import find_and_rank_matches


def generate_synthetic_lender_specs() -> List[Dict[str, Any]]:
    """
    Generates specifications for 100 diverse synthetic lenders.
    Distribution:
      ~20 Excellent compatibility
      ~30 Good compatibility
      ~25 Partial / tighter compatibility
      ~25 Incompatible (violating loan amount, interest rate, tenure, capacity, or collateral)
    """
    lenders = []

    # --------------------------------------------------------------------------
    # 1..20: Excellent compatibility (Top tier)
    # High capacity, generous max loan, competitive low rates (8.0% - 11.5%), long tenure (60-84m)
    # --------------------------------------------------------------------------
    amounts_exc = [750000, 1000000, 1000000, 1200000, 1500000, 1500000, 2000000, 2000000, 2500000, 2500000]
    for i in range(1, 21):
        rate = round(8.0 + (i - 1) * (3.5 / 19), 2)  # 8.00% to 11.50%
        amt = amounts_exc[(i - 1) % len(amounts_exc)]
        tenure = 60 if i % 2 == 0 else (72 if i % 3 == 0 else 84)
        cap = amt * (3 + (i % 4))  # ₹2,250,000 to ₹15,000,000
        ret = round(1.08 + (rate / 100.0) * 0.7, 2)
        lenders.append({
            "idx": i,
            "max_loan_amount": float(amt),
            "min_interest_rate": float(rate),
            "max_tenure": int(tenure),
            "min_expected_return": float(ret),
            "collateral_required": False,
            "available_capacity": float(cap),
            "category": "Excellent Compatibility"
        })

    # --------------------------------------------------------------------------
    # 21..50: Good compatibility (Mid-tier)
    # Solid capacity (₹1M - ₹3M), rates 11.6% - 13.5%, tenure 48-60m
    # --------------------------------------------------------------------------
    amounts_good = [500000, 600000, 750000, 800000, 1000000]
    for i in range(21, 51):
        rate = round(11.6 + (i - 21) * (1.9 / 29), 2)  # 11.60% to 13.50%
        amt = amounts_good[(i - 21) % len(amounts_good)]
        tenure = 48 if i % 3 == 0 else 60
        cap = amt * (2 + (i % 3))  # ₹1,000,000 to ₹3,000,000
        ret = round(1.15 + (rate / 100.0) * 0.7, 2)
        lenders.append({
            "idx": i,
            "max_loan_amount": float(amt),
            "min_interest_rate": float(rate),
            "max_tenure": int(tenure),
            "min_expected_return": float(ret),
            "collateral_required": False,
            "available_capacity": float(cap),
            "category": "Good Compatibility"
        })

    # --------------------------------------------------------------------------
    # 51..75: Partial / Tighter compatibility
    # Tighter rates (13.5% - 14.0%), tighter tenure (42-48m), or exact ticket fit
    # --------------------------------------------------------------------------
    for i in range(51, 76):
        rate = round(13.5 + (i - 51) * (0.5 / 24), 2)  # 13.50% to 14.00%
        amt = 500000.0 if i % 2 == 0 else 550000.0
        tenure = 42 if i % 2 == 0 else 48
        cap = amt + 100000.0 * (1 + (i % 5))  # ₹600,000 to ₹1,100,000
        ret = round(1.22 + (rate / 100.0) * 0.7, 2)
        # A few require collateral
        collat = (i in (60, 65, 70))
        lenders.append({
            "idx": i,
            "max_loan_amount": float(amt),
            "min_interest_rate": float(rate),
            "max_tenure": int(tenure),
            "min_expected_return": float(ret),
            "collateral_required": collat,
            "available_capacity": float(cap),
            "category": "Partial Compatibility"
        })

    # --------------------------------------------------------------------------
    # 76..100: Incompatible lenders (Constraint stress tests)
    # --------------------------------------------------------------------------
    # 76-80: Incompatible Loan Amount (< ₹500,000)
    low_amounts = [100000, 150000, 200000, 250000, 300000]
    for i in range(76, 81):
        amt = low_amounts[i - 76]
        lenders.append({
            "idx": i,
            "max_loan_amount": float(amt),
            "min_interest_rate": 10.5,
            "max_tenure": 60,
            "min_expected_return": 1.12,
            "collateral_required": False,
            "available_capacity": float(amt * 3),
            "category": "Incompatible (Loan Amount Too Low)"
        })

    # 81-86: Incompatible Interest Rate (14.5% - 18.0% > 14.0% borrower max)
    high_rates = [14.5, 15.0, 15.5, 16.5, 17.5, 18.0]
    for i in range(81, 87):
        rate = high_rates[i - 81]
        lenders.append({
            "idx": i,
            "max_loan_amount": 1000000.0,
            "min_interest_rate": float(rate),
            "max_tenure": 60,
            "min_expected_return": round(1.2 + rate / 100.0, 2),
            "collateral_required": False,
            "available_capacity": 2000000.0,
            "category": "Incompatible (Interest Rate Too High)"
        })

    # 87-91: Incompatible Tenure (12m - 36m < 42m borrower preferred)
    short_tenures = [12, 18, 24, 30, 36]
    for i in range(87, 92):
        t = short_tenures[i - 87]
        lenders.append({
            "idx": i,
            "max_loan_amount": 750000.0,
            "min_interest_rate": 10.0,
            "max_tenure": int(t),
            "min_expected_return": 1.10,
            "collateral_required": False,
            "available_capacity": 1500000.0,
            "category": "Incompatible (Tenure Too Short)"
        })

    # 92-96: Incompatible Capacity (Capacity ₹50,000 - ₹400,000 < ₹500,000)
    low_capacities = [50000, 100000, 150000, 250000, 400000]
    for i in range(92, 97):
        cap = low_capacities[i - 92]
        lenders.append({
            "idx": i,
            "max_loan_amount": 1000000.0,
            "min_interest_rate": 11.0,
            "max_tenure": 60,
            "min_expected_return": 1.15,
            "collateral_required": False,
            "available_capacity": float(cap),
            "category": "Incompatible (Insufficient Capacity)"
        })

    # 97-100: Incompatible Collateral (Mandatory collateral)
    for i in range(97, 101):
        lenders.append({
            "idx": i,
            "max_loan_amount": 800000.0,
            "min_interest_rate": 9.5 + (i - 97) * 0.5,
            "max_tenure": 60,
            "min_expected_return": 1.14,
            "collateral_required": True,
            "available_capacity": 2000000.0,
            "category": "Incompatible (Mandatory Collateral)"
        })

    return lenders


class SampleBorrower:
    """Mock borrower profile for matching engine verification."""
    def __init__(self):
        self.loan_amount = 500000.0
        self.monthly_income = 85000.0
        self.monthly_expenses = 32000.0
        self.existing_emi = 8500.0
        self.max_emi = 16000.0
        self.max_interest_rate = 14.0
        self.preferred_tenure = 42
        self.max_tenure = 60
        self.collateral_required = False


def seed_lenders():
    print("=" * 60)
    print("SettleX Synthetic Lender Pool Seeder")
    print("=" * 60)

    # 1. Initialize DB and verify connection
    init_db()
    db = SessionLocal()

    try:
        # Pre-fetch existing synthetic users and lender profiles in a single query
        existing_users = {
            u.email: u
            for u in db.query(User).filter(User.email.like("lender%@settlex-demo.local")).all()
        }
        existing_profiles = {
            p.user_id: p
            for p in db.query(LenderProfile).all()
        }

        print(f"Connected to database: {engine.url.drivername}")
        print(f"Existing synthetic lenders found: {len(existing_users)}")

        # Hash secure synthetic password once for all synthetic accounts
        default_hashed_pass = hash_password("SyntheticLender2026!Secure")

        specs = generate_synthetic_lender_specs()
        created_users = 0
        created_profiles = 0
        skipped_count = 0

        print(f"\nProcessing {len(specs)} synthetic lender specifications...")

        for spec in specs:
            idx = spec["idx"]
            email = f"lender{idx:03d}@settlex-demo.local"
            name = f"Lender Pool {idx:03d}"

            # Check if user already exists
            user = existing_users.get(email)
            if not user:
                user = User(
                    full_name=name,
                    email=email,
                    password_hash=default_hashed_pass,
                    role=UserRole.LENDER,
                    is_active=True
                )
                db.add(user)
                db.flush()  # Populates user.id
                existing_users[email] = user
                created_users += 1

            # Check if lender profile already exists for this user
            profile = existing_profiles.get(user.id)
            if not profile:
                profile = LenderProfile(
                    user_id=user.id,
                    max_loan_amount=spec["max_loan_amount"],
                    min_interest_rate=spec["min_interest_rate"],
                    max_tenure=spec["max_tenure"],
                    min_expected_return=spec["min_expected_return"],
                    collateral_required=spec["collateral_required"],
                    available_capacity=spec["available_capacity"]
                )
                db.add(profile)
                existing_profiles[user.id] = profile
                created_profiles += 1
            else:
                skipped_count += 1

        db.commit()

        print("\nSeeding Operation Results:")
        print(f"  Synthetic users created:           {created_users}")
        print(f"  Lender profiles created:          {created_profiles}")
        print(f"  Already existing records skipped: {skipped_count}")

        # ----------------------------------------------------------------------
        # Verification & Aggregate Statistics
        # ----------------------------------------------------------------------
        all_synthetic_profiles = (
            db.query(LenderProfile)
            .join(User, LenderProfile.user_id == User.id)
            .filter(User.email.like("lender%@settlex-demo.local"))
            .all()
        )

        total_count = len(all_synthetic_profiles)
        if total_count > 0:
            min_loan = min(p.max_loan_amount for p in all_synthetic_profiles)
            max_loan = max(p.max_loan_amount for p in all_synthetic_profiles)
            min_rate = min(p.min_interest_rate for p in all_synthetic_profiles)
            max_rate = max(p.min_interest_rate for p in all_synthetic_profiles)
            collat_count = sum(1 for p in all_synthetic_profiles if p.collateral_required)
            total_cap = sum(p.available_capacity for p in all_synthetic_profiles)

            print("\n" + "=" * 60)
            print("Synthetic Lender Pool Overview")
            print("=" * 60)
            print(f"Total active synthetic lenders: {total_count}")
            print(f"Loan amount range:             INR {min_loan:,.0f} -> INR {max_loan:,.0f}")
            print(f"Interest rate range:           {min_rate:.1f}% -> {max_rate:.1f}%")
            print(f"Tenures supported:             12 -> 84 months")
            print(f"Collateral required count:     {collat_count} / {total_count}")
            print(f"Total available capacity:      INR {total_cap:,.0f}")

            # ------------------------------------------------------------------
            # Test Matching Engine with Benchmark Borrower
            # ------------------------------------------------------------------
            print("\n" + "=" * 60)
            print("Matching Engine Verification with Benchmark Borrower")
            print("=" * 60)
            benchmark_borrower = SampleBorrower()
            print(f"Borrower criteria:")
            print(f"  Loan Amount:       INR {benchmark_borrower.loan_amount:,.0f}")
            print(f"  Max Interest Rate: {benchmark_borrower.max_interest_rate:.1f}%")
            print(f"  Preferred Tenure:  {benchmark_borrower.preferred_tenure} months (Max: {benchmark_borrower.max_tenure}m)")
            print(f"  Collateral:        {benchmark_borrower.collateral_required}")

            ranked_results = find_and_rank_matches(benchmark_borrower, all_synthetic_profiles)
            compatible_count = len(ranked_results)
            incompatible_count = total_count - compatible_count

            print(f"\nMatching Results:")
            print(f"  Total lenders evaluated:  {total_count}")
            print(f"  Compatible lenders found: {compatible_count}")
            print(f"  Incompatible filtered:    {incompatible_count}")

            print(f"\nTop 5 Ranked Compatible Matches:")
            for i, match in enumerate(ranked_results[:5], 1):
                l = match["lender"]
                s = match["match_score"]
                bd = match["score_breakdown"]
                print(f"  #{i} Lender ID {l.id} (User ID {l.user_id}): Score = {s:.4f}")
                print(f"      Max Amount: INR {l.max_loan_amount:,.0f} | Min Rate: {l.min_interest_rate}% | Max Tenure: {l.max_tenure}m | Cap: INR {l.available_capacity:,.0f}")
                print(f"      Score Breakdown: Interest={bd['interest']:.2f}, Amount={bd['loan_amount']:.2f}, Tenure={bd['tenure']:.2f}, Collateral={bd['collateral']:.2f}, Capacity={bd['capacity']:.2f}")

        print("\nSeeding and verification completed successfully!")

    except Exception as e:
        db.rollback()
        print(f"\nError occurred during seeding: {e}", file=sys.stderr)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_lenders()
