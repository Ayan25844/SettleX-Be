import argparse
import getpass
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database import SessionLocal, init_db
from models.user import User, UserRole
from services.auth import hash_password


def prompt_admin_creation(
    name: str | None = None,
    email: str | None = None,
    password: str | None = None
) -> None:
    """Prompt for admin credentials and persist an admin user to the database."""
    init_db()
    db = SessionLocal()

    try:
        # Prompt for full name
        if not name:
            name = input("Enter admin full name: ").strip()
        else:
            name = name.strip()

        if not name:
            print("Error: Full name cannot be empty.", file=sys.stderr)
            sys.exit(1)

        # Prompt for email
        if not email:
            email = input("Enter admin email: ").strip()
        else:
            email = email.strip()

        email = email.lower()

        if not email or "@" not in email:
            print("Error: A valid email address is required.", file=sys.stderr)
            sys.exit(1)

        # Reject duplicate email
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            print(f"Error: Email '{email}' is already registered.", file=sys.stderr)
            sys.exit(1)

        # Prompt for password securely
        if not password:
            while True:
                password = getpass.getpass("Enter admin password (min 8 chars): ")
                if len(password) < 8:
                    print("Password must be at least 8 characters long. Please try again.")
                    continue
                confirm_password = getpass.getpass("Confirm admin password: ")
                if password != confirm_password:
                    print("Passwords do not match. Please try again.")
                    continue
                break
        else:
            if len(password) < 8:
                print("Error: Password must be at least 8 characters long.", file=sys.stderr)
                sys.exit(1)

        # Hash password and create admin user
        password_hash = hash_password(password)
        admin_user = User(
            full_name=name,
            email=email,
            password_hash=password_hash,
            role=UserRole.ADMIN,
            is_active=True
        )

        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)

        print(f"\nAdmin user created successfully!")
        print(f"ID:    {admin_user.id}")
        print(f"Name:  {admin_user.full_name}")
        print(f"Email: {admin_user.email}")
        print(f"Role:  {admin_user.role.value if hasattr(admin_user.role, 'value') else admin_user.role}")

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Create an admin user for SettleX")
    parser.add_argument("--name", help="Admin full name (optional, prompts if omitted)")
    parser.add_argument("--email", help="Admin email (optional, prompts if omitted)")
    parser.add_argument("--password", help="Admin password (optional, prompts securely if omitted)")
    args = parser.parse_args()

    prompt_admin_creation(name=args.name, email=args.email, password=args.password)


if __name__ == "__main__":
    main()
