from routers.auth import router as auth_router
from routers.admin import router as admin_router
from routers.test_auth import router as test_auth_router
from routers.profiles import borrower_router, lender_router
from routers.matching import router as matching_router

__all__ = [
    "auth_router",
    "admin_router",
    "test_auth_router",
    "borrower_router",
    "lender_router",
    "matching_router",
]
