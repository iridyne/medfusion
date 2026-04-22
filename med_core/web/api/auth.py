"""Authentication API for issuing JWT access tokens."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..auth import (
    create_access_token,
    is_jwt_runtime_available,
    normalize_role,
    verify_bootstrap_credentials,
)
from ..config import settings

router = APIRouter()


class TokenRequest(BaseModel):
    username: str = Field(..., min_length=1, description="登录用户名")
    password: str = Field(..., min_length=1, description="登录密码")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str


@router.post("/token", response_model=TokenResponse)
async def issue_token(request: TokenRequest) -> TokenResponse:
    """Issue JWT token when auth is enabled and bootstrap credentials are configured."""
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "auth_disabled",
                "message": "认证未启用，请先开启 MEDFUSION_AUTH_ENABLED。",
            },
        )

    if not is_jwt_runtime_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "auth_jwt_runtime_unavailable",
                "message": "当前环境缺少 PyJWT，JWT token 发放不可用。",
            },
        )

    if settings.auth_password is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "auth_password_not_configured",
                "message": (
                    "未配置 auth_password，无法发放 JWT。"
                    "请设置 MEDFUSION_AUTH_PASSWORD，或使用静态 token 方式。"
                ),
            },
        )

    if not verify_bootstrap_credentials(
        username=request.username,
        password=request.password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "auth_invalid_credentials",
                "message": "用户名或密码错误。",
            },
        )

    role = normalize_role(settings.auth_default_role)
    expires_in_minutes = settings.auth_access_token_expire_minutes
    try:
        token = create_access_token(
            subject=request.username,
            role=role,
            expires_minutes=expires_in_minutes,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "auth_jwt_runtime_unavailable",
                "message": str(exc),
            },
        ) from exc

    return TokenResponse(
        access_token=token,
        expires_in_minutes=expires_in_minutes,
        role=role,
    )
