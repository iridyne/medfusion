"""Auth and RBAC helpers for MedFusion Web APIs."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from fastapi import HTTPException, Request, status

from .config import settings


class Role(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"


_ROLE_LEVELS: dict[str, int] = {
    Role.VIEWER.value: 1,
    Role.OPERATOR.value: 2,
    Role.ADMIN.value: 3,
}
_PERMISSION_LEVELS: dict[str, int] = {
    Permission.READ.value: 1,
    Permission.WRITE.value: 2,
}

_PUBLIC_API_PATHS: tuple[str, ...] = (
    "/api/auth/token",
    "/api/system/version",
    "/api/system/features",
)
_JWT_ALGORITHM = "HS256"


@dataclass(slots=True)
class Principal:
    subject: str
    role: str
    token_type: str


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _import_pyjwt() -> Any | None:
    try:
        import jwt as pyjwt
    except ImportError:
        return None
    return pyjwt


def is_jwt_runtime_available() -> bool:
    return _import_pyjwt() is not None


def normalize_role(role: str | None) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in _ROLE_LEVELS:
        return normalized
    return Role.ADMIN.value


def role_allows_permission(role: str, permission: str) -> bool:
    role_level = _ROLE_LEVELS.get(normalize_role(role), _ROLE_LEVELS[Role.ADMIN.value])
    required_level = _PERMISSION_LEVELS.get(permission, _PERMISSION_LEVELS[Permission.WRITE.value])
    return role_level >= required_level


def create_access_token(
    *,
    subject: str,
    role: str | None = None,
    expires_minutes: int | None = None,
) -> str:
    pyjwt = _import_pyjwt()
    if pyjwt is None:
        raise RuntimeError(
            "PyJWT runtime is unavailable. Install web extras to enable JWT mode."
        )

    ttl_minutes = expires_minutes or settings.auth_access_token_expire_minutes
    payload: dict[str, Any] = {
        "sub": subject,
        "role": normalize_role(role or settings.auth_default_role),
        "exp": _utcnow() + timedelta(minutes=ttl_minutes),
    }
    return pyjwt.encode(payload, settings.secret_key, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str) -> Principal:
    pyjwt = _import_pyjwt()
    if pyjwt is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "auth_jwt_runtime_unavailable",
                "message": "当前环境缺少 PyJWT，JWT 模式不可用。",
            },
        )

    try:
        payload = pyjwt.decode(token, settings.secret_key, algorithms=[_JWT_ALGORITHM])
    except pyjwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "auth_invalid_token",
                "message": "访问令牌无效或已过期。",
            },
        ) from exc

    subject = str(payload.get("sub") or "").strip() or "unknown"
    role = normalize_role(str(payload.get("role") or settings.auth_default_role))
    return Principal(subject=subject, role=role, token_type="jwt")


def verify_bootstrap_credentials(*, username: str, password: str) -> bool:
    if settings.auth_password is None:
        return False
    if username != settings.auth_username:
        return False
    return secrets.compare_digest(password, settings.auth_password)


def _is_public_api_path(path: str) -> bool:
    return any(path == allowed or path.startswith(f"{allowed}/") for allowed in _PUBLIC_API_PATHS)


def _required_permission(method: str) -> str:
    if method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return Permission.READ.value
    return Permission.WRITE.value


def _resolve_principal(token: str) -> Principal:
    if settings.auth_token and secrets.compare_digest(token, settings.auth_token):
        return Principal(
            subject="static-token",
            role=Role.ADMIN.value,
            token_type="static",
        )
    return decode_access_token(token)


def enforce_request_auth(request: Request) -> Principal | None:
    if not settings.auth_enabled:
        return None

    path = request.url.path
    if not path.startswith("/api/") or _is_public_api_path(path):
        return None

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "auth_missing_bearer_token",
                "message": "缺少 Bearer token，请在 Authorization 头中提供。",
            },
        )

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "auth_empty_bearer_token",
                "message": "Bearer token 为空。",
            },
        )

    principal = _resolve_principal(token)
    permission = _required_permission(request.method)
    if not role_allows_permission(principal.role, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "auth_permission_denied",
                "message": "当前角色无权限执行该操作。",
                "required_permission": permission,
                "role": principal.role,
            },
        )

    request.state.principal = principal
    return principal
