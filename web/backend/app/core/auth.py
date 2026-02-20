"""身份认证和授权模块

实现 JWT token 认证
"""
from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# JWT 配置
SECRET_KEY = "your-secret-key-change-this-in-production"  # 生产环境应从环境变量读取
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# HTTP Bearer 认证
security = HTTPBearer()


class TokenData(BaseModel):
    """Token 数据模型"""
    username: str | None = None
    user_id: int | None = None


class User(BaseModel):
    """用户模型"""
    id: int
    username: str
    email: str | None = None
    disabled: bool = False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码

    Args:
        plain_password: 明文密码
        hashed_password: 哈希后的密码

    Returns:
        bool: 密码是否匹配
    """
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """生成密码哈希

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码

    注意：bcrypt 限制密码最大长度为 72 字节
    """
    # bcrypt 限制密码长度为 72 字节
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """创建访问 token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """解码访问 token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭证",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return TokenData(username=username, user_id=user_id)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """获取当前用户（依赖注入）"""
    token = credentials.credentials
    token_data = decode_access_token(token)
    return token_data


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """获取当前活跃用户"""
    # 这里可以添加额外的用户状态检查
    # 例如从数据库查询用户是否被禁用
    return current_user


# 可选的依赖项（用于不强制认证的端点）
class OptionalAuth:
    """可选认证"""
    async def __call__(
        self, credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False))
    ) -> TokenData | None:
        if credentials is None:
            return None
        try:
            token = credentials.credentials
            return decode_access_token(token)
        except HTTPException:
            return None


optional_auth = OptionalAuth()
