"""认证 API 端点"""
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.core.auth import (
    create_access_token,
    get_password_hash,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from app.core.database import get_db

router = APIRouter()


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class UserRegister(BaseModel):
    """用户注册请求"""
    username: str
    password: str
    email: Optional[EmailStr] = None


class Token(BaseModel):
    """Token 响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class UserResponse(BaseModel):
    """用户响应"""
    id: int
    username: str
    email: Optional[str] = None


# 临时用户存储（生产环境应使用数据库）
fake_users_db = {
    "admin": {
        "id": 1,
        "username": "admin",
        "email": "admin@medfusion.com",
        "hashed_password": get_password_hash("admin123"),  # 默认密码
        "disabled": False,
    }
}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """认证用户"""
    user = fake_users_db.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


@router.post("/login", response_model=Token)
async def login(user_login: UserLogin):
    """用户登录
    
    默认账号：
    - 用户名: admin
    - 密码: admin123
    """
    user = authenticate_user(user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 创建访问 token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/register", response_model=UserResponse)
async def register(user_register: UserRegister):
    """用户注册"""
    # 检查用户是否已存在
    if user_register.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 创建新用户
    user_id = len(fake_users_db) + 1
    hashed_password = get_password_hash(user_register.password)
    
    fake_users_db[user_register.username] = {
        "id": user_id,
        "username": user_register.username,
        "email": user_register.email,
        "hashed_password": hashed_password,
        "disabled": False,
    }
    
    return UserResponse(
        id=user_id,
        username=user_register.username,
        email=user_register.email
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    token_data = Depends(lambda: None)  # 这里需要导入 get_current_user
):
    """获取当前用户信息"""
    from app.core.auth import get_current_user
    from fastapi import Depends as FastAPIDepends
    
    # 这是一个占位符，实际实现需要重构
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="此功能尚未完全实现"
    )
