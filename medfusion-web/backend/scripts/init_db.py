#!/usr/bin/env python3
"""数据库初始化脚本"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.database import init_db


def main():
    """初始化数据库"""
    print(f"Initializing database at: {settings.DATABASE_URL}")

    try:
        init_db()
        print("✅ Database initialized successfully!")
        print(f"   Database file: {settings.DATABASE_URL.replace('sqlite:///', '')}")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
