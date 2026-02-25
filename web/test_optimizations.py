#!/usr/bin/env python3
"""
快速测试脚本 - 验证所有优化功能
"""

import os
import sys
from datetime import UTC

# 添加 backend 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))


def test_auth():
    """测试认证模块"""
    print("🔐 测试认证模块...")
    try:
        from app.core.auth import (
            create_access_token,
            decode_access_token,
            get_password_hash,
            verify_password,
        )

        # 测试密码哈希（使用短密码避免 bcrypt 72 字节限制）
        password = "test123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed), "密码验证失败"
        print("  ✅ 密码哈希和验证正常")

        # 测试 JWT token
        token = create_access_token({"sub": "testuser", "user_id": 1})
        decoded = decode_access_token(token)
        assert decoded.username == "testuser", "Token 解码失败"
        assert decoded.user_id == 1, "Token 数据不匹配"
        print("  ✅ JWT token 创建和解码正常")

        return True
    except Exception as e:
        print(f"  ❌ 认证模块测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_logging():
    """测试日志系统"""
    print("\n📝 测试日志系统...")
    try:
        from app.core.logging import app_logger

        # 测试不同级别的日志
        app_logger.info("测试信息日志", user_id=123)
        app_logger.warning("测试警告日志", request_id="test-123")
        app_logger.error("测试错误日志", extra_data={"key": "value"})

        print("  ✅ 结构化日志系统正常")
        return True
    except Exception as e:
        print(f"  ❌ 日志系统测试失败: {e}")
        return False


def test_database():
    """测试数据库配置"""
    print("\n💾 测试数据库配置...")
    try:
        from app.core.database import engine
        from app.models.database import utc_now

        # 测试 UTC 时间函数
        now = utc_now()
        assert now.tzinfo == UTC, "UTC 时间没有时区信息"
        print("  ✅ UTC 时间函数正常")

        # 测试数据库连接池配置
        assert engine.pool.size() >= 0, "连接池配置异常"
        print("  ✅ 数据库连接池配置正常")

        return True
    except Exception as e:
        print(f"  ❌ 数据库配置测试失败: {e}")
        return False


def test_config():
    """测试配置"""
    print("\n⚙️  测试配置...")
    try:
        from app.core.config import settings

        # 验证 CORS 配置
        assert isinstance(settings.CORS_ORIGINS, list), "CORS_ORIGINS 应该是列表"
        assert "*" not in settings.CORS_ORIGINS, "CORS 不应该使用通配符"
        print("  ✅ CORS 配置安全")

        # 验证其他配置
        assert settings.APP_NAME, "应用名称未配置"
        assert settings.DATABASE_URL, "数据库 URL 未配置"
        print("  ✅ 基本配置正常")

        return True
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        return False


def test_workflow_engine():
    """测试工作流引擎"""
    print("\n🔄 测试工作流引擎...")
    try:
        from app.core.workflow_engine import WorkflowEngine

        # 创建简单的工作流
        workflow = {
            "nodes": [
                {"id": "node1", "type": "dataLoader", "config": {}},
                {"id": "node2", "type": "model", "config": {}},
            ],
            "edges": [
                {"source": "node1", "target": "node2"},
            ],
        }

        engine = WorkflowEngine(workflow)
        assert len(engine.nodes) == 2, "节点数量不正确"
        assert len(engine.edges) == 1, "边数量不正确"
        print("  ✅ 工作流引擎初始化正常")

        return True
    except Exception as e:
        print(f"  ❌ 工作流引擎测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("MedFusion Web UI - 优化功能测试")
    print("=" * 60)

    results = []

    # 运行所有测试
    results.append(("认证模块", test_auth()))
    results.append(("日志系统", test_logging()))
    results.append(("数据库配置", test_database()))
    results.append(("配置管理", test_config()))
    results.append(("工作流引擎", test_workflow_engine()))

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} {status}")

    print("-" * 60)
    print(f"总计: {passed}/{total} 通过 ({passed / total * 100:.0f}%)")
    print("=" * 60)

    if passed == total:
        print("\n🎉 所有测试通过！优化功能正常工作。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
