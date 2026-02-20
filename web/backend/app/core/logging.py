"""结构化日志系统

提供 JSON 格式的日志输出，便于日志分析和监控
"""
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON 格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON"""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加额外的字段
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """结构化日志器"""

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Path | None = None,
        use_json: bool = True,
    ):
        """初始化日志器

        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径（可选）
            use_json: 是否使用 JSON 格式
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # 清除已有的处理器

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if use_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )

        self.logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """记录 DEBUG 级别日志"""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """记录 INFO 级别日志"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """记录 WARNING 级别日志"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """记录 ERROR 级别日志"""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """记录 CRITICAL 级别日志"""
        self._log(logging.CRITICAL, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        """内部日志方法"""
        extra = {}

        # 提取特殊字段
        if "user_id" in kwargs:
            extra["user_id"] = kwargs.pop("user_id")

        if "request_id" in kwargs:
            extra["request_id"] = kwargs.pop("request_id")

        # 其他字段作为 extra
        if kwargs:
            extra["extra"] = kwargs

        self.logger.log(level, message, extra=extra)


# 创建默认日志器
def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> StructuredLogger:
    """获取日志器

    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径

    Returns:
        结构化日志器实例
    """
    return StructuredLogger(name, level, log_file)


# 应用日志器
app_logger = get_logger(
    "medfusion.app",
    level=logging.INFO,
    log_file=Path("./logs/app.log"),
)

# API 日志器
api_logger = get_logger(
    "medfusion.api",
    level=logging.INFO,
    log_file=Path("./logs/api.log"),
)

# 数据库日志器
db_logger = get_logger(
    "medfusion.db",
    level=logging.WARNING,
    log_file=Path("./logs/db.log"),
)
