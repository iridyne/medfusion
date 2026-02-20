"""预处理服务

提供图像预处理功能，集成 med_core.preprocessing.ImagePreprocessor
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from med_core.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class PreprocessingService:
    """预处理服务

    提供异步图像预处理功能，支持进度回调和任务控制
    """

    def __init__(self):
        """初始化预处理服务"""
        self._tasks: Dict[str, asyncio.Task] = {}
        self._should_cancel: Dict[str, bool] = {}
        logger.info("PreprocessingService initialized")

    async def start_preprocessing(
        self,
        task_id: str,
        input_dir: str,
        output_dir: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """启动预处理任务

        Args:
            task_id: 任务 ID
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            config: 预处理配置
                - size: int (目标图像大小，默认 224)
                - normalize: str (归一化方法: minmax, zscore, percentile, none)
                - remove_artifacts: bool (是否去除伪影)
                - enhance_contrast: bool (是否增强对比度)
            progress_callback: 进度回调函数

        Returns:
            处理结果字典

        Raises:
            ValueError: 输入目录不存在或没有找到图像
        """
        logger.info(f"Starting preprocessing task {task_id}")
        logger.info(f"Input: {input_dir}, Output: {output_dir}")
        logger.info(f"Config: {config}")

        # 验证输入目录
        input_path = Path(input_dir)
        if not input_path.exists():
            error_msg = f"Input directory not found: {input_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not input_path.is_dir():
            error_msg = f"Input path is not a directory: {input_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")

        # 获取所有图像文件
        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))

        # 去重
        image_paths = list(set(image_paths))

        if not image_paths:
            error_msg = f"No valid images found in {input_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        total_images = len(image_paths)
        logger.info(f"Found {total_images} images to process")

        # 通知开始
        if progress_callback:
            await progress_callback(
                {
                    "type": "started",
                    "task_id": task_id,
                    "total_images": total_images,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # 创建预处理器
        try:
            preprocessor = ImagePreprocessor(
                normalize_method=config.get("normalize", "percentile"),
                remove_watermark=config.get("remove_artifacts", False),
                apply_clahe=config.get("enhance_contrast", False),
                output_size=(config.get("size", 224), config.get("size", 224)),
            )
            logger.info("ImagePreprocessor created successfully")
        except Exception as e:
            error_msg = f"Failed to create preprocessor: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 初始化计数器
        processed_images = 0
        failed_images = 0
        failed_files = []
        processed_files = []

        # 处理图像
        start_time = datetime.now(timezone.utc)

        for i, image_path in enumerate(image_paths):
            # 检查是否取消
            if self._should_cancel.get(task_id, False):
                logger.info(f"Task {task_id} cancelled by user")
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "cancelled",
                            "task_id": task_id,
                            "processed_images": processed_images,
                            "failed_images": failed_images,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                # 清理取消标志
                del self._should_cancel[task_id]

                return {
                    "status": "cancelled",
                    "total_images": total_images,
                    "processed_images": processed_images,
                    "failed_images": failed_images,
                    "processed_files": processed_files[:100],
                    "failed_files": failed_files,
                }

            try:
                # 处理单个图像
                output_file = output_path / image_path.name
                processed_image = preprocessor(str(image_path))
                processed_image.save(str(output_file))
                processed_images += 1
                processed_files.append(str(image_path.name))

                logger.debug(f"Processed {image_path.name} ({i + 1}/{total_images})")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                failed_images += 1
                failed_files.append(
                    {
                        "file": str(image_path.name),
                        "error": str(e),
                    }
                )

            # 更新进度
            progress = (i + 1) / total_images

            # 每处理 10 张图像或最后一张时发送进度更新
            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "progress",
                            "task_id": task_id,
                            "progress": progress,
                            "processed_images": processed_images,
                            "failed_images": failed_images,
                            "current_file": image_path.name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            # 让出控制权，避免阻塞事件循环
            await asyncio.sleep(0)

        # 计算处理时间
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Task {task_id} completed in {duration:.2f}s")
        logger.info(f"Processed: {processed_images}, Failed: {failed_images}")

        # 完成
        result = {
            "status": "completed",
            "total_images": total_images,
            "processed_images": processed_images,
            "failed_images": failed_images,
            "processed_files": processed_files[:100],  # 限制返回数量
            "failed_files": failed_files,
            "duration": duration,
            "output_dir": str(output_path),
        }

        if progress_callback:
            await progress_callback(
                {
                    "type": "completed",
                    "task_id": task_id,
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # 清理任务记录
        if task_id in self._tasks:
            del self._tasks[task_id]

        return result

    def cancel_task(self, task_id: str) -> bool:
        """取消任务

        Args:
            task_id: 任务 ID

        Returns:
            是否成功设置取消标志
        """
        if task_id in self._tasks:
            logger.info(f"Cancelling task {task_id}")
            self._should_cancel[task_id] = True
            return True

        logger.warning(f"Task {task_id} not found for cancellation")
        return False

    def get_task_status(self, task_id: str) -> Optional[str]:
        """获取任务状态

        Args:
            task_id: 任务 ID

        Returns:
            任务状态 (running, completed) 或 None
        """
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.done():
                return "completed"
            return "running"
        return None

    def register_task(self, task_id: str, task: asyncio.Task) -> None:
        """注册任务

        Args:
            task_id: 任务 ID
            task: asyncio 任务对象
        """
        self._tasks[task_id] = task
        logger.info(f"Task {task_id} registered")

    def cleanup_task(self, task_id: str) -> None:
        """清理任务记录

        Args:
            task_id: 任务 ID
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
        if task_id in self._should_cancel:
            del self._should_cancel[task_id]
        logger.info(f"Task {task_id} cleaned up")


# 全局服务实例
preprocessing_service = PreprocessingService()
