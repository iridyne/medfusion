"""MedFusion Web UI 安装配置

使用方式:
    pip install -e .                    # 开发模式安装
    pip install .                       # 正式安装

安装后可以使用以下命令:
    medfusion-web start                 # 启动服务
    medfusion-web stop                  # 停止服务
    medfusion-web status                # 查看状态
    medfusion-web logs                  # 查看日志
    medfusion-web init                  # 初始化环境
"""

from pathlib import Path

from setuptools import find_packages, setup

# 读取 README
readme_file = Path(__file__).parent.parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# 读取依赖
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="medfusion-web",
    version="0.1.0",
    description="MedFusion 医学深度学习框架 Web 界面",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MedFusion Team",
    author_email="medfusion@example.com",
    url="https://github.com/your-org/medfusion",
    license="MIT",
    # 包配置
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    zip_safe=False,
    # Python 版本要求
    python_requires=">=3.8",
    # 依赖项
    install_requires=requirements,
    # 额外依赖
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "httpx>=0.26.0",
        ],
    },
    # CLI 命令入口点
    entry_points={
        "console_scripts": [
            "medfusion-web=app.cli:cli",
        ],
    },
    # 分类信息
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    # 关键词
    keywords=[
        "medical imaging",
        "deep learning",
        "machine learning",
        "web ui",
        "fastapi",
        "react",
    ],
    # 项目链接
    project_urls={
        "Bug Reports": "https://github.com/your-org/medfusion/issues",
        "Source": "https://github.com/your-org/medfusion",
        "Documentation": "https://medfusion.readthedocs.io",
    },
)
