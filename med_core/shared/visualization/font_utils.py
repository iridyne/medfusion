"""Matplotlib font configuration helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import matplotlib
from matplotlib import font_manager

logger = logging.getLogger(__name__)

_FONT_PATH_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("SourceHanSansCN-Regular.otf", "SourceHanSansCN", "sourcehansanscn"),
    ("NotoSansCJK-Regular.ttc", "NotoSansCJK", "notosanscjk"),
    ("NotoSansSC", "notosanssc"),
    ("WenQuanYiZenHei", "wqy-zenhei", "wqy"),
    ("MicrosoftYaHei", "msyh", "yahei"),
    ("SimHei", "simhei"),
    ("PingFang", "pingfang"),
    ("ArialUnicode", "arialuni"),
)

_FONT_FAMILY_CANDIDATES: tuple[str, ...] = (
    "Source Han Sans CN",
    "思源黑体 CN",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


@lru_cache(maxsize=1)
def _discover_cjk_font() -> tuple[str | None, str | None]:
    system_fonts = sorted({Path(path) for path in font_manager.findSystemFonts()})

    for keywords in _FONT_PATH_KEYWORDS:
        for font_path in system_fonts:
            path_text = str(font_path).lower()
            if not any(keyword.lower() in path_text for keyword in keywords):
                continue

            try:
                font_manager.fontManager.addfont(str(font_path))
                family_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                return str(font_path), family_name
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.debug("Failed to register CJK font %s: %s", font_path, exc)

    installed_families = {font.name for font in font_manager.fontManager.ttflist}
    for family_name in _FONT_FAMILY_CANDIDATES:
        if family_name in installed_families:
            return None, family_name

    return None, None


@lru_cache(maxsize=1)
def configure_matplotlib_fonts() -> str | None:
    """Configure matplotlib to prefer a CJK-capable sans-serif font."""
    font_path, family_name = _discover_cjk_font()

    current_sans = list(matplotlib.rcParams.get("font.sans-serif", []))
    if family_name and family_name not in current_sans:
        matplotlib.rcParams["font.sans-serif"] = [family_name, *current_sans]
    elif current_sans:
        matplotlib.rcParams["font.sans-serif"] = current_sans

    matplotlib.rcParams["font.family"] = ["sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    if family_name:
        logger.info(
            "Configured matplotlib CJK font: %s%s",
            family_name,
            f" ({font_path})" if font_path else "",
        )
    else:
        logger.warning(
            "No preferred CJK font found for matplotlib; Chinese glyph rendering may degrade.",
        )

    return family_name
