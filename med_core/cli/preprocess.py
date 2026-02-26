"""Preprocessing command implementation."""

import argparse
import logging
import sys
from pathlib import Path

from med_core.preprocessing import ImagePreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def preprocess() -> None:
    """Command-line entry point for image preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess medical images")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory",
    )
    parser.add_argument("--size", type=int, default=224, help="Target image size")
    parser.add_argument(
        "--normalize",
        type=str,
        default="percentile",
        choices=["minmax", "zscore", "percentile", "none"],
        help="Normalization method",
    )
    parser.add_argument(
        "--remove-artifacts", action="store_true", help="Remove artifacts",
    )
    parser.add_argument(
        "--enhance-contrast", action="store_true", help="Enhance contrast",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    if args.size <= 0:
        logger.error(f"Invalid image size: {args.size}")
        sys.exit(1)

    try:
        preprocessor = ImagePreprocessor(
            normalize_method=args.normalize,
            remove_watermark=args.remove_artifacts,
            apply_clahe=args.enhance_contrast,
            output_size=(args.size, args.size),
        )
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        sys.exit(1)

    image_paths = list(input_path.glob("*.*"))
    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_paths = [
        p for p in image_paths if p.suffix.lower() in valid_extensions
    ]

    if not image_paths:
        logger.error(f"No valid images found in {args.input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images in {args.input_dir}")
    try:
        preprocessor.process_batch(image_paths, args.output_dir)
    except Exception as e:
        logger.error(f"Failed to preprocess images: {e}")
        sys.exit(1)
