"""
DICOM Image Loader with Medical-Grade Processing

Provides specialized loading and preprocessing for DICOM medical images:
- HU (Hounsfield Unit) conversion with Rescale Intercept/Slope
- CT Windowing (window level and width)
- Proper handling of medical metadata
"""

import logging
from operator import itemgetter
from pathlib import Path
from typing import Literal

import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


# Standard CT window presets
WINDOW_PRESETS = {
    "lung": {"center": -600, "width": 1500},
    "mediastinum": {"center": 50, "width": 350},
    "bone": {"center": 400, "width": 1800},
    "soft_tissue": {"center": 40, "width": 400},
    "brain": {"center": 40, "width": 80},
    "liver": {"center": 30, "width": 150},
    "abdomen": {"center": 60, "width": 400},
}


class DICOMLoader:
    """
    DICOM file loader with medical-grade processing.

    Handles:
    - HU value conversion (Rescale Intercept/Slope)
    - CT windowing for proper visualization
    - Metadata extraction

    Example:
        >>> loader = DICOMLoader(window_preset="soft_tissue")
        >>> image = loader.load("path/to/scan.dcm")
        >>> # Returns normalized numpy array ready for model input
    """

    def __init__(
        self,
        window_preset: str | None = "soft_tissue",
        window_center: float | None = None,
        window_width: float | None = None,
        output_range: tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize DICOM loader.

        Args:
            window_preset: Preset name from WINDOW_PRESETS
            window_center: Custom window center (overrides preset)
            window_width: Custom window width (overrides preset)
            output_range: Output value range after normalization
        """
        # Determine windowing parameters
        if window_center is not None and window_width is not None:
            self.window_center = window_center
            self.window_width = window_width
        elif window_preset and window_preset in WINDOW_PRESETS:
            preset = WINDOW_PRESETS[window_preset]
            self.window_center = preset["center"]
            self.window_width = preset["width"]
        else:
            # Default to soft tissue
            self.window_center = 40
            self.window_width = 400

        self.output_range = output_range

    def load(self, path: str | Path) -> np.ndarray:
        """
        Load DICOM file and convert to HU values with windowing.

        Args:
            path: Path to DICOM file

        Returns:
            Normalized numpy array (H, W) in output_range
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path}")

        # Read DICOM
        try:
            ds = pydicom.dcmread(str(path))
        except Exception as e:
            raise ValueError(f"Failed to read DICOM file {path}: {e}") from e

        # Get pixel array
        pixel_array = ds.pixel_array.astype(np.float32)

        # Convert to HU values
        hu_array = self._convert_to_hu(ds, pixel_array)

        # Apply windowing
        windowed = self._apply_window(hu_array)

        return windowed

    def _convert_to_hu(
        self, ds: pydicom.Dataset, pixel_array: np.ndarray,
    ) -> np.ndarray:
        """
        Convert pixel values to Hounsfield Units.

        HU = pixel_value * slope + intercept
        """
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slope = float(getattr(ds, "RescaleSlope", 1.0))

        hu_array = pixel_array * slope + intercept

        logger.debug(
            f"HU conversion: slope={slope}, intercept={intercept}, "
            f"range=[{hu_array.min():.1f}, {hu_array.max():.1f}]",
        )

        return hu_array

    def _apply_window(self, hu_array: np.ndarray) -> np.ndarray:
        """
        Apply CT windowing and normalize to output range.

        Clips HU values to [center - width/2, center + width/2]
        then normalizes to output_range.
        """
        lower = self.window_center - self.window_width / 2
        upper = self.window_center + self.window_width / 2

        # Clip to window
        windowed = np.clip(hu_array, lower, upper)

        # Normalize to [0, 1]
        normalized = (windowed - lower) / (upper - lower)

        # Scale to output range
        out_min, out_max = self.output_range
        scaled = normalized * (out_max - out_min) + out_min

        return scaled

    def load_as_pil(
        self, path: str | Path, mode: Literal["L", "RGB"] = "L",
    ) -> Image.Image:
        """
        Load DICOM and return as PIL Image.

        Args:
            path: Path to DICOM file
            mode: PIL image mode ("L" for grayscale, "RGB" for 3-channel)

        Returns:
            PIL Image
        """
        array = self.load(path)

        # Convert to uint8
        uint8_array = (array * 255).astype(np.uint8)

        # Create PIL image
        image = Image.fromarray(uint8_array, mode="L")

        if mode == "RGB":
            image = image.convert("RGB")

        return image

    def extract_metadata(self, path: str | Path) -> dict:
        """
        Extract useful metadata from DICOM file.

        Returns:
            Dictionary with metadata fields
        """
        ds = pydicom.dcmread(str(path))

        metadata = {
            "patient_id": getattr(ds, "PatientID", None),
            "study_date": getattr(ds, "StudyDate", None),
            "modality": getattr(ds, "Modality", None),
            "slice_thickness": getattr(ds, "SliceThickness", None),
            "pixel_spacing": getattr(ds, "PixelSpacing", None),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "rescale_intercept": getattr(ds, "RescaleIntercept", 0.0),
            "rescale_slope": getattr(ds, "RescaleSlope", 1.0),
        }

        return metadata


def load_dicom_series(
    directory: str | Path,
    window_preset: str = "soft_tissue",
    sort_by: Literal["instance", "position"] = "instance",
) -> np.ndarray:
    """
    Load a series of DICOM files from a directory.

    Args:
        directory: Directory containing DICOM files
        window_preset: Window preset to apply
        sort_by: How to sort slices ("instance" or "position")

    Returns:
        3D numpy array (slices, height, width)
    """
    directory = Path(directory)
    dicom_files = sorted(directory.glob("*.dcm"))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    loader = DICOMLoader(window_preset=window_preset)

    # Load all slices
    slices = []
    metadata = []

    for dcm_file in dicom_files:
        try:
            slice_array = loader.load(dcm_file)
            slices.append(slice_array)

            # Get position for sorting
            ds = pydicom.dcmread(str(dcm_file))
            if sort_by == "instance":
                pos = int(getattr(ds, "InstanceNumber", 0))
            else:
                pos = float(getattr(ds, "ImagePositionPatient", [0, 0, 0])[2])
            metadata.append((pos, slice_array))
        except Exception as e:
            logger.warning(f"Failed to load {dcm_file}: {e}")
            continue

    # Sort by position
    metadata.sort(key=itemgetter(0))
    sorted_slices = [s for _, s in metadata]

    # Stack into 3D array
    volume = np.stack(sorted_slices, axis=0)

    logger.info(f"Loaded DICOM series: {volume.shape} from {len(dicom_files)} files")

    return volume
