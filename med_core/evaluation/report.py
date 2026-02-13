"""
Evaluation report generator.

Generates comprehensive markdown reports for medical ML experiments,
aggregating metrics, visualization links, and configuration details.

This module now uses a modular architecture with separate components
for metrics calculation, visualization, and report generation.
"""

from pathlib import Path
from typing import Any

from med_core.evaluation.report_generator import (
    ReportGenerator,
    generate_evaluation_report,
)

# Re-export for backward compatibility
__all__ = [
    "ReportGenerator",
    "generate_evaluation_report",
    # Legacy alias
    "EvaluationReport",
]

# Backward compatibility alias
EvaluationReport = ReportGenerator

