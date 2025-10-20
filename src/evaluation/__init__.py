"""
Evaluation modules for Radiology Report Generation project.

This package contains:
- Model validation and testing
- Performance metrics calculation
- Evaluation report generation
"""

# Import only available modules
try:
    from .validator import ModelValidator
except ImportError:
    ModelValidator = None

try:
    from .metrics_calculator import MetricsCalculator
except ImportError:
    MetricsCalculator = None

try:
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    PerformanceAnalyzer = None

__all__ = [
    'ModelValidator',
    'MetricsCalculator', 
    'PerformanceAnalyzer'
]
