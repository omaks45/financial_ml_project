# ml/financial_metrics_calculator.py
"""
This module imports the FinancialMetricsCalculator from data.financial_metrics
to maintain proper separation of concerns while allowing flexible imports.
"""

from data.financial_metrics import FinancialMetricsCalculator

# Re-export for backward compatibility
__all__ = ['FinancialMetricsCalculator']