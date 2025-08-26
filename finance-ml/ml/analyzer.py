"""
Financial Metrics Calculation Functions

This module handles:
- Financial ratios calculation
- Growth metrics computation
- Composite financial health scores
- ML-ready feature engineering
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class FinancialMetricsCalculator:
    """
    Implementation: Financial Metrics Calculation
    
    Calculates comprehensive financial metrics and ratios from cleaned financial data.
    Prepares data for ML analysis with proper feature engineering.
    """
    
    def __init__(self):
        """Initialize financial metrics calculator"""
        self.calculation_stats = {
            'total_calculations': 0,
            'calculation_errors': 0,
            'zero_division_errors': 0
        }
        
        # Define metric categories for organized calculation
        self.metric_categories = {
            'profitability': [
                'return_on_equity', 'return_on_assets', 'profit_margin',
                'gross_margin', 'operating_margin', 'net_margin'
            ],
            'liquidity': [
                'current_ratio', 'quick_ratio', 'cash_ratio', 'operating_cash_flow_ratio'
            ],
            'leverage': [
                'debt_to_equity', 'debt_ratio', 'equity_ratio', 'times_interest_earned'
            ],
            'efficiency': [
                'asset_turnover', 'inventory_turnover', 'receivables_turnover',
                'working_capital_turnover'
            ],
            'growth': [
                'revenue_growth', 'profit_growth', 'asset_growth', 'equity_growth'
            ],
            'market': [
                'price_to_earnings', 'price_to_book', 'dividend_yield', 'earnings_per_share'
            ]
        }
        
        logger.info("FinancialMetricsCalculator initialized for Day 3 Task 2")
    
    def calculate_all_metrics(self, cleaned_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Main calculation function - computes all financial metrics
        
        Args:
            cleaned_data: Cleaned financial data from DataProcessor
            
        Returns:
            Dictionary containing all calculated financial metrics
        """
        if not cleaned_data or 'error' in cleaned_data:
            logger.warning("Invalid or error-containing data provided for metrics calculation")
            return {}
        
        try:
            logger.info("Starting comprehensive financial metrics calculation...")
            
            # Extract data sections
            financial_metrics = cleaned_data.get('financial_metrics', {})
            balance_sheet = cleaned_data.get('balance_sheet', {})
            profit_loss = cleaned_data.get('profit_loss', {})
            cash_flow = cleaned_data.get('cash_flow', {})
            ratios = cleaned_data.get('ratios', {})
            growth_metrics = cleaned_data.get('growth_metrics', {})
            
            # Initialize results dictionary
            calculated_metrics = {}
            
            # 1. Extract primary metrics from API data
            primary_metrics = self._extract_primary_metrics(
                financial_metrics, ratios, growth_metrics
            )
            calculated_metrics.update(primary_metrics)
            
            # 2. Calculate profitability ratios
            profitability_ratios = self._calculate_profitability_ratios(
                profit_loss, balance_sheet, financial_metrics
            )
            calculated_metrics.update(profitability_ratios)
            
            # 3. Calculate liquidity ratios  
            liquidity_ratios = self._calculate_liquidity_ratios(
                balance_sheet, cash_flow, financial_metrics
            )
            calculated_metrics.update(liquidity_ratios)
            
            # 4. Calculate leverage ratios
            leverage_ratios = self._calculate_leverage_ratios(
                balance_sheet, profit_loss, financial_metrics
            )
            calculated_metrics.update(leverage_ratios)
            
            # 5. Calculate efficiency ratios
            efficiency_ratios = self._calculate_efficiency_ratios(
                balance_sheet, profit_loss, financial_metrics
            )
            calculated_metrics.update(efficiency_ratios)
            
            # 6. Calculate growth metrics
            growth_calculations = self._calculate_growth_metrics(
                growth_metrics, financial_metrics
            )
            calculated_metrics.update(growth_calculations)
            
            # 7. Calculate market ratios (if data available)
            market_ratios = self._calculate_market_ratios(
                financial_metrics, ratios
            )
            calculated_metrics.update(market_ratios)
            
            # 8. Calculate composite scores
            composite_scores = self._calculate_composite_scores(calculated_metrics)
            calculated_metrics.update(composite_scores)
            
            # 9. Calculate ML features
            ml_features = self._calculate_ml_features(calculated_metrics)
            calculated_metrics.update(ml_features)
            
            # Update statistics
            self.calculation_stats['total_calculations'] += 1
            
            logger.info(f"Calculated {len(calculated_metrics)} financial metrics successfully")
            return calculated_metrics
            
        except Exception as e:
            error_msg = f"Error calculating financial metrics: {str(e)}"
            logger.error(error_msg)
            self.calculation_stats['calculation_errors'] += 1
            return {'calculation_error': error_msg}
    
    def _extract_primary_metrics(self, financial_metrics: Dict, ratios: Dict, 
                                growth_metrics: Dict) -> Dict[str, float]:
        """Extract primary metrics directly from API data"""
        primary = {}
        
        # Primary financial figures
        metric_mappings = {
            'revenue': ['revenue', 'total_revenue', 'net_sales'],
            'net_income': ['net_income', 'profit_after_tax', 'net_profit'],
            'total_assets': ['total_assets', 'assets'],
            'total_equity': ['total_equity', 'shareholders_equity', 'equity'],
            'total_debt': ['total_debt', 'debt', 'total_liabilities'],
            'current_assets': ['current_assets'],
            'current_liabilities': ['current_liabilities'],
            'cash': ['cash', 'cash_and_equivalents', 'cash_equivalents']
        }
        
        # Extract from financial_metrics first, then ratios
        for metric, possible_keys in metric_mappings.items():
            primary[metric] = self._get_value_from_sources(
                [financial_metrics, ratios], possible_keys
            )
        
        # Extract ratios directly
        ratio_mappings = {
            'roe': ['roe', 'return_on_equity'],
            'roa': ['roa', 'return_on_assets'], 
            'current_ratio': ['current_ratio'],
            'debt_to_equity': ['debt_to_equity', 'debt_equity_ratio'],
            'profit_margin': ['profit_margin', 'net_margin']
        }
        
        for ratio, possible_keys in ratio_mappings.items():
            primary[ratio] = self._get_value_from_sources(
                [ratios, financial_metrics], possible_keys
            )
        
        # Extract growth metrics
        growth_mappings = {
            'sales_growth': ['sales_growth', 'revenue_growth'],
            'profit_growth': ['profit_growth', 'net_income_growth'],
            'stock_cagr': ['stock_cagr', 'stock_return', 'share_price_cagr']
        }
        
        for growth, possible_keys in growth_mappings.items():
            primary[growth] = self._get_value_from_sources(
                [growth_metrics, financial_metrics], possible_keys
            )
        
        return primary
    
    def _calculate_profitability_ratios(self, profit_loss: Dict, balance_sheet: Dict,
                                      financial_metrics: Dict) -> Dict[str, float]:
        """Calculate profitability ratios"""
        ratios = {}
        
        try:
            # Get base values
            revenue = self._get_value_from_sources([profit_loss, financial_metrics], 
                                                 ['revenue', 'net_sales', 'total_revenue'])
            net_income = self._get_value_from_sources([profit_loss, financial_metrics],
                                                    ['net_income', 'profit_after_tax'])
            gross_profit = self._get_value_from_sources([profit_loss], 
                                                       ['gross_profit', 'gross_income'])
            operating_income = self._get_value_from_sources([profit_loss],
                                                          ['operating_income', 'ebit'])
            total_assets = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                       ['total_assets', 'assets'])
            shareholders_equity = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                             ['shareholders_equity', 'total_equity'])
            
            # Calculate ratios with zero-division protection
            ratios['calculated_profit_margin'] = self._safe_divide(net_income, revenue) * 100
            ratios['calculated_gross_margin'] = self._safe_divide(gross_profit, revenue) * 100
            ratios['calculated_operating_margin'] = self._safe_divide(operating_income, revenue) * 100
            ratios['calculated_roa'] = self._safe_divide(net_income, total_assets) * 100
            ratios['calculated_roe'] = self._safe_divide(net_income, shareholders_equity) * 100
            
            # Asset utilization ratios
            ratios['revenue_per_asset'] = self._safe_divide(revenue, total_assets)
            ratios['income_per_equity'] = self._safe_divide(net_income, shareholders_equity)
            
        except Exception as e:
            logger.warning(f"Error calculating profitability ratios: {e}")
            ratios['profitability_error'] = str(e)
        
        return ratios
    
    def _calculate_liquidity_ratios(self, balance_sheet: Dict, cash_flow: Dict,
                                  financial_metrics: Dict) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        ratios = {}
        
        try:
            # Get base values
            current_assets = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                         ['current_assets'])
            current_liabilities = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                             ['current_liabilities'])
            cash = self._get_value_from_sources([balance_sheet, financial_metrics],
                                               ['cash', 'cash_and_equivalents'])
            inventory = self._get_value_from_sources([balance_sheet],
                                                    ['inventory', 'inventories'])
            operating_cash_flow = self._get_value_from_sources([cash_flow],
                                                             ['operating_cash_flow', 'cash_from_operations'])
            
            # Calculate liquidity ratios
            ratios['calculated_current_ratio'] = self._safe_divide(current_assets, current_liabilities)
            ratios['calculated_quick_ratio'] = self._safe_divide(
                (current_assets - inventory), current_liabilities
            )
            ratios['calculated_cash_ratio'] = self._safe_divide(cash, current_liabilities)
            ratios['operating_cash_flow_ratio'] = self._safe_divide(
                operating_cash_flow, current_liabilities
            )
            
            # Working capital metrics
            working_capital = current_assets - current_liabilities
            ratios['working_capital'] = working_capital
            ratios['working_capital_ratio'] = self._safe_divide(working_capital, current_assets)
            
        except Exception as e:
            logger.warning(f"Error calculating liquidity ratios: {e}")
            ratios['liquidity_error'] = str(e)
        
        return ratios
    
    def _calculate_leverage_ratios(self, balance_sheet: Dict, profit_loss: Dict,
                                 financial_metrics: Dict) -> Dict[str, float]:
        """Calculate leverage/debt ratios"""
        ratios = {}
        
        try:
            # Get base values
            total_debt = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                     ['total_debt', 'debt'])
            total_equity = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                       ['total_equity', 'shareholders_equity'])
            total_assets = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                       ['total_assets'])
            interest_expense = self._get_value_from_sources([profit_loss],
                                                          ['interest_expense', 'finance_cost'])
            ebit = self._get_value_from_sources([profit_loss],
                                              ['ebit', 'operating_income'])
            
            # Calculate leverage ratios
            ratios['calculated_debt_to_equity'] = self._safe_divide(total_debt, total_equity)
            ratios['calculated_debt_ratio'] = self._safe_divide(total_debt, total_assets)
            ratios['calculated_equity_ratio'] = self._safe_divide(total_equity, total_assets)
            ratios['times_interest_earned'] = self._safe_divide(ebit, interest_expense)
            
            # Additional leverage metrics
            ratios['debt_to_assets'] = self._safe_divide(total_debt, total_assets) * 100
            ratios['equity_multiplier'] = self._safe_divide(total_assets, total_equity)
            
        except Exception as e:
            logger.warning(f"Error calculating leverage ratios: {e}")
            ratios['leverage_error'] = str(e)
        
        return ratios
    
    def _calculate_efficiency_ratios(self, balance_sheet: Dict, profit_loss: Dict,
                                   financial_metrics: Dict) -> Dict[str, float]:
        """Calculate efficiency/activity ratios"""
        ratios = {}
        
        try:
            # Get base values
            revenue = self._get_value_from_sources([profit_loss, financial_metrics],
                                                 ['revenue', 'net_sales'])
            total_assets = self._get_value_from_sources([balance_sheet, financial_metrics],
                                                       ['total_assets'])
            inventory = self._get_value_from_sources([balance_sheet],
                                                    ['inventory'])
            accounts_receivable = self._get_value_from_sources([balance_sheet],
                                                             ['accounts_receivable', 'receivables'])
            cost_of_goods_sold = self._get_value_from_sources([profit_loss],
                                                            ['cost_of_goods_sold', 'cogs'])
            
            # Calculate efficiency ratios
            ratios['asset_turnover'] = self._safe_divide(revenue, total_assets)
            ratios['inventory_turnover'] = self._safe_divide(cost_of_goods_sold, inventory)
            ratios['receivables_turnover'] = self._safe_divide(revenue, accounts_receivable)
            
            # Days calculations
            if ratios['inventory_turnover'] > 0:
                ratios['days_in_inventory'] = 365 / ratios['inventory_turnover']
            if ratios['receivables_turnover'] > 0:
                ratios['days_sales_outstanding'] = 365 / ratios['receivables_turnover']
            
            # Revenue efficiency
            ratios['revenue_per_employee'] = self._get_value_from_sources(
                [financial_metrics], ['revenue_per_employee']
            )
            
        except Exception as e:
            logger.warning(f"Error calculating efficiency ratios: {e}")
            ratios['efficiency_error'] = str(e)
        
        return ratios
    
    def _calculate_growth_metrics(self, growth_metrics: Dict, 
                                financial_metrics: Dict) -> Dict[str, float]:
        """Calculate and validate growth metrics"""
        growth = {}
        
        try:
            # Direct growth metrics from API
            growth_fields = [
                'revenue_growth', 'profit_growth', 'asset_growth',
                'equity_growth', 'eps_growth'
            ]
            
            for field in growth_fields:
                value = self._get_value_from_sources([growth_metrics, financial_metrics], [field])
                # Validate growth rates (cap extreme values)
                if abs(value) > 200:  # >200% growth might be unrealistic
                    logger.warning(f"Extreme growth rate for {field}: {value}%")
                    value = min(max(value, -100), 200)  # Cap between -100% and 200%
                growth[field] = value
            
            # Calculate compound annual growth rates if historical data available
            # (This would require multi-year data - placeholder for now)
            growth['revenue_cagr_3y'] = growth_metrics.get('revenue_cagr_3y', 0)
            growth['revenue_cagr_5y'] = growth_metrics.get('revenue_cagr_5y', 0)
            growth['profit_cagr_3y'] = growth_metrics.get('profit_cagr_3y', 0)
            
            # Growth consistency metrics
            revenue_growth = growth.get('revenue_growth', 0)
            profit_growth = growth.get('profit_growth', 0)
            
            if revenue_growth != 0:
                growth['profit_to_revenue_growth_ratio'] = self._safe_divide(
                    profit_growth, revenue_growth
                )
            
            # Growth quality score
            growth_score = 0
            if revenue_growth > 0:
                growth_score += min(revenue_growth, 20)  # Cap at 20 points
            if profit_growth > revenue_growth:
                growth_score += 10  # Bonus for profit growing faster than revenue
            
            growth['growth_quality_score'] = max(0, min(100, growth_score))
            
        except Exception as e:
            logger.warning(f"Error calculating growth metrics: {e}")
            growth['growth_error'] = str(e)
        
        return growth
    
    def _calculate_market_ratios(self, financial_metrics: Dict, ratios: Dict) -> Dict[str, float]:
        """Calculate market-based ratios if data available"""
        market = {}
        
        try:
            # Market ratios (if available from API)
            market_fields = {
                'pe_ratio': ['pe_ratio', 'price_to_earnings', 'p_e'],
                'pb_ratio': ['pb_ratio', 'price_to_book', 'p_b'],
                'dividend_yield': ['dividend_yield', 'div_yield'],
                'earnings_per_share': ['eps', 'earnings_per_share'],
                'book_value_per_share': ['book_value_per_share', 'bvps']
            }
            
            for ratio_name, possible_keys in market_fields.items():
                market[ratio_name] = self._get_value_from_sources(
                    [ratios, financial_metrics], possible_keys
                )
            
            # Market valuation indicators
            pe_ratio = market.get('pe_ratio', 0)
            pb_ratio = market.get('pb_ratio', 0)
            
            if pe_ratio > 0:
                if pe_ratio < 15:
                    market['pe_interpretation'] = 'undervalued'
                elif pe_ratio > 25:
                    market['pe_interpretation'] = 'overvalued'
                else:
                    market['pe_interpretation'] = 'fairly_valued'
            
            if pb_ratio > 0:
                if pb_ratio < 1:
                    market['pb_interpretation'] = 'undervalued'
                elif pb_ratio > 3:
                    market['pb_interpretation'] = 'overvalued'
                else:
                    market['pb_interpretation'] = 'fairly_valued'
            
        except Exception as e:
            logger.warning(f"Error calculating market ratios: {e}")
            market['market_error'] = str(e)
        
        return market
    
    def _calculate_composite_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite financial health scores"""
        scores = {}
        
        try:
            # Profitability Score (0-100)
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            profit_margin = metrics.get('profit_margin', 0) or metrics.get('calculated_profit_margin', 0)
            roa = metrics.get('roa', 0) or metrics.get('calculated_roa', 0)
            
            profitability_components = [
                min(max(roe * 2, 0), 40),  # ROE component (max 40 points)
                min(max(profit_margin * 3, 0), 30),  # Profit margin (max 30 points)
                min(max(roa * 3, 0), 30)   # ROA component (max 30 points)
            ]
            scores['profitability_score'] = sum(profitability_components)
            
            # Liquidity Score (0-100)
            current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
            quick_ratio = metrics.get('quick_ratio', 0) or metrics.get('calculated_quick_ratio', 0)
            
            liquidity_components = [
                min(max((current_ratio - 1) * 50, 0), 50),  # Current ratio (max 50)
                min(max((quick_ratio - 0.5) * 50, 0), 50)   # Quick ratio (max 50)
            ]
            scores['liquidity_score'] = sum(liquidity_components)
            
            # Leverage/Stability Score (0-100)
            debt_to_equity = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
            debt_ratio = metrics.get('debt_ratio', 0) or metrics.get('calculated_debt_ratio', 0)
            
            # Lower debt is better for stability
            stability_components = [
                max(0, 100 - (debt_to_equity * 20)),  # Debt-to-equity penalty
                max(0, 100 - (debt_ratio * 100))      # Debt ratio penalty
            ]
            scores['stability_score'] = sum(stability_components) / 2
            
            # Growth Score (0-100)
            revenue_growth = max(0, metrics.get('sales_growth', 0) or metrics.get('revenue_growth', 0))
            profit_growth = max(0, metrics.get('profit_growth', 0))
            
            growth_components = [
                min(revenue_growth * 2, 50),  # Revenue growth (max 50)
                min(profit_growth * 2, 50)    # Profit growth (max 50)
            ]
            scores['growth_score'] = sum(growth_components)
            
            # Overall Financial Health Score (weighted average)
            weights = {
                'profitability_score': 0.35,
                'liquidity_score': 0.20,
                'stability_score': 0.25,
                'growth_score': 0.20
            }
            
            weighted_score = sum(
                scores.get(metric, 0) * weight 
                for metric, weight in weights.items()
            )
            scores['financial_health_score'] = min(100, max(0, weighted_score))
            
            # Risk Score (inverse of stability)
            scores['risk_score'] = max(0, 100 - scores.get('stability_score', 50))
            
        except Exception as e:
            logger.warning(f"Error calculating composite scores: {e}")
            scores['composite_error'] = str(e)
        
        return scores
    
    def _calculate_ml_features(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate ML-ready features for analysis"""
        ml_features = {}
        
        try:
            # Normalized features for ML (0-1 scale)
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            profit_margin = metrics.get('profit_margin', 0) or metrics.get('calculated_profit_margin', 0)
            current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
            debt_to_equity = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
            
            # Normalize key metrics (sigmoid-like transformation)
            ml_features['roe_normalized'] = self._normalize_metric(roe, 0, 50)
            ml_features['profit_margin_normalized'] = self._normalize_metric(profit_margin, 0, 30)
            ml_features['current_ratio_normalized'] = self._normalize_metric(current_ratio, 0.5, 3)
            ml_features['debt_equity_normalized'] = 1 - self._normalize_metric(debt_to_equity, 0, 2)
            
            # Binary features for ML
            ml_features['is_profitable'] = 1 if roe > 0 and profit_margin > 0 else 0
            ml_features['is_liquid'] = 1 if current_ratio > 1.2 else 0
            ml_features['is_low_debt'] = 1 if debt_to_equity < 0.5 else 0
            revenue_growth = metrics.get('sales_growth', 0) or metrics.get('revenue_growth', 0)
            ml_features['is_growing'] = 1 if revenue_growth > 5 else 0
            
            # Categorical features
            financial_health = metrics.get('financial_health_score', 0)
            if financial_health >= 80:
                ml_features['health_category'] = 3  # Excellent
            elif financial_health >= 60:
                ml_features['health_category'] = 2  # Good
            elif financial_health >= 40:
                ml_features['health_category'] = 1  # Fair
            else:
                ml_features['health_category'] = 0  # Poor
            
            # Interaction features
            profitability = metrics.get('profitability_score', 0)
            stability = metrics.get('stability_score', 0)
            ml_features['profitability_stability_interaction'] = (profitability * stability) / 10000
            
        except Exception as e:
            logger.warning(f"Error calculating ML features: {e}")
            ml_features['ml_features_error'] = str(e)
        
        return ml_features
    
    def _get_value_from_sources(self, sources: List[Dict], keys: List[str]) -> float:
        """Get value from multiple possible sources and keys"""
        for source in sources:
            if isinstance(source, dict):
                for key in keys:
                    if key in source and source[key] is not None:
                        try:
                            value = source[key]
                            if isinstance(value, str):
                                if value.upper() in ['N/A', 'NA', 'NULL', '-', '']:
                                    continue
                                # Remove formatting and convert
                                cleaned = value.replace(',', '').replace('%', '').replace('₹', '').strip()
                                return float(cleaned)
                            return float(value)
                        except (ValueError, TypeError):
                            continue
        return 0.0
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safely divide two numbers, handling zero division"""
        try:
            if denominator == 0 or denominator is None:
                self.calculation_stats['zero_division_errors'] += 1
                return 0.0
            result = numerator / denominator
            # Handle infinity and NaN
            if math.isinf(result) or math.isnan(result):
                return 0.0
            return result
        except (TypeError, ZeroDivisionError):
            self.calculation_stats['zero_division_errors'] += 1
            return 0.0
    
    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize metric to 0-1 scale"""
        try:
            if max_val <= min_val:
                return 0.0
            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))  # Clamp to [0,1]
        except:
            return 0.0
    
    def get_metrics_summary(self, calculated_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get summary of calculated metrics organized by category"""
        summary = {
            'total_metrics': len(calculated_metrics),
            'categories': {},
            'key_insights': []
        }
        
        try:
            # Organize metrics by category
            for category, metric_list in self.metric_categories.items():
                category_metrics = {}
                for metric in metric_list:
                    if metric in calculated_metrics:
                        category_metrics[metric] = calculated_metrics[metric]
                    # Also check for calculated_ prefixed versions
                    calc_metric = f"calculated_{metric}"
                    if calc_metric in calculated_metrics:
                        category_metrics[calc_metric] = calculated_metrics[calc_metric]
                
                summary['categories'][category] = category_metrics
            
            # Generate key insights
            insights = []
            
            # Profitability insights
            roe = calculated_metrics.get('roe', 0) or calculated_metrics.get('calculated_roe', 0)
            if roe > 20:
                insights.append(f"Excellent ROE of {roe:.1f}%")
            elif roe < 10:
                insights.append(f"Low ROE of {roe:.1f}%")
            
            # Growth insights
            revenue_growth = calculated_metrics.get('sales_growth', 0) or calculated_metrics.get('revenue_growth', 0)
            if revenue_growth > 15:
                insights.append(f"Strong revenue growth of {revenue_growth:.1f}%")
            elif revenue_growth < 0:
                insights.append(f"Revenue declining by {abs(revenue_growth):.1f}%")
            
            # Debt insights
            debt_to_equity = calculated_metrics.get('debt_to_equity', 0) or calculated_metrics.get('calculated_debt_to_equity', 0)
            if debt_to_equity < 0.3:
                insights.append("Company is almost debt-free")
            elif debt_to_equity > 1:
                insights.append("High debt levels - potential risk")
            
            # Liquidity insights
            current_ratio = calculated_metrics.get('current_ratio', 0) or calculated_metrics.get('calculated_current_ratio', 0)
            if current_ratio > 2:
                insights.append("Strong liquidity position")
            elif current_ratio < 1:
                insights.append("Liquidity concerns - current ratio below 1")
            
            summary['key_insights'] = insights
            
        except Exception as e:
            logger.warning(f"Error generating metrics summary: {e}")
            summary['summary_error'] = str(e)
        
        return summary
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """Get statistics about calculation operations"""
        total_calcs = self.calculation_stats['total_calculations']
        return {
            'total_calculations': total_calcs,
            'calculation_errors': self.calculation_stats['calculation_errors'],
            'zero_division_errors': self.calculation_stats['zero_division_errors'],
            'success_rate': (
                (total_calcs - self.calculation_stats['calculation_errors']) 
                / max(1, total_calcs)
            ) * 100,
            'error_rate': (
                self.calculation_stats['calculation_errors'] / max(1, total_calcs)
            ) * 100
        }
    
    def reset_statistics(self):
        """Reset calculation statistics"""
        self.calculation_stats = {
            'total_calculations': 0,
            'calculation_errors': 0,
            'zero_division_errors': 0
        }
    
    def validate_metrics(self, calculated_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate calculated metrics for reasonableness"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Define reasonable ranges for key metrics
            validation_rules = {
                'roe': (-50, 100, '%'),
                'roa': (-20, 50, '%'),
                'current_ratio': (0, 10, 'ratio'),
                'debt_to_equity': (0, 5, 'ratio'),
                'profit_margin': (-50, 50, '%'),
                'financial_health_score': (0, 100, 'score')
            }
            
            for metric, (min_val, max_val, unit) in validation_rules.items():
                value = calculated_metrics.get(metric, 0)
                
                if value < min_val or value > max_val:
                    warning_msg = f"{metric} value {value:.2f}{unit} outside expected range ({min_val}, {max_val})"
                    validation_results['warnings'].append(warning_msg)
                    logger.warning(warning_msg)
            
            # Check for calculation errors
            error_metrics = [k for k in calculated_metrics.keys() if k.endswith('_error')]
            if error_metrics:
                validation_results['errors'].extend(error_metrics)
                validation_results['is_valid'] = False
            
            # Check for all zeros (potential data issue)
            non_zero_metrics = [v for v in calculated_metrics.values() 
                            if isinstance(v, (int, float)) and v != 0]
            if len(non_zero_metrics) < 5:
                validation_results['warnings'].append("Most metrics are zero - possible data quality issue")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results