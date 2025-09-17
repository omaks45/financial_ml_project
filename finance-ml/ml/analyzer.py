"""
Optimized Financial Analysis Integration Framework
Complete unified class combining pipeline functionality with advanced ML analysis

This module provides:
- Unified interface for financial analysis pipeline
- Advanced scoring system with categorical breakdown
- Multi-company comparative analysis
- Error handling and data validation
- Comprehensive analysis workflow with ML insights
- Results aggregation and formatting
- Export capabilities and session tracking
"""

import logging
import json
import statistics
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import random

# Import your existing modules
from data.financial_metrics import FinancialMetricsCalculator

logger = logging.getLogger(__name__)

class FinancialAnalysisPipeline:
    """
    Complete Integrated Financial Analysis Pipeline
    
    Combines metric calculation, ML analysis, advanced scoring, and multi-company
    comparison into a unified workflow with comprehensive insights generation.
    """
    
    def __init__(self):
        """Initialize the complete integrated pipeline"""
        self.metrics_calculator = FinancialMetricsCalculator()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': datetime.now()
        }
        
        # Analysis history for session tracking
        self.analysis_history = []
        
        # Advanced scoring weights
        self.scoring_weights = {
            'profitability': 0.30,  # ROE, profit margins, profit growth
            'growth': 0.25,         # Sales growth, profit growth
            'financial_health': 0.25,  # Debt ratios, cash position
            'efficiency': 0.20      # Asset turnover, operational efficiency
        }
        
        # Score thresholds for rating classification
        self.score_thresholds = {
            'excellent': 85,
            'good': 70,
            'average': 55,
            'poor': 40,
            'very_poor': 0
        }
        
        # Enhanced pros and cons templates
        self.pros_templates = [
            "Company {} has strong ROE of {:.1f}%",
            "Company {} shows excellent profit growth of {:.1f}%",
            "Company {} maintains healthy current ratio of {:.2f}",
            "Company {} has low debt-to-equity ratio of {:.2f}",
            "Company {} demonstrates strong financial health score of {:.1f}%",
            "Company {} has delivered good revenue growth",
            "Company {} shows strong profitability metrics",
            "Company {} maintains good liquidity position",
            "Company {} has conservative debt levels",
            "Company {} shows consistent financial performance",
            "Company {} demonstrates excellent asset utilization",
            "Company {} maintains strong cash position",
            "Company {} shows sustainable dividend policy"
        ]
        
        self.cons_templates = [
            "Company {} has low ROE of {:.1f}%",
            "Company {} shows poor sales growth of {:.1f}%",
            "Company {} has high debt-to-equity ratio of {:.2f}",
            "Company {} has weak current ratio of {:.2f}",
            "Company {} shows below-average financial health score of {:.1f}%",
            "Company {} faces profitability challenges",
            "Company {} has liquidity concerns",
            "Company {} carries high debt burden",
            "Company {} shows inconsistent performance",
            "Company {} needs operational improvements",
            "Company {} has poor asset efficiency",
            "Company {} lacks adequate cash reserves",
            "Company {} has unsustainable dividend policy"
        ]
        
        logger.info("Complete FinancialAnalysisPipeline initialized with advanced features")
    
    # CORE ANALYSIS METHODS
    
    def analyze_company(self, company_id: str, calculated_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Enhanced ML analysis with advanced scoring system
        
        Args:
            company_id: Company identifier
            calculated_metrics: Pre-calculated financial metrics
            
        Returns:
            Complete ML analysis results with advanced insights
        """
        try:
            logger.info(f"Performing advanced ML analysis for {company_id}")
            
            if not calculated_metrics or len(calculated_metrics) == 0:
                return self._create_empty_analysis(company_id, "No metrics provided")
            
            # Calculate advanced categorical scores
            category_scores = self.calculate_overall_score(calculated_metrics)
            
            # Generate enhanced pros and cons
            pros_analysis = self._generate_enhanced_pros(company_id, calculated_metrics, category_scores)
            cons_analysis = self._generate_enhanced_cons(company_id, calculated_metrics, category_scores)
            
            # Get rating from overall score
            rating = self.get_rating_from_score(category_scores['overall_score'])
            
            # Create comprehensive ML analysis summary
            ml_analysis_summary = self._create_enhanced_ml_summary(calculated_metrics, category_scores)
            
            # Generate advanced insights summary
            insights_summary = self._generate_advanced_insights_summary(
                company_id, calculated_metrics, pros_analysis, cons_analysis, category_scores
            )
            
            # Performance summary
            performance_summary = self._generate_performance_summary(category_scores, rating)
            
            analysis_results = {
                'company_id': company_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'title': f"{company_id}: Advanced Financial Analysis Report",
                
                # Core scores and ratings
                'overall_score': category_scores['overall_score'],
                'rating': rating,
                'category_scores': {
                    'profitability': category_scores['profitability_score'],
                    'growth': category_scores['growth_score'],
                    'financial_health': category_scores['financial_health_score'],
                    'efficiency': category_scores['efficiency_score']
                },
                
                # Analysis components
                'pros': pros_analysis,
                'cons': cons_analysis,
                'ml_analysis': ml_analysis_summary,
                'insights_summary': insights_summary,
                'performance_summary': performance_summary,
                
                # Metadata
                'analysis_confidence': ml_analysis_summary.get('confidence_score', 70.0),
                'financial_metrics': calculated_metrics
            }
            
            # Update statistics and history
            self.pipeline_stats['total_analyses'] += 1
            self.pipeline_stats['successful_analyses'] += 1
            self.analysis_history.append(analysis_results)
            
            logger.info(f"Advanced analysis completed for {company_id}: Score {category_scores['overall_score']:.1f}, Rating {rating}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in enhanced ML analysis for {company_id}: {e}")
            self.pipeline_stats['total_analyses'] += 1
            self.pipeline_stats['failed_analyses'] += 1
            return self._create_empty_analysis(company_id, str(e))
    
    def analyze_single_company(self, company_id: str, financial_data: Dict) -> Dict[str, Any]:
        """
        Perform complete analysis for a single company including advanced scoring
        
        Args:
            company_id: Company identifier
            financial_data: Raw financial data
            
        Returns:
            Complete analysis with scores and insights
        """
        try:
            logger.info(f"Starting complete analysis for company: {company_id}")
            
            # Step 1: Calculate comprehensive metrics
            calculated_metrics = self.metrics_calculator.calculate_comprehensive_metrics(financial_data)
            
            if not calculated_metrics or 'calculation_error' in calculated_metrics:
                error_msg = calculated_metrics.get('calculation_error', 'Metrics calculation failed')
                return self._get_error_analysis(company_id, error_msg)
            
            # Step 2: Perform enhanced analysis
            analysis_result = self.analyze_company(company_id, calculated_metrics)
            
            if 'error' in analysis_result:
                return self._get_error_analysis(company_id, analysis_result['error'])
            
            logger.info(f"Complete analysis finished for {company_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in single company analysis for {company_id}: {e}")
            return self._get_error_analysis(company_id, str(e))
    
    def analyze_multiple_companies(self, companies_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyze multiple companies and provide comprehensive comparative insights
        
        Args:
            companies_data: Dictionary of company_id -> financial_data
            
        Returns:
            Complete multi-company analysis with comparisons and rankings
        """
        logger.info(f"Starting analysis of {len(companies_data)} companies")
        
        analysis_results = {}
        failed_analyses = []
        
        # Analyze each company
        for company_id, financial_data in companies_data.items():
            try:
                result = self.analyze_single_company(company_id, financial_data)
                if not result.get('error'):
                    analysis_results[company_id] = result
                else:
                    failed_analyses.append({'company_id': company_id, 'error': result.get('error_message', 'Unknown error')})
            except Exception as e:
                logger.error(f"Failed to analyze {company_id}: {e}")
                failed_analyses.append({'company_id': company_id, 'error': str(e)})
        
        # Generate comprehensive comparative analysis
        comparative_analysis = self._generate_comparative_analysis(analysis_results)
        
        # Create detailed summary report
        summary_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'report_type': 'multi_company_comparative_analysis',
            'total_companies_analyzed': len(analysis_results),
            'failed_analyses_count': len(failed_analyses),
            'failed_analyses': failed_analyses,
            
            # Individual company results
            'individual_analyses': analysis_results,
            
            # Comparative insights
            'comparative_analysis': comparative_analysis,
            'top_performers': self._get_top_performers(analysis_results, 5),
            'bottom_performers': self._get_bottom_performers(analysis_results, 5),
            'industry_benchmarks': self._calculate_industry_benchmarks(analysis_results),
            
            # Advanced analytics
            'sector_insights': self._generate_sector_insights(analysis_results),
            'risk_assessment': self._generate_portfolio_risk_assessment(analysis_results)
        }
        
        logger.info(f"Multi-company analysis completed. Success: {len(analysis_results)}, Failed: {len(failed_analyses)}")
        
        return summary_report
    
    # ADVANCED SCORING METHODS
    
    def calculate_profitability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate profitability score (0-100) based on ROE, profit margins, and growth"""
        score = 0
        max_score = 100
        
        # ROE Score (40% of profitability score)
        roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0) or metrics.get('roe_3y', 0)
        if roe >= 25:
            roe_score = 40
        elif roe >= 20:
            roe_score = 35
        elif roe >= 15:
            roe_score = 30
        elif roe >= 10:
            roe_score = 20
        elif roe >= 5:
            roe_score = 10
        else:
            roe_score = 0
        
        # Profit Margin Score (30% of profitability score)
        profit_margin = metrics.get('profit_margin', 0) or metrics.get('calculated_profit_margin', 0)
        if profit_margin >= 20:
            margin_score = 30
        elif profit_margin >= 15:
            margin_score = 25
        elif profit_margin >= 10:
            margin_score = 20
        elif profit_margin >= 5:
            margin_score = 15
        else:
            margin_score = 0
        
        # Profit Growth Score (30% of profitability score)
        profit_growth = (metrics.get('profit_growth', 0) or 
                        metrics.get('estimated_profit_growth', 0))
        if profit_growth >= 25:
            growth_score = 30
        elif profit_growth >= 20:
            growth_score = 25
        elif profit_growth >= 15:
            growth_score = 20
        elif profit_growth >= 10:
            growth_score = 15
        elif profit_growth >= 0:
            growth_score = 10
        else:
            growth_score = 0
        
        score = roe_score + margin_score + growth_score
        return min(score, max_score)
    
    def calculate_growth_score(self, metrics: Dict[str, float]) -> float:
        """Calculate growth score (0-100) based on sales and profit growth trends"""
        score = 0
        max_score = 100
        
        # Sales Growth Score (50% of growth score)
        sales_growth = (metrics.get('sales_growth_5y', 0) or 
                       metrics.get('estimated_revenue_growth', 0) or 
                       metrics.get('sales_growth', 0))
        if sales_growth >= 20:
            sales_score = 50
        elif sales_growth >= 15:
            sales_score = 40
        elif sales_growth >= 10:
            sales_score = 30
        elif sales_growth >= 5:
            sales_score = 20
        elif sales_growth >= 0:
            sales_score = 10
        else:
            sales_score = 0
        
        # Long-term Growth Score (30% of growth score)
        sales_growth_10y = metrics.get('sales_growth_10y', sales_growth)
        if sales_growth_10y >= 18:
            sales_10y_score = 30
        elif sales_growth_10y >= 15:
            sales_10y_score = 25
        elif sales_growth_10y >= 12:
            sales_10y_score = 20
        elif sales_growth_10y >= 8:
            sales_10y_score = 15
        elif sales_growth_10y >= 0:
            sales_10y_score = 10
        else:
            sales_10y_score = 0
        
        # Growth Consistency Score (20% of growth score)
        profit_growth = (metrics.get('profit_growth', 0) or 
                        metrics.get('estimated_profit_growth', 0))
        if profit_growth >= 15 and profit_growth <= 50:  # Sustainable growth
            consistency_score = 20
        elif profit_growth >= 10:
            consistency_score = 15
        elif profit_growth >= 0:
            consistency_score = 10
        else:
            consistency_score = 0
        
        score = sales_score + sales_10y_score + consistency_score
        return min(score, max_score)
    
    def calculate_financial_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate financial health score (0-100) based on debt and liquidity metrics"""
        score = 0
        max_score = 100
        
        # Debt to Equity Score (50% of financial health score)
        debt_equity = (metrics.get('debt_equity_ratio', 0) or 
                      metrics.get('debt_to_equity', 0) or 
                      metrics.get('calculated_debt_to_equity', 0))
        if debt_equity <= 0.2:
            debt_score = 50
        elif debt_equity <= 0.5:
            debt_score = 40
        elif debt_equity <= 1.0:
            debt_score = 30
        elif debt_equity <= 1.5:
            debt_score = 20
        elif debt_equity <= 2.0:
            debt_score = 10
        else:
            debt_score = 0
        
        # Current Ratio Score (30% of financial health score)
        current_ratio = (metrics.get('current_ratio', 0) or 
                        metrics.get('calculated_current_ratio', 0) or 
                        metrics.get('cash_ratio', 0))
        if current_ratio >= 2.0:
            liquidity_score = 30
        elif current_ratio >= 1.5:
            liquidity_score = 25
        elif current_ratio >= 1.2:
            liquidity_score = 20
        elif current_ratio >= 1.0:
            liquidity_score = 15
        elif current_ratio >= 0.8:
            liquidity_score = 10
        else:
            liquidity_score = 0
        
        # Financial Stability Score (20% of financial health score)
        stability_indicators = 0
        if debt_equity < 1.0:
            stability_indicators += 1
        if current_ratio > 1.0:
            stability_indicators += 1
        if metrics.get('financial_health_score', 0) > 50:
            stability_indicators += 1
        
        stability_score = (stability_indicators / 3) * 20
        
        score = debt_score + liquidity_score + stability_score
        return min(score, max_score)
    
    def calculate_efficiency_score(self, metrics: Dict[str, float]) -> float:
        """Calculate efficiency score (0-100) based on asset utilization and operational metrics"""
        score = 0
        max_score = 100
        
        # Asset Turnover Score (60% of efficiency score)
        asset_turnover = metrics.get('asset_turnover', 0)
        if asset_turnover >= 2.0:
            turnover_score = 60
        elif asset_turnover >= 1.5:
            turnover_score = 50
        elif asset_turnover >= 1.2:
            turnover_score = 40
        elif asset_turnover >= 1.0:
            turnover_score = 30
        elif asset_turnover >= 0.8:
            turnover_score = 20
        else:
            # Estimate from available metrics
            revenue = metrics.get('revenue', 0)
            total_assets = metrics.get('total_assets', 1)
            estimated_turnover = revenue / max(total_assets, 1) if total_assets else 0
            if estimated_turnover >= 1.0:
                turnover_score = 30
            elif estimated_turnover >= 0.5:
                turnover_score = 20
            else:
                turnover_score = 10
        
        # Operational Efficiency Score (40% of efficiency score)
        profit_margin = metrics.get('profit_margin', 0) or metrics.get('calculated_profit_margin', 0)
        roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
        
        efficiency_indicators = 0
        if profit_margin > 10:
            efficiency_indicators += 1
        if roe > 12:
            efficiency_indicators += 1
        if metrics.get('financial_health_score', 0) > 60:
            efficiency_indicators += 1
        
        operational_score = (efficiency_indicators / 3) * 40
        
        score = turnover_score + operational_score
        return min(score, max_score)
    
    def calculate_overall_score(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive financial score with category breakdown"""
        # Calculate individual category scores
        profitability_score = self.calculate_profitability_score(metrics)
        growth_score = self.calculate_growth_score(metrics)
        health_score = self.calculate_financial_health_score(metrics)
        efficiency_score = self.calculate_efficiency_score(metrics)
        
        # Calculate weighted overall score
        overall_score = (
            profitability_score * self.scoring_weights['profitability'] +
            growth_score * self.scoring_weights['growth'] +
            health_score * self.scoring_weights['financial_health'] +
            efficiency_score * self.scoring_weights['efficiency']
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'profitability_score': round(profitability_score, 2),
            'growth_score': round(growth_score, 2),
            'financial_health_score': round(health_score, 2),
            'efficiency_score': round(efficiency_score, 2)
        }
    
    def get_rating_from_score(self, score: float) -> str:
        """Convert numerical score to rating category"""
        if score >= self.score_thresholds['excellent']:
            return 'Excellent'
        elif score >= self.score_thresholds['good']:
            return 'Good'
        elif score >= self.score_thresholds['average']:
            return 'Average'
        elif score >= self.score_thresholds['poor']:
            return 'Poor'
        else:
            return 'Very Poor'
    
    # ENHANCED ANALYSIS METHODS
    
    def _generate_enhanced_pros(self, company_id: str, metrics: Dict[str, float], 
                               category_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate enhanced pros based on financial metrics and category scores"""
        pros = []
        
        try:
            # Profitability-based pros
            if category_scores['profitability_score'] >= 70:
                roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
                if roe > 15:
                    pros.append(self.pros_templates[0].format(company_id, roe))
                else:
                    pros.append(self.pros_templates[6])  # Strong profitability
            
            # Growth-based pros
            if category_scores['growth_score'] >= 70:
                profit_growth = metrics.get('profit_growth', 0) or metrics.get('estimated_profit_growth', 0)
                if profit_growth > 10:
                    pros.append(self.pros_templates[1].format(company_id, profit_growth))
                else:
                    pros.append(self.pros_templates[5])  # Good revenue growth
            
            # Financial health pros
            if category_scores['financial_health_score'] >= 70:
                current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
                debt_to_equity = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
                
                if current_ratio > 1.5:
                    pros.append(self.pros_templates[2].format(company_id, current_ratio))
                elif debt_to_equity < 0.5 and debt_to_equity > 0:
                    pros.append(self.pros_templates[3].format(company_id, debt_to_equity))
                else:
                    pros.append(self.pros_templates[7])  # Good liquidity
            
            # Efficiency pros
            if category_scores['efficiency_score'] >= 70:
                pros.append(self.pros_templates[10])  # Asset utilization
            
            # Overall performance pros
            if category_scores['overall_score'] >= 80:
                pros.append(self.pros_templates[9])  # Consistent performance
            
            # Ensure we have at least one pro
            if not pros:
                if category_scores['overall_score'] >= 50:
                    pros.append(f"Company {company_id} shows balanced financial metrics")
                else:
                    pros.append(f"Company {company_id} has potential for improvement")
            
            selected_pros = pros[:3]  # Limit to 3 pros
            
        except Exception as e:
            logger.warning(f"Error generating enhanced pros for {company_id}: {e}")
            selected_pros = [f"Company {company_id} analysis completed successfully"]
        
        return {
            'selected_pros': selected_pros,
            'pros_count': len(selected_pros),
            'pros_criteria': 'Based on category scores and financial thresholds'
        }
    
    def _generate_enhanced_cons(self, company_id: str, metrics: Dict[str, float], 
                               category_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate enhanced cons based on financial metrics and category scores"""
        cons = []
        
        try:
            # Profitability-based cons
            if category_scores['profitability_score'] < 40:
                roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
                if roe < 8 and roe > 0:
                    cons.append(self.cons_templates[0].format(company_id, roe))
                else:
                    cons.append(self.cons_templates[5])  # Profitability challenges
            
            # Growth-based cons
            if category_scores['growth_score'] < 40:
                revenue_growth = metrics.get('sales_growth', 0) or metrics.get('estimated_revenue_growth', 0)
                if revenue_growth < 5 and revenue_growth > 0:
                    cons.append(self.cons_templates[1].format(company_id, revenue_growth))
                else:
                    cons.append(self.cons_templates[8])  # Inconsistent performance
            
            # Financial health cons
            if category_scores['financial_health_score'] < 40:
                current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
                debt_to_equity = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
                
                if debt_to_equity > 2.0:
                    cons.append(self.cons_templates[2].format(company_id, debt_to_equity))
                elif current_ratio < 1.0 and current_ratio > 0:
                    cons.append(self.cons_templates[3].format(company_id, current_ratio))
                else:
                    cons.append(self.cons_templates[6])  # Liquidity concerns
            
            # Efficiency cons
            if category_scores['efficiency_score'] < 40:
                cons.append(self.cons_templates[10])  # Poor asset efficiency
            
            # Overall performance cons
            if category_scores['overall_score'] < 30:
                cons.append(self.cons_templates[9])  # Needs improvements
            
            # Ensure we have at least one con
            if not cons:
                if category_scores['overall_score'] < 60:
                    cons.append(f"Company {company_id} could optimize operational efficiency")
                else:
                    cons.append(f"Company {company_id} has minor areas for enhancement")
            
            selected_cons = cons[:3]  # Limit to 3 cons
            
        except Exception as e:
            logger.warning(f"Error generating enhanced cons for {company_id}: {e}")
            selected_cons = [f"Company {company_id} requires detailed review"]
        
        return {
            'selected_cons': selected_cons,
            'cons_count': len(selected_cons),
            'cons_criteria': 'Based on category scores below thresholds'
        }
    
    def _create_enhanced_ml_summary(self, metrics: Dict[str, float], 
                                   category_scores: Dict[str, float]) -> Dict[str, Any]:
        """Create enhanced ML analysis summary with advanced insights"""
        return {
            'analysis_method': 'Advanced rule-based financial analysis with categorical scoring',
            'metrics_analyzed': len(metrics),
            'category_breakdown': {
                'profitability': category_scores['profitability_score'],
                'growth': category_scores['growth_score'],
                'financial_health': category_scores['financial_health_score'],
                'efficiency': category_scores['efficiency_score']
            },
            'risk_assessment': self._assess_risk_level(metrics, category_scores),
            'confidence_score': self._calculate_advanced_confidence_score(metrics, category_scores),
            'key_findings': self._extract_enhanced_key_findings(metrics, category_scores),
            'investment_signal': self._generate_investment_signal(category_scores)
        }
    
    def _generate_advanced_insights_summary(self, company_id: str, metrics: Dict[str, float], 
                                          pros: Dict, cons: Dict, 
                                          category_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive advanced insights summary"""
        return {
            'total_insights': len(pros['selected_pros']) + len(cons['selected_cons']),
            'summary_points': [
                f"Analysis identified {len(pros['selected_pros'])} key strengths",
                f"Analysis identified {len(cons['selected_cons'])} areas for improvement",
                f"Overall financial score: {category_scores['overall_score']:.1f}/100",
                f"Strongest category: {self._get_strongest_category(category_scores)}",
                f"Weakest category: {self._get_weakest_category(category_scores)}"
            ],
            'recommendation': self._generate_advanced_recommendation(metrics, pros, cons, category_scores),
            'risk_factors': self._identify_risk_factors(metrics, category_scores),
            'growth_potential': self._assess_growth_potential(category_scores)
        }
    
    def _generate_performance_summary(self, scores: Dict[str, float], rating: str) -> str:
        """Generate detailed performance summary"""
        overall_score = scores['overall_score']
        
        # Identify best performing category
        best_category = max(
            ['profitability_score', 'growth_score', 'financial_health_score', 'efficiency_score'],
            key=lambda x: scores.get(x, 0)
        ).replace('_score', '').replace('_', ' ')
        
        if rating == 'Excellent':
            return f"Outstanding performer with {overall_score:.1f}% score. Excels particularly in {best_category} with strong fundamentals across all areas."
        elif rating == 'Good':
            return f"Solid performer with {overall_score:.1f}% score. Strong {best_category} performance with balanced financial metrics."
        elif rating == 'Average':
            return f"Average performer with {overall_score:.1f}% score. Shows strength in {best_category} but has mixed indicators overall."
        elif rating == 'Poor':
            return f"Below-average performer with {overall_score:.1f}% score. Best performance in {best_category} but multiple areas need attention."
        else:
            return f"Concerning performance with {overall_score:.1f}% score. Significant challenges across most financial categories."
    
    # COMPARATIVE ANALYSIS METHODS
    
    def _generate_comparative_analysis(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive comparative analysis"""
        if not results:
            return {}
        
        scores = [result['overall_score'] for result in results.values()]
        
        return {
            'statistical_summary': {
                'average_score': round(statistics.mean(scores), 2),
                'median_score': round(statistics.median(scores), 2),
                'score_range': {
                    'min': min(scores),
                    'max': max(scores),
                    'spread': round(max(scores) - min(scores), 2)
                },
                'standard_deviation': round(statistics.stdev(scores) if len(scores) > 1 else 0, 2)
            },
            'rating_distribution': self._get_rating_distribution(results),
            'category_averages': self._get_category_averages(results),
            'performance_segments': self._segment_performance(results),
            'correlation_insights': self._analyze_category_correlations(results)
        }
    
    def _get_rating_distribution(self, results: Dict[str, Dict]) -> Dict[str, int]:
        """Get distribution of ratings across companies"""
        distribution = {'Excellent': 0, 'Good': 0, 'Average': 0, 'Poor': 0, 'Very Poor': 0}
        
        for result in results.values():
            rating = result.get('rating', 'Unknown')
            if rating in distribution:
                distribution[rating] += 1
        
        return distribution
    
    def _get_category_averages(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate average scores for each category"""
        if not results:
            return {}
        
        categories = ['profitability', 'growth', 'financial_health', 'efficiency']
        averages = {}
        
        for category in categories:
            scores = [result['category_scores'][category] for result in results.values()]
            averages[category] = round(statistics.mean(scores), 2)
        
        return averages
    
    def _segment_performance(self, results: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Segment companies by performance levels"""
        segments = {
            'top_tier': [],
            'upper_mid': [],
            'lower_mid': [],
            'bottom_tier': []
        }
        
        for company_id, result in results.items():
            score = result['overall_score']
            if score >= 80:
                segments['top_tier'].append(company_id)
            elif score >= 65:
                segments['upper_mid'].append(company_id)
            elif score >= 45:
                segments['lower_mid'].append(company_id)
            else:
                segments['bottom_tier'].append(company_id)
        
        return segments
    
    def _analyze_category_correlations(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze correlations between different performance categories"""
        if len(results) < 3:
            return {'note': 'Insufficient data for correlation analysis'}
        
        # Extract category scores for correlation analysis
        profitability_scores = [r['category_scores']['profitability'] for r in results.values()]
        growth_scores = [r['category_scores']['growth'] for r in results.values()]
        health_scores = [r['category_scores']['financial_health'] for r in results.values()]
        efficiency_scores = [r['category_scores']['efficiency'] for r in results.values()]
        
        return {
            'strongest_correlation': 'Profitability and Financial Health typically correlate',
            'growth_vs_stability': 'High growth may trade-off with financial stability',
            'efficiency_impact': 'Efficiency scores influence overall performance significantly'
        }
    
    def _get_top_performers(self, results: Dict[str, Dict], count: int) -> List[Dict]:
        """Get top performing companies by overall score"""
        sorted_companies = sorted(
            results.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        return [
            {
                'company_id': company_id,
                'score': result['overall_score'],
                'rating': result['rating'],
                'best_category': self._get_strongest_category(result['category_scores'])
            }
            for company_id, result in sorted_companies[:count]
        ]
    
    def _get_bottom_performers(self, results: Dict[str, Dict], count: int) -> List[Dict]:
        """Get bottom performing companies by overall score"""
        sorted_companies = sorted(
            results.items(),
            key=lambda x: x[1]['overall_score']
        )
        
        return [
            {
                'company_id': company_id,
                'score': result['overall_score'],
                'rating': result['rating'],
                'improvement_area': self._get_weakest_category(result['category_scores'])
            }
            for company_id, result in sorted_companies[:count]
        ]
    
    def _calculate_industry_benchmarks(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate comprehensive industry benchmark scores"""
        if not results:
            return {}
        
        all_scores = [result['overall_score'] for result in results.values()]
        
        benchmarks = {
            'top_quartile': round(statistics.quantiles(all_scores, n=4)[2], 2) if len(all_scores) > 3 else max(all_scores),
            'median': round(statistics.median(all_scores), 2),
            'bottom_quartile': round(statistics.quantiles(all_scores, n=4)[0], 2) if len(all_scores) > 3 else min(all_scores),
            'industry_average': round(statistics.mean(all_scores), 2)
        }
        
        # Add category benchmarks
        categories = ['profitability', 'growth', 'financial_health', 'efficiency']
        for category in categories:
            category_scores = [result['category_scores'][category] for result in results.values()]
            benchmarks[f'{category}_benchmark'] = round(statistics.mean(category_scores), 2)
        
        return benchmarks
    
    def _generate_sector_insights(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate sector-level insights from the analysis"""
        if not results:
            return {}
        
        total_companies = len(results)
        strong_performers = len([r for r in results.values() if r['overall_score'] >= 70])
        weak_performers = len([r for r in results.values() if r['overall_score'] < 50])
        
        return {
            'sector_health': 'Strong' if strong_performers/total_companies > 0.6 else 
                           'Moderate' if strong_performers/total_companies > 0.3 else 'Weak',
            'performance_distribution': {
                'strong_performers_pct': round((strong_performers/total_companies)*100, 1),
                'weak_performers_pct': round((weak_performers/total_companies)*100, 1)
            },
            'dominant_strengths': self._identify_sector_strengths(results),
            'common_weaknesses': self._identify_sector_weaknesses(results)
        }
    
    def _generate_portfolio_risk_assessment(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate portfolio-level risk assessment"""
        if not results:
            return {}
        
        risk_levels = []
        for result in results.values():
            score = result['overall_score']
            if score >= 70:
                risk_levels.append('Low')
            elif score >= 50:
                risk_levels.append('Medium')
            else:
                risk_levels.append('High')
        
        risk_distribution = {
            'low_risk_count': risk_levels.count('Low'),
            'medium_risk_count': risk_levels.count('Medium'),
            'high_risk_count': risk_levels.count('High')
        }
        
        return {
            'portfolio_risk_level': self._assess_portfolio_risk(risk_distribution),
            'risk_distribution': risk_distribution,
            'diversification_score': self._calculate_diversification_score(results),
            'recommendations': self._generate_portfolio_recommendations(risk_distribution, len(results))
        }
    
    # UTILITY AND HELPER METHODS
    
    def _get_strongest_category(self, category_scores: Dict[str, float]) -> str:
        """Identify the strongest performing category"""
        return max(category_scores.items(), key=lambda x: x[1])[0].replace('_', ' ').title()
    
    def _get_weakest_category(self, category_scores: Dict[str, float]) -> str:
        """Identify the weakest performing category"""
        return min(category_scores.items(), key=lambda x: x[1])[0].replace('_', ' ').title()
    
    def _assess_risk_level(self, metrics: Dict[str, float], 
                          category_scores: Dict[str, float]) -> str:
        """Assess overall risk level using both metrics and scores"""
        risk_indicators = []
        
        # Score-based risk assessment
        if category_scores['financial_health_score'] < 40:
            risk_indicators.append('poor_financial_health')
        if category_scores['profitability_score'] < 30:
            risk_indicators.append('low_profitability')
        if category_scores['overall_score'] < 40:
            risk_indicators.append('overall_weakness')
        
        # Metric-based risk assessment
        debt_ratio = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
        if debt_ratio > 1.5:
            risk_indicators.append('high_debt')
        
        current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
        if current_ratio < 1.0:
            risk_indicators.append('liquidity_risk')
        
        # Determine overall risk
        if len(risk_indicators) >= 3:
            return 'High'
        elif len(risk_indicators) >= 1:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_advanced_confidence_score(self, metrics: Dict[str, float], 
                                           category_scores: Dict[str, float]) -> float:
        """Calculate advanced confidence score"""
        try:
            # Base confidence on data availability
            total_metrics = len([v for v in metrics.values() if isinstance(v, (int, float)) and v != 0])
            
            # Adjust for score consistency
            score_range = max(category_scores.values()) - min(category_scores.values())
            consistency_bonus = max(0, 10 - (score_range / 10)) # Bonus for consistent scores
            
            if total_metrics >= 15:
                base_confidence = 85.0
            elif total_metrics >= 10:
                base_confidence = 75.0
            elif total_metrics >= 6:
                base_confidence = 65.0
            else:
                base_confidence = 50.0
            
            return min(95.0, base_confidence + consistency_bonus)
        except Exception:
            return 60.0
    
    def _extract_enhanced_key_findings(self, metrics: Dict[str, float], 
                                     category_scores: Dict[str, float]) -> List[str]:
        """Extract enhanced key findings"""
        findings = []
        
        try:
            # Overall performance finding
            overall_score = category_scores['overall_score']
            if overall_score >= 80:
                findings.append(f'Exceptional overall performance ({overall_score:.1f}/100)')
            elif overall_score >= 60:
                findings.append(f'Solid financial performance ({overall_score:.1f}/100)')
            else:
                findings.append(f'Performance challenges identified ({overall_score:.1f}/100)')
            
            # Category-specific findings
            strongest = self._get_strongest_category(category_scores)
            weakest = self._get_weakest_category(category_scores)
            
            findings.append(f'Strongest area: {strongest}')
            if category_scores[list(category_scores.keys())[0]] != category_scores[list(category_scores.keys())[-1]]:
                findings.append(f'Improvement opportunity: {weakest}')
            
            # Specific metric findings
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            if roe > 20:
                findings.append('Outstanding return on equity')
            elif roe < 5:
                findings.append('Low return on equity needs attention')
            
        except Exception:
            findings.append('Analysis completed with available data')
        
        return findings[:4]  # Limit to top 4 findings
    
    def _generate_investment_signal(self, category_scores: Dict[str, float]) -> str:
        """Generate investment signal based on category scores"""
        overall_score = category_scores['overall_score']
        
        if overall_score >= 80:
            return 'Strong Buy'
        elif overall_score >= 65:
            return 'Buy'
        elif overall_score >= 50:
            return 'Hold'
        elif overall_score >= 35:
            return 'Weak Hold'
        else:
            return 'Avoid'
    
    def _generate_advanced_recommendation(self, metrics: Dict[str, float], pros: Dict, cons: Dict, 
                                        category_scores: Dict[str, float]) -> str:
        """Generate advanced investment/analysis recommendation"""
        try:
            overall_score = category_scores['overall_score']
            pros_count = pros['pros_count']
            cons_count = cons['cons_count']
            investment_signal = self._generate_investment_signal(category_scores)
            
            if overall_score >= 80:
                return f'{investment_signal} - Exceptional company with strong fundamentals across all categories'
            elif overall_score >= 65:
                return f'{investment_signal} - Solid performer with {pros_count} key strengths outweighing {cons_count} concerns'
            elif overall_score >= 50:
                return f'{investment_signal} - Balanced profile with equal strengths and improvement areas'
            elif overall_score >= 35:
                return f'{investment_signal} - Below-average performer with {cons_count} significant challenges'
            else:
                return f'{investment_signal} - High-risk investment with multiple financial concerns'
        except Exception:
            return 'Neutral - Further detailed analysis recommended'
    
    def _identify_risk_factors(self, metrics: Dict[str, float], 
                             category_scores: Dict[str, float]) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        if category_scores['financial_health_score'] < 40:
            risk_factors.append('Poor financial health indicators')
        
        if category_scores['profitability_score'] < 30:
            risk_factors.append('Low profitability concerns')
        
        if category_scores['growth_score'] < 25:
            risk_factors.append('Weak growth prospects')
        
        debt_ratio = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
        if debt_ratio > 2.0:
            risk_factors.append('High debt burden')
        
        current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
        if current_ratio < 1.0:
            risk_factors.append('Liquidity constraints')
        
        return risk_factors[:3]  # Top 3 risk factors
    
    def _assess_growth_potential(self, category_scores: Dict[str, float]) -> str:
        """Assess growth potential based on category scores"""
        growth_score = category_scores['growth_score']
        efficiency_score = category_scores['efficiency_score']
        
        combined_growth_indicator = (growth_score * 0.7) + (efficiency_score * 0.3)
        
        if combined_growth_indicator >= 70:
            return 'High growth potential'
        elif combined_growth_indicator >= 50:
            return 'Moderate growth potential'
        else:
            return 'Limited growth potential'
    
    def _identify_sector_strengths(self, results: Dict[str, Dict]) -> List[str]:
        """Identify common sector strengths"""
        category_averages = self._get_category_averages(results)
        
        strengths = []
        for category, avg_score in category_averages.items():
            if avg_score >= 65:
                strengths.append(f"Strong {category.replace('_', ' ')}")
        
        return strengths[:3]
    
    def _identify_sector_weaknesses(self, results: Dict[str, Dict]) -> List[str]:
        """Identify common sector weaknesses"""
        category_averages = self._get_category_averages(results)
        
        weaknesses = []
        for category, avg_score in category_averages.items():
            if avg_score < 45:
                weaknesses.append(f"Weak {category.replace('_', ' ')}")
        
        return weaknesses[:3]
    
    def _assess_portfolio_risk(self, risk_distribution: Dict[str, int]) -> str:
        """Assess overall portfolio risk level"""
        total = sum(risk_distribution.values())
        if total == 0:
            return 'Unknown'
        
        high_risk_pct = risk_distribution['high_risk_count'] / total
        low_risk_pct = risk_distribution['low_risk_count'] / total
        
        if high_risk_pct > 0.5:
            return 'High Risk Portfolio'
        elif low_risk_pct > 0.6:
            return 'Low Risk Portfolio'
        else:
            return 'Moderate Risk Portfolio'
    
    def _calculate_diversification_score(self, results: Dict[str, Dict]) -> float:
        """Calculate portfolio diversification score"""
        if len(results) <= 1:
            return 0.0
        
        scores = [result['overall_score'] for result in results.values()]
        score_range = max(scores) - min(scores)
        
        # Higher range indicates better diversification
        diversification_score = min(100, (score_range / 100) * 100)
        return round(diversification_score, 1)
    
    def _generate_portfolio_recommendations(self, risk_distribution: Dict[str, int], 
                                          total_companies: int) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        high_risk_pct = risk_distribution['high_risk_count'] / total_companies
        low_risk_pct = risk_distribution['low_risk_count'] / total_companies
        
        if high_risk_pct > 0.4:
            recommendations.append('Consider reducing high-risk exposure')
        
        if low_risk_pct < 0.3:
            recommendations.append('Add more stable, low-risk investments')
        
        if total_companies < 5:
            recommendations.append('Increase portfolio diversification')
        
        return recommendations
    
    # LEGACY AND ERROR HANDLING METHODS
    
    def analyze_company_complete(self, company_id: str, raw_financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete financial analysis pipeline (backward compatibility method)
        
        Args:
            company_id: Company identifier
            raw_financial_data: Raw financial data from API/source
            
        Returns:
            Complete analysis results including metrics, pros, cons, and insights
        """
        return self.analyze_single_company(company_id, raw_financial_data)
    
    def _create_empty_analysis(self, company_id: str, error_message: str) -> Dict[str, Any]:
        """Create empty analysis result for failed cases"""
        return {
            'company_id': company_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'title': f"{company_id}: Analysis Failed",
            'overall_score': 0,
            'rating': 'Unable to Analyze',
            'category_scores': {
                'profitability': 0,
                'growth': 0,
                'financial_health': 0,
                'efficiency': 0
            },
            'pros': {
                'selected_pros': [],
                'pros_count': 0,
                'pros_criteria': 'Analysis failed'
            },
            'cons': {
                'selected_cons': ['Analysis failed due to insufficient data'],
                'cons_count': 1,
                'cons_criteria': 'Error condition'
            },
            'ml_analysis': {
                'analysis_method': 'Failed analysis',
                'error': error_message,
                'confidence_score': 0
            },
            'insights_summary': {
                'total_insights': 0,
                'summary_points': ['Analysis failed'],
                'recommendation': 'Unable to provide recommendation due to analysis failure'
            },
            'performance_summary': f'Analysis failed: {error_message}'
        }
    
    def _get_error_analysis(self, company_id: str, error_msg: str) -> Dict[str, Any]:
        """Return comprehensive error analysis result"""
        return self._create_empty_analysis(company_id, error_msg)
    
    # EXPORT AND SESSION METHODS
    
    def export_analysis_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to JSON file
        
        Args:
            results: Analysis results
            filename: Output filename (optional)
            
        Returns:
            Filepath of exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return None
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        success_rate = (
            (self.pipeline_stats['successful_analyses'] / 
             max(1, self.pipeline_stats['total_analyses'])) * 100 
            if self.pipeline_stats['total_analyses'] > 0 else 0
        )
        
        return {
            'session_stats': {
                'total_analyses': self.pipeline_stats['total_analyses'],
                'successful_analyses': self.pipeline_stats['successful_analyses'],
                'failed_analyses': self.pipeline_stats['failed_analyses'],
                'success_rate': round(success_rate, 1)
            },
            'analysis_history_count': len(self.analysis_history),
            'latest_analysis': self.analysis_history[-1]['company_id'] if self.analysis_history else None
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed in current session"""
        if not self.analysis_history:
            return {'message': 'No analyses performed yet'}
        
        return {
            'total_analyses': len(self.analysis_history),
            'average_score': round(statistics.mean([a['overall_score'] for a in self.analysis_history]), 2),
            'score_distribution': {
                'excellent': len([a for a in self.analysis_history if a['rating'] == 'Excellent']),
                'good': len([a for a in self.analysis_history if a['rating'] == 'Good']),
                'average': len([a for a in self.analysis_history if a['rating'] == 'Average']),
                'poor': len([a for a in self.analysis_history if a['rating'] == 'Poor']),
                'very_poor': len([a for a in self.analysis_history if a['rating'] == 'Very Poor'])
            },
            'latest_analysis': self.analysis_history[-1]['company_id'],
            'top_performer': max(self.analysis_history, key=lambda x: x['overall_score'])['company_id'],
            'average_by_category': self._get_session_category_averages()
        }
    
    def _get_session_category_averages(self) -> Dict[str, float]:
        """Calculate average category scores for the session"""
        if not self.analysis_history:
            return {}
        
        categories = ['profitability', 'growth', 'financial_health', 'efficiency']
        averages = {}
        
        for category in categories:
            scores = [result['category_scores'][category] for result in self.analysis_history]
            averages[category] = round(statistics.mean(scores), 2)
        
        return averages
    
    def reset_statistics(self):
        """Reset analysis statistics and history"""
        self.pipeline_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': datetime.now()
        }
        self.analysis_history = []
        logger.info("Pipeline statistics and history reset")