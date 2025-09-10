"""
Fixed Financial Analysis Integration Framework
Adds the missing analyze_company method

This module provides:
- Unified interface for financial analysis pipeline
- Error handling and data validation
- Comprehensive analysis workflow
- Results aggregation and formatting
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import random

# Import your existing modules
from data.financial_metrics import FinancialMetricsCalculator

logger = logging.getLogger(__name__)

class FinancialAnalysisPipeline:
    """
    Integrated Financial Analysis Pipeline
    
    Combines metric calculation and ML analysis into a unified workflow
    """
    
    def __init__(self):
        """Initialize the integrated pipeline"""
        self.metrics_calculator = FinancialMetricsCalculator()
        
        self.pipeline_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': datetime.now()
        }
        
        # Pros and cons templates for ML analysis
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
            "Company {} shows consistent financial performance"
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
            "Company {} needs operational improvements"
        ]
        
        logger.info("FinancialAnalysisPipeline initialized")
    
    def analyze_company(self, company_id: str, calculated_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        FIXED: Added the missing method that the debug script expects
        
        Perform ML analysis on calculated financial metrics
        
        Args:
            company_id: Company identifier
            calculated_metrics: Pre-calculated financial metrics
            
        Returns:
            ML analysis results with pros, cons, and insights
        """
        try:
            logger.info(f"Performing ML analysis for {company_id}")
            
            if not calculated_metrics or len(calculated_metrics) == 0:
                return self._create_empty_analysis(company_id, "No metrics provided")
            
            # Generate pros and cons based on metrics
            pros_analysis = self._generate_pros(company_id, calculated_metrics)
            cons_analysis = self._generate_cons(company_id, calculated_metrics)
            
            # Calculate overall scores
            overall_score = self._calculate_overall_score(calculated_metrics)
            
            # Create ML analysis summary
            ml_analysis_summary = self._create_ml_summary(calculated_metrics)
            
            # Generate insights summary
            insights_summary = self._generate_insights_summary(company_id, calculated_metrics, pros_analysis, cons_analysis)
            
            analysis_results = {
                'company_id': company_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'pros': pros_analysis,
                'cons': cons_analysis,
                'overall_score': overall_score,
                'ml_analysis': ml_analysis_summary,
                'insights_summary': insights_summary,
                'title': f"{company_id}: Financial Analysis Report",
                'analysis_confidence': overall_score.get('confidence_level', 'medium')
            }
            
            # Update statistics
            self.pipeline_stats['total_analyses'] += 1
            self.pipeline_stats['successful_analyses'] += 1
            
            logger.info(f"ML analysis completed for {company_id}: {len(pros_analysis['selected_pros'])} pros, {len(cons_analysis['selected_cons'])} cons")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in ML analysis for {company_id}: {e}")
            self.pipeline_stats['total_analyses'] += 1
            self.pipeline_stats['failed_analyses'] += 1
            return self._create_empty_analysis(company_id, str(e))
    
    def _generate_pros(self, company_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate pros based on financial metrics"""
        pros = []
        
        try:
            # Check ROE
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            if roe > 15:
                pros.append(self.pros_templates[0].format(company_id, roe))
            elif roe > 10:
                pros.append(self.pros_templates[6])  # Strong profitability
            
            # Check current ratio
            current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
            if current_ratio > 1.5:
                pros.append(self.pros_templates[2].format(company_id, current_ratio))
            elif current_ratio > 1.2:
                pros.append(self.pros_templates[7])  # Good liquidity
            
            # Check debt levels
            debt_to_equity = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
            if 0 < debt_to_equity < 0.5:
                pros.append(self.pros_templates[3].format(company_id, debt_to_equity))
            elif debt_to_equity < 1.0:
                pros.append(self.pros_templates[8])  # Conservative debt
            
            # Check financial health score
            financial_health = metrics.get('financial_health_score', 0)
            if financial_health > 70:
                pros.append(self.pros_templates[4].format(company_id, financial_health))
            
            # Check growth metrics
            revenue_growth = (metrics.get('sales_growth', 0) or 
                            metrics.get('estimated_revenue_growth', 0))
            profit_growth = (metrics.get('profit_growth', 0) or 
                           metrics.get('estimated_profit_growth', 0))
            
            if revenue_growth > 10:
                pros.append(self.pros_templates[5].format(company_id))
            
            if profit_growth > 10:
                pros.append(self.pros_templates[1].format(company_id, profit_growth))
            
            # Check profit margin
            profit_margin = metrics.get('profit_margin', 0) or metrics.get('calculated_profit_margin', 0)
            if profit_margin > 15:
                pros.append(self.pros_templates[6])  # Strong profitability
            
            # If no specific pros found, add general positive statements
            if not pros:
                if financial_health > 50:
                    pros.append(self.pros_templates[9])  # Consistent performance
                else:
                    pros.append("Company shows potential for improvement")
            
            # Limit to maximum 3 pros as per requirements
            selected_pros = pros[:3]
            
        except Exception as e:
            logger.warning(f"Error generating pros for {company_id}: {e}")
            selected_pros = [f"Company {company_id} financial analysis completed"]
        
        return {
            'selected_pros': selected_pros,
            'pros_count': len(selected_pros),
            'pros_criteria': 'Based on financial metrics > threshold values'
        }
    
    def _generate_cons(self, company_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate cons based on financial metrics"""
        cons = []
        
        try:
            # Check ROE
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            if 0 < roe < 8:
                cons.append(self.cons_templates[0].format(company_id, roe))
            elif roe == 0:
                cons.append(self.cons_templates[5])  # Profitability challenges
            
            # Check current ratio
            current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
            if 0 < current_ratio < 1.0:
                cons.append(self.cons_templates[3].format(company_id, current_ratio))
            elif current_ratio == 0:
                cons.append(self.cons_templates[6])  # Liquidity concerns
            
            # Check debt levels
            debt_to_equity = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
            if debt_to_equity > 2.0:
                cons.append(self.cons_templates[2].format(company_id, debt_to_equity))
            elif debt_to_equity > 1.5:
                cons.append(self.cons_templates[7])  # High debt burden
            
            # Check financial health score
            financial_health = metrics.get('financial_health_score', 0)
            if 0 < financial_health < 40:
                cons.append(self.cons_templates[4].format(company_id, financial_health))
            
            # Check growth metrics
            revenue_growth = (metrics.get('sales_growth', 0) or 
                            metrics.get('estimated_revenue_growth', 0))
            if 0 < revenue_growth < 5:
                cons.append(self.cons_templates[1].format(company_id, revenue_growth))
            
            # Check for missing key metrics (indicates data quality issues)
            key_metrics = ['roe', 'current_ratio', 'debt_to_equity', 'profit_margin']
            missing_metrics = 0
            for metric in key_metrics:
                if (metrics.get(metric, 0) == 0 and 
                    metrics.get(f'calculated_{metric}', 0) == 0):
                    missing_metrics += 1
            
            if missing_metrics >= 2:
                cons.append(self.cons_templates[8])  # Inconsistent performance
            
            # If no specific cons found, add general areas for improvement
            if not cons:
                if financial_health < 60:
                    cons.append(self.cons_templates[9])  # Needs improvements
                else:
                    cons.append("Company could optimize operational efficiency")
            
            # Limit to maximum 3 cons as per requirements
            selected_cons = cons[:3]
            
        except Exception as e:
            logger.warning(f"Error generating cons for {company_id}: {e}")
            selected_cons = [f"Company {company_id} requires detailed analysis"]
        
        return {
            'selected_cons': selected_cons,
            'cons_count': len(selected_cons),
            'cons_criteria': 'Based on financial metrics < threshold values'
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate overall analysis score"""
        try:
            # Get key scores
            financial_health = metrics.get('financial_health_score', 0)
            profitability_score = metrics.get('profitability_score', 0)
            liquidity_score = metrics.get('liquidity_score', 0)
            stability_score = metrics.get('stability_score', 0)
            growth_score = metrics.get('growth_score', 0)
            
            # Calculate weighted final score
            weights = {
                'financial_health_score': 0.3,
                'profitability_score': 0.25,
                'liquidity_score': 0.15,
                'stability_score': 0.15,
                'growth_score': 0.15
            }
            
            final_score = (
                financial_health * weights['financial_health_score'] +
                profitability_score * weights['profitability_score'] +
                liquidity_score * weights['liquidity_score'] +
                stability_score * weights['stability_score'] +
                growth_score * weights['growth_score']
            )
            
            # Determine confidence level
            non_zero_scores = sum(1 for score in [financial_health, profitability_score, 
                                               liquidity_score, stability_score, growth_score] if score > 0)
            
            if non_zero_scores >= 4:
                confidence_level = 'high'
            elif non_zero_scores >= 2:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            return {
                'final_score': round(final_score, 1),
                'confidence_level': confidence_level,
                'component_scores': {
                    'financial_health': financial_health,
                    'profitability': profitability_score,
                    'liquidity': liquidity_score,
                    'stability': stability_score,
                    'growth': growth_score
                },
                'score_interpretation': self._interpret_score(final_score)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating overall score: {e}")
            return {
                'final_score': 50.0,
                'confidence_level': 'low',
                'calculation_error': str(e)
            }
    
    def _interpret_score(self, score: float) -> str:
        """Interpret the overall financial score"""
        if score >= 80:
            return 'Excellent financial position'
        elif score >= 65:
            return 'Good financial health'
        elif score >= 50:
            return 'Average financial performance'
        elif score >= 35:
            return 'Below average financial metrics'
        else:
            return 'Weak financial position'
    
    def _create_ml_summary(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create ML analysis summary"""
        return {
            'analysis_method': 'Rule-based financial analysis',
            'metrics_analyzed': len(metrics),
            'risk_assessment': self._assess_risk_level(metrics),
            'confidence_score': self._calculate_confidence_score(metrics),
            'key_findings': self._extract_key_findings(metrics)
        }
    
    def _assess_risk_level(self, metrics: Dict[str, float]) -> str:
        """Assess overall risk level"""
        try:
            risk_indicators = []
            
            # High debt indicates higher risk
            debt_ratio = metrics.get('debt_to_equity', 0) or metrics.get('calculated_debt_to_equity', 0)
            if debt_ratio > 1.5:
                risk_indicators.append('high_debt')
            
            # Low liquidity indicates higher risk
            current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
            if current_ratio < 1.0:
                risk_indicators.append('low_liquidity')
            
            # Low profitability indicates higher risk
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            if roe < 10:
                risk_indicators.append('low_profitability')
            
            # Determine overall risk
            if len(risk_indicators) >= 2:
                return 'high'
            elif len(risk_indicators) == 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'
    
    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence in the analysis"""
        try:
            # Base confidence on number of available metrics
            total_metrics = len([v for v in metrics.values() if isinstance(v, (int, float)) and v != 0])
            
            if total_metrics >= 10:
                return 85.0
            elif total_metrics >= 6:
                return 70.0
            elif total_metrics >= 3:
                return 55.0
            else:
                return 40.0
                
        except Exception:
            return 50.0
    
    def _extract_key_findings(self, metrics: Dict[str, float]) -> List[str]:
        """Extract key findings from metrics"""
        findings = []
        
        try:
            financial_health = metrics.get('financial_health_score', 0)
            if financial_health > 70:
                findings.append('Strong overall financial health')
            elif financial_health < 40:
                findings.append('Financial health needs improvement')
            
            roe = metrics.get('roe', 0) or metrics.get('calculated_roe', 0)
            if roe > 15:
                findings.append('Excellent return on equity')
            elif roe < 8:
                findings.append('Low return on equity')
            
            current_ratio = metrics.get('current_ratio', 0) or metrics.get('calculated_current_ratio', 0)
            if current_ratio > 2:
                findings.append('Strong liquidity position')
            elif current_ratio < 1:
                findings.append('Liquidity concerns present')
            
        except Exception:
            findings.append('Analysis completed with available data')
        
        return findings[:3]  # Limit to top 3 findings
    
    def _generate_insights_summary(self, company_id: str, metrics: Dict[str, float], 
                                 pros: Dict, cons: Dict) -> Dict[str, Any]:
        """Generate comprehensive insights summary"""
        return {
            'total_insights': len(pros['selected_pros']) + len(cons['selected_cons']),
            'summary_points': [
                f"Analysis identified {len(pros['selected_pros'])} key strengths",
                f"Analysis identified {len(cons['selected_cons'])} areas for improvement",
                f"Financial health score: {metrics.get('financial_health_score', 0):.1f}/100"
            ],
            'recommendation': self._generate_recommendation(metrics, pros, cons)
        }
    
    def _generate_recommendation(self, metrics: Dict[str, float], pros: Dict, cons: Dict) -> str:
        """Generate investment/analysis recommendation"""
        try:
            financial_health = metrics.get('financial_health_score', 0)
            pros_count = pros['pros_count']
            cons_count = cons['cons_count']
            
            if financial_health >= 70 and pros_count >= 2:
                return 'Positive outlook - Company shows strong financial metrics'
            elif financial_health >= 50 and pros_count >= cons_count:
                return 'Moderate outlook - Company has balanced strengths and weaknesses'
            elif cons_count > pros_count:
                return 'Cautious outlook - Company faces several challenges'
            else:
                return 'Neutral outlook - Further analysis recommended'
                
        except Exception:
            return 'Analysis completed - Review detailed metrics for decision making'
    
    def _create_empty_analysis(self, company_id: str, error_message: str) -> Dict[str, Any]:
        """Create empty analysis result for failed cases"""
        return {
            'company_id': company_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'pros': {
                'selected_pros': [],
                'pros_count': 0,
                'pros_criteria': 'Analysis failed'
            },
            'cons': {
                'selected_cons': [],
                'cons_count': 0,
                'cons_criteria': 'Analysis failed'
            },
            'overall_score': {
                'final_score': 0,
                'confidence_level': 'low',
                'error': error_message
            },
            'ml_analysis': {
                'analysis_method': 'Failed analysis',
                'error': error_message
            },
            'insights_summary': {
                'total_insights': 0,
                'summary_points': ['Analysis failed'],
                'recommendation': 'Unable to provide recommendation due to analysis failure'
            },
            'title': f"{company_id}: Analysis Failed"
        }
    
    # Keep your existing methods for backward compatibility
    def analyze_company_complete(self, company_id: str, raw_financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete financial analysis pipeline (original method)
        
        Args:
            company_id: Company identifier
            raw_financial_data: Raw financial data from API/source
            
        Returns:
            Complete analysis results including metrics, pros, cons, and insights
        """
        analysis_start = datetime.now()
        
        try:
            logger.info(f"Starting complete financial analysis for {company_id}")
            
            # Step 1: Calculate financial metrics
            logger.info("Step 1: Calculating financial metrics...")
            calculated_metrics = self.metrics_calculator.calculate_comprehensive_metrics(raw_financial_data)
            
            if not calculated_metrics or 'calculation_error' in calculated_metrics:
                error_msg = calculated_metrics.get('calculation_error', 'Metrics calculation failed')
                return self._create_failed_analysis(company_id, error_msg, analysis_start)
            
            # Step 2: Perform ML analysis using the new method
            logger.info("Step 2: Performing ML analysis...")
            analysis_results = self.analyze_company(company_id, calculated_metrics)
            
            if 'error' in analysis_results:
                return self._create_failed_analysis(company_id, analysis_results['error'], analysis_start)
            
            # Step 3: Combine results into comprehensive report
            logger.info("Step 3: Generating comprehensive report...")
            comprehensive_results = self._create_comprehensive_report(
                company_id, raw_financial_data, calculated_metrics, analysis_results, analysis_start
            )
            
            return comprehensive_results
            
        except Exception as e:
            error_msg = f"Pipeline error for {company_id}: {str(e)}"
            logger.error(error_msg)
            return self._create_failed_analysis(company_id, error_msg, analysis_start)
    
    def _create_comprehensive_report(self, company_id: str, raw_data: Dict[str, Any], 
                                calculated_metrics: Dict[str, float], 
                                analysis_results: Dict[str, Any],
                                analysis_start: datetime) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        
        comprehensive_report = {
            # Basic Information
            'company_id': company_id,
            'report_timestamp': datetime.now().isoformat(),
            'analysis_duration_seconds': analysis_duration,
            'report_type': 'comprehensive_financial_analysis',
            
            # Core Results - use the results from analyze_company
            'calculated_metrics': calculated_metrics,
            'ml_analysis': analysis_results.get('ml_analysis', {}),
            'pros': analysis_results.get('pros', {}),
            'cons': analysis_results.get('cons', {}),
            'title': analysis_results.get('title', f"{company_id}: Financial Analysis"),
            'overall_score': analysis_results.get('overall_score', {}),
            'insights_summary': analysis_results.get('insights_summary', {})
        }
        
        return comprehensive_report
    
    def _create_failed_analysis(self, company_id: str, error_message: str, 
                            analysis_start: datetime) -> Dict[str, Any]:
        """Create result structure for failed analysis"""
        return {
            'company_id': company_id,
            'report_timestamp': datetime.now().isoformat(),
            'analysis_duration_seconds': (datetime.now() - analysis_start).total_seconds(),
            'report_type': 'failed_analysis',
            'error': error_message,
            'calculated_metrics': {},
            'ml_analysis': {},
            'pros': {'selected_pros': [], 'pros_count': 0},
            'cons': {'selected_cons': [], 'cons_count': 0},
            'title': f"{company_id}: Analysis Failed",
            'overall_score': {'final_score': 0, 'confidence_level': 'low'},
            'insights_summary': {'summary_points': [], 'total_insights': 0}
        }
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return {
            'total_analyses': self.pipeline_stats['total_analyses'],
            'successful_analyses': self.pipeline_stats['successful_analyses'],
            'failed_analyses': self.pipeline_stats['failed_analyses'],
            'success_rate': (
                self.pipeline_stats['successful_analyses'] / 
                max(1, self.pipeline_stats['total_analyses'])
            ) * 100 if self.pipeline_stats['total_analyses'] > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset analysis statistics"""
        self.pipeline_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': datetime.now()
        }