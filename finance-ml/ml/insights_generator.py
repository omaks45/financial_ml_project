# ml/insights_generator.py
import random
from typing import Dict, List, Tuple, Any

class FinancialInsightsGenerator:
    """
    Generates intelligent financial insights with pros/cons categorization
    and dynamic title generation for companies.
    """
    
    def __init__(self):
        # Pros templates with threshold conditions
        self.pros_templates = {
            'debt_free': {
                'template': "Company is almost debt-free",
                'conditions': ['debt_equity_ratio', '<', 0.1],
                'weight': 10
            },
            'debt_reduced': {
                'template': "Company has reduced debt significantly",
                'conditions': ['debt_reduction_rate', '>', 15.0],
                'weight': 8
            },
            'good_roe': {
                'template': "Company has a good return on equity (ROE) track record: {roe_3y}% over 3 years",
                'conditions': ['roe_3y', '>', 15.0],
                'weight': 9
            },
            'high_roe': {
                'template': "Company has excellent return on equity (ROE) track record: {roe_3y}% over 3 years",
                'conditions': ['roe_3y', '>', 25.0],
                'weight': 10
            },
            'healthy_dividend': {
                'template': "Company has been maintaining a healthy dividend payout of {dividend_payout}%",
                'conditions': ['dividend_payout', '>', 10.0, 'dividend_payout', '<', 80.0],
                'weight': 7
            },
            'good_profit_growth': {
                'template': "Company has delivered good profit growth of {profit_growth}%",
                'conditions': ['profit_growth', '>', 15.0],
                'weight': 9
            },
            'excellent_profit_growth': {
                'template': "Company has delivered excellent profit growth of {profit_growth}%",
                'conditions': ['profit_growth', '>', 25.0],
                'weight': 10
            },
            'strong_sales_growth': {
                'template': "Company's sales growth is impressive at {sales_growth_10y}% over last 10 years",
                'conditions': ['sales_growth_10y', '>', 20.0],
                'weight': 9
            },
            'consistent_sales': {
                'template': "Company's median sales growth is {sales_growth_10y}% over last 10 years",
                'conditions': ['sales_growth_10y', '>', 12.0],
                'weight': 7
            },
            'strong_margins': {
                'template': "Company maintains strong profit margins of {profit_margin}%",
                'conditions': ['profit_margin', '>', 15.0],
                'weight': 8
            },
            'cash_rich': {
                'template': "Company has strong cash reserves and liquidity position",
                'conditions': ['cash_ratio', '>', 0.2],
                'weight': 7
            },
            'asset_efficiency': {
                'template': "Company shows excellent asset utilization efficiency",
                'conditions': ['asset_turnover', '>', 1.5],
                'weight': 6
            }
        }
        
        # Cons templates with threshold conditions
        self.cons_templates = {
            'poor_sales_growth_5y': {
                'template': "The company has delivered poor sales growth of {sales_growth_5y}% over past five years",
                'conditions': ['sales_growth_5y', '<', 10.0],
                'weight': 8
            },
            'no_dividend': {
                'template': "Company is not paying out dividend to shareholders",
                'conditions': ['dividend_payout', '=', 0],
                'weight': 6
            },
            'low_dividend': {
                'template': "Company has low dividend payout of {dividend_payout}%",
                'conditions': ['dividend_payout', '>', 0, 'dividend_payout', '<', 5.0],
                'weight': 5
            },
            'low_roe': {
                'template': "Company has a low return on equity of {roe_3y}% over last 3 years",
                'conditions': ['roe_3y', '<', 12.0],
                'weight': 9
            },
            'very_low_roe': {
                'template': "Company has a concerning low return on equity of {roe_3y}% over last 3 years",
                'conditions': ['roe_3y', '<', 8.0],
                'weight': 10
            },
            'high_debt': {
                'template': "Company has high debt levels with debt-to-equity ratio of {debt_equity_ratio}",
                'conditions': ['debt_equity_ratio', '>', 1.0],
                'weight': 9
            },
            'declining_profits': {
                'template': "Company shows declining profit trend of {profit_growth}%",
                'conditions': ['profit_growth', '<', 0],
                'weight': 10
            },
            'poor_profit_growth': {
                'template': "Company has delivered poor profit growth of {profit_growth}%",
                'conditions': ['profit_growth', '>', 0, 'profit_growth', '<', 8.0],
                'weight': 7
            },
            'weak_margins': {
                'template': "Company has weak profit margins of {profit_margin}%",
                'conditions': ['profit_margin', '<', 8.0],
                'weight': 8
            },
            'cash_crunch': {
                'template': "Company shows signs of cash flow constraints",
                'conditions': ['cash_ratio', '<', 0.1],
                'weight': 9
            },
            'asset_inefficiency': {
                'template': "Company shows poor asset utilization efficiency",
                'conditions': ['asset_turnover', '<', 0.8],
                'weight': 7
            }
        }
    
    def calculate_financial_metrics(self, financial_data: Dict) -> Dict[str, float]:
        """
        Calculate key financial metrics from raw financial data.
        
        Args:
            financial_data (Dict): Raw financial data from API
            
        Returns:
            Dict[str, float]: Calculated financial metrics
        """
        try:
            # Extract data from different statements
            balance_sheet = financial_data.get('balance_sheet', {})
            profit_loss = financial_data.get('profit_loss', {})
            cash_flow = financial_data.get('cash_flow', {})
            
            # Calculate metrics (mock calculations - adapt based on actual API structure)
            metrics = {}
            
            # ROE Calculation
            net_income = profit_loss.get('net_income', 0)
            shareholders_equity = balance_sheet.get('shareholders_equity', 1)
            metrics['roe_3y'] = (net_income / shareholders_equity) * 100 if shareholders_equity != 0 else 0
            
            # Debt to Equity Ratio
            total_debt = balance_sheet.get('total_debt', 0)
            metrics['debt_equity_ratio'] = total_debt / shareholders_equity if shareholders_equity != 0 else 0
            
            # Sales Growth (mock calculation)
            current_revenue = profit_loss.get('revenue', 0)
            previous_revenue = profit_loss.get('previous_revenue', current_revenue)
            metrics['sales_growth_5y'] = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue != 0 else 0
            metrics['sales_growth_10y'] = metrics['sales_growth_5y'] * 1.2  # Mock 10-year growth
            
            # Profit Growth
            current_profit = profit_loss.get('net_income', 0)
            previous_profit = profit_loss.get('previous_net_income', current_profit)
            metrics['profit_growth'] = ((current_profit - previous_profit) / previous_profit * 100) if previous_profit != 0 else 0
            
            # Dividend Payout
            dividends_paid = cash_flow.get('dividends_paid', 0)
            metrics['dividend_payout'] = (abs(dividends_paid) / net_income * 100) if net_income > 0 else 0
            
            # Profit Margin
            metrics['profit_margin'] = (net_income / current_revenue * 100) if current_revenue != 0 else 0
            
            # Cash Ratio
            cash_equivalents = balance_sheet.get('cash_and_equivalents', 0)
            current_liabilities = balance_sheet.get('current_liabilities', 1)
            metrics['cash_ratio'] = cash_equivalents / current_liabilities if current_liabilities != 0 else 0
            
            # Asset Turnover
            total_assets = balance_sheet.get('total_assets', 1)
            metrics['asset_turnover'] = current_revenue / total_assets if total_assets != 0 else 0
            
            # Debt Reduction Rate (mock calculation)
            previous_debt = balance_sheet.get('previous_total_debt', total_debt)
            metrics['debt_reduction_rate'] = ((previous_debt - total_debt) / previous_debt * 100) if previous_debt != 0 else 0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating financial metrics: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics for testing purposes"""
        return {
            'roe_3y': 15.5,
            'debt_equity_ratio': 0.3,
            'sales_growth_5y': 12.0,
            'sales_growth_10y': 18.5,
            'profit_growth': 20.0,
            'dividend_payout': 45.0,
            'profit_margin': 12.5,
            'cash_ratio': 0.25,
            'asset_turnover': 1.2,
            'debt_reduction_rate': 5.0
        }
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a condition based on operator"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '=':
            return abs(value - threshold) < 0.01  # For floating point comparison
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        return False
    
    def _check_template_conditions(self, template_data: Dict, metrics: Dict[str, float]) -> bool:
        """Check if a template's conditions are met"""
        conditions = template_data['conditions']
        
        # Process conditions in pairs (metric, operator, value)
        for i in range(0, len(conditions), 3):
            if i + 2 >= len(conditions):
                break
                
            metric_name = conditions[i]
            operator = conditions[i + 1]
            threshold = conditions[i + 2]
            
            if metric_name not in metrics:
                continue
                
            metric_value = metrics[metric_name]
            if not self._evaluate_condition(metric_value, operator, threshold):
                return False
        
        return True
    
    def generate_pros(self, metrics: Dict[str, float], max_pros: int = 3) -> List[str]:
        """
        Generate pros based on financial metrics.
        
        Args:
            metrics (Dict[str, float]): Calculated financial metrics
            max_pros (int): Maximum number of pros to return
            
        Returns:
            List[str]: List of pros statements
        """
        applicable_pros = []
        
        for pros_key, template_data in self.pros_templates.items():
            if self._check_template_conditions(template_data, metrics):
                # Format template with actual values
                formatted_template = template_data['template'].format(**{
                    k: round(v, 1) for k, v in metrics.items()
                })
                
                applicable_pros.append({
                    'text': formatted_template,
                    'weight': template_data['weight'],
                    'key': pros_key
                })
        
        # Sort by weight (highest first) and select top ones
        applicable_pros.sort(key=lambda x: x['weight'], reverse=True)
        
        # Remove duplicates and select best ones
        selected_pros = []
        used_keys = set()
        
        for pro in applicable_pros:
            if pro['key'] not in used_keys and len(selected_pros) < max_pros:
                selected_pros.append(pro['text'])
                used_keys.add(pro['key'])
        
        return selected_pros
    
    def generate_cons(self, metrics: Dict[str, float], max_cons: int = 3) -> List[str]:
        """
        Generate cons based on financial metrics.
        
        Args:
            metrics (Dict[str, float]): Calculated financial metrics
            max_cons (int): Maximum number of cons to return
            
        Returns:
            List[str]: List of cons statements
        """
        applicable_cons = []
        
        for cons_key, template_data in self.cons_templates.items():
            if self._check_template_conditions(template_data, metrics):
                # Format template with actual values
                formatted_template = template_data['template'].format(**{
                    k: round(v, 1) for k, v in metrics.items()
                })
                
                applicable_cons.append({
                    'text': formatted_template,
                    'weight': template_data['weight'],
                    'key': cons_key
                })
        
        # Sort by weight (highest first) and select top ones
        applicable_cons.sort(key=lambda x: x['weight'], reverse=True)
        
        # Remove duplicates and select best ones
        selected_cons = []
        used_keys = set()
        
        for con in applicable_cons:
            if con['key'] not in used_keys and len(selected_cons) < max_cons:
                selected_cons.append(con['text'])
                used_keys.add(con['key'])
        
        return selected_cons
    
    def generate_dynamic_title(self, company_id: str, metrics: Dict[str, float]) -> str:
        """
        Generate dynamic title based on company performance and metrics.
        
        Args:
            company_id (str): Company identifier
            metrics (Dict[str, float]): Financial metrics
            
        Returns:
            str: Dynamic title for the analysis
        """
        # Calculate overall performance score
        score = 0
        total_weight = 0
        
        # Positive indicators
        if metrics.get('roe_3y', 0) > 15:
            score += 20
            total_weight += 20
        elif metrics.get('roe_3y', 0) > 10:
            score += 10
            total_weight += 20
        else:
            total_weight += 20
            
        if metrics.get('profit_growth', 0) > 15:
            score += 20
            total_weight += 20
        elif metrics.get('profit_growth', 0) > 0:
            score += 10
            total_weight += 20
        else:
            total_weight += 20
            
        if metrics.get('debt_equity_ratio', 1) < 0.3:
            score += 15
            total_weight += 15
        elif metrics.get('debt_equity_ratio', 1) < 0.7:
            score += 8
            total_weight += 15
        else:
            total_weight += 15
            
        if metrics.get('sales_growth_5y', 0) > 12:
            score += 15
            total_weight += 15
        elif metrics.get('sales_growth_5y', 0) > 5:
            score += 8
            total_weight += 15
        else:
            total_weight += 15
            
        if metrics.get('dividend_payout', 0) > 10:
            score += 10
            total_weight += 10
        else:
            total_weight += 10
        
        # Calculate performance percentage
        performance_pct = (score / total_weight * 100) if total_weight > 0 else 0
        
        # Generate title based on performance
        company_name = company_id.upper()
        
        if performance_pct >= 80:
            title_templates = [
                f"{company_name} - Exceptional Financial Performance Analysis",
                f"{company_name} - Strong Market Leader Analysis",
                f"{company_name} - Outstanding Growth & Profitability Review"
            ]
        elif performance_pct >= 60:
            title_templates = [
                f"{company_name} - Solid Financial Performance Analysis",
                f"{company_name} - Stable Growth Company Review",
                f"{company_name} - Good Investment Potential Analysis"
            ]
        elif performance_pct >= 40:
            title_templates = [
                f"{company_name} - Mixed Financial Performance Analysis",
                f"{company_name} - Moderate Performance Review",
                f"{company_name} - Balanced Risk-Return Analysis"
            ]
        else:
            title_templates = [
                f"{company_name} - Challenging Financial Metrics Analysis",
                f"{company_name} - Areas for Improvement Review",
                f"{company_name} - Cautious Investment Analysis"
            ]
        
        return random.choice(title_templates)
    
    def generate_complete_insights(self, company_id: str, financial_data: Dict) -> Dict[str, Any]:
        """
        Generate complete insights including metrics, pros, cons, and title.
        
        Args:
            company_id (str): Company identifier
            financial_data (Dict): Raw financial data
            
        Returns:
            Dict[str, Any]: Complete insights package
        """
        try:
            # Calculate metrics
            metrics = self.calculate_financial_metrics(financial_data)
            
            # Generate insights
            pros = self.generate_pros(metrics)
            cons = self.generate_cons(metrics)
            title = self.generate_dynamic_title(company_id, metrics)
            
            return {
                'company_id': company_id,
                'title': title,
                'metrics': metrics,
                'pros': pros,
                'cons': cons,
                'analysis_date': financial_data.get('analysis_date', 'N/A'),
                'performance_score': self._calculate_performance_score(metrics)
            }
            
        except Exception as e:
            print(f"Error generating insights for {company_id}: {e}")
            return self._get_default_insights(company_id)
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-100)"""
        score = 0
        max_score = 100
        
        # ROE contribution (25 points)
        roe = metrics.get('roe_3y', 0)
        if roe > 20:
            score += 25
        elif roe > 15:
            score += 20
        elif roe > 10:
            score += 15
        elif roe > 5:
            score += 10
        
        # Growth contribution (25 points)
        profit_growth = metrics.get('profit_growth', 0)
        if profit_growth > 20:
            score += 25
        elif profit_growth > 15:
            score += 20
        elif profit_growth > 10:
            score += 15
        elif profit_growth > 0:
            score += 10
        
        # Debt management (25 points)
        debt_ratio = metrics.get('debt_equity_ratio', 1)
        if debt_ratio < 0.2:
            score += 25
        elif debt_ratio < 0.5:
            score += 20
        elif debt_ratio < 1.0:
            score += 15
        elif debt_ratio < 1.5:
            score += 10
        
        # Sales growth (25 points)
        sales_growth = metrics.get('sales_growth_5y', 0)
        if sales_growth > 15:
            score += 25
        elif sales_growth > 10:
            score += 20
        elif sales_growth > 5:
            score += 15
        elif sales_growth > 0:
            score += 10
        
        return min(score, max_score)
    
    def _get_default_insights(self, company_id: str) -> Dict[str, Any]:
        """Return default insights when analysis fails"""
        return {
            'company_id': company_id,
            'title': f"{company_id.upper()} - Financial Analysis",
            'metrics': self._get_default_metrics(),
            'pros': ["Data analysis in progress"],
            'cons': ["Complete financial data not available"],
            'analysis_date': 'N/A',
            'performance_score': 50.0
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the insights generator
    generator = FinancialInsightsGenerator()
    
    # Sample financial data (mock structure)
    sample_data = {
        'balance_sheet': {
            'total_debt': 50000,
            'shareholders_equity': 200000,
            'cash_and_equivalents': 30000,
            'current_liabilities': 80000,
            'total_assets': 300000,
            'previous_total_debt': 60000
        },
        'profit_loss': {
            'revenue': 500000,
            'net_income': 40000,
            'previous_revenue': 450000,
            'previous_net_income': 35000
        },
        'cash_flow': {
            'dividends_paid': 18000
        },
        'analysis_date': '2024-01-15'
    }
    
    # Test the insights generation
    print("Testing Financial Insights Generator")
    print("=" * 50)
    
    insights = generator.generate_complete_insights("TCS", sample_data)
    
    print(f"Company: {insights['company_id']}")
    print(f"Title: {insights['title']}")
    print(f"Performance Score: {insights['performance_score']:.1f}/100")
    print()
    
    print("Financial Metrics:")
    for metric, value in insights['metrics'].items():
        print(f"  {metric}: {value:.2f}")
    print()
    
    print("Pros:")
    for i, pro in enumerate(insights['pros'], 1):
        print(f"  {i}. {pro}")
    print()
    
    print("Cons:")
    for i, con in enumerate(insights['cons'], 1):
        print(f"  {i}. {con}")
    print()
    
    # Test with multiple scenarios
    print("\nTesting Different Scenarios:")
    print("=" * 30)
    
    # High performance company
    high_perf_data = sample_data.copy()
    high_perf_data['profit_loss']['net_income'] = 60000
    high_perf_data['balance_sheet']['total_debt'] = 10000
    
    insights_high = generator.generate_complete_insights("HDFCBANK", high_perf_data)
    print(f"High Performance - {insights_high['title']}")
    print(f"Score: {insights_high['performance_score']:.1f}")
    print("Pros:", len(insights_high['pros']))
    print("Cons:", len(insights_high['cons']))
    print()
    
    # Low performance company
    low_perf_data = sample_data.copy()
    low_perf_data['profit_loss']['net_income'] = 5000
    low_perf_data['balance_sheet']['total_debt'] = 250000
    
    insights_low = generator.generate_complete_insights("COMPANY_X", low_perf_data)
    print(f"Low Performance - {insights_low['title']}")
    print(f"Score: {insights_low['performance_score']:.1f}")
    print("Pros:", len(insights_low['pros']))
    print("Cons:", len(insights_low['cons']))