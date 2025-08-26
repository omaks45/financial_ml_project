"""
Data Cleaning and Preprocessing Pipeline

This module handles:
- Raw financial data cleaning
- Data type conversion and validation
- Missing value handling
- Data quality assessment
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Implementation: Data Cleaning and Preprocessing

    Processes raw financial data from API into clean, structured format
    suitable for ML analysis and database storage.
    """
    
    def __init__(self):
        """Initialize data processor with cleaning rules and validation criteria"""
        self.required_sections = [
            'company_info', 'financial_metrics', 'balance_sheet', 
            'profit_loss', 'cash_flow', 'ratios'
        ]
        
        self.cleaning_stats = {
            'total_processed': 0,
            'cleaning_errors': 0,
            'quality_issues': 0
        }
        
        logger.info("DataProcessor initialized for Day 3 Task 1")
    
    def clean_financial_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main cleaning function - processes raw API data
        
        Args:
            raw_data: Raw financial data from API
            
        Returns:
            Dictionary with cleaned and structured data
        """
        if not raw_data:
            logger.warning("No raw data provided for cleaning")
            return self._create_empty_result("No raw data provided")
        
        try:
            logger.info("Starting financial data cleaning process...")
            
            # Initialize cleaned data structure
            cleaned_data = {
                'company_id': self._extract_company_id(raw_data),
                'processing_timestamp': datetime.now().isoformat(),
                'cleaning_metadata': {
                    'original_data_size': len(str(raw_data)),
                    'processing_version': '1.0',
                    'cleaning_rules_applied': []
                }
            }
            
            # Clean each section of financial data
            cleaned_sections = {}
            cleaning_rules_applied = []
            
            # 1. Clean company information
            company_info = self._clean_company_info(raw_data.get('company_info', {}))
            cleaned_sections['company_info'] = company_info
            cleaning_rules_applied.append('company_info_standardization')
            
            # 2. Clean financial metrics
            financial_metrics = self._clean_financial_metrics(raw_data.get('financial_metrics', {}))
            cleaned_sections['financial_metrics'] = financial_metrics
            cleaning_rules_applied.append('financial_metrics_normalization')
            
            # 3. Clean financial statements
            for statement_type in ['balance_sheet', 'profit_loss', 'cash_flow']:
                statement_data = raw_data.get(statement_type, {})
                cleaned_statement = self._clean_financial_statement(statement_data, statement_type)
                cleaned_sections[statement_type] = cleaned_statement
                cleaning_rules_applied.append(f'{statement_type}_cleaning')
            
            # 4. Clean ratios and growth metrics
            ratios = self._clean_ratios(raw_data.get('ratios', {}))
            growth_metrics = self._clean_growth_metrics(raw_data.get('growth_metrics', {}))
            cleaned_sections['ratios'] = ratios
            cleaned_sections['growth_metrics'] = growth_metrics
            cleaning_rules_applied.extend(['ratios_validation', 'growth_metrics_processing'])
            
            # Add cleaned sections to result
            cleaned_data.update(cleaned_sections)
            
            # 5. Assess data quality
            data_quality = self._assess_data_quality(raw_data, cleaned_sections)
            cleaned_data['data_quality'] = data_quality
            
            # 6. Update metadata
            cleaned_data['cleaning_metadata']['cleaning_rules_applied'] = cleaning_rules_applied
            cleaned_data['cleaning_metadata']['sections_processed'] = len(cleaned_sections)
            
            # Update stats
            self.cleaning_stats['total_processed'] += 1
            
            logger.info(f"Data cleaning completed successfully")
            logger.info(f"Data quality score: {data_quality.get('completeness_score', 0):.1f}%")
            
            return cleaned_data
            
        except Exception as e:
            error_msg = f"Error during data cleaning: {str(e)}"
            logger.error(error_msg)
            self.cleaning_stats['cleaning_errors'] += 1
            return self._create_empty_result(error_msg)
    
    def _extract_company_id(self, raw_data: Dict) -> str:
        """Extract company ID from raw data"""
        possible_keys = ['company_id', 'id', 'symbol', 'ticker']
        
        for key in possible_keys:
            if key in raw_data and raw_data[key]:
                return str(raw_data[key]).strip().upper()
        
        # Fallback to company name or unknown
        company_info = raw_data.get('company_info', {})
        if company_info.get('name'):
            return company_info['name'].replace(' ', '').upper()[:10]
        
        return 'UNKNOWN'
    
    def _clean_company_info(self, company_info: Dict) -> Dict:
        """Clean company information section"""
        cleaned = {}
        
        try:
            # Standard company information fields
            cleaned['name'] = self._clean_text_field(company_info.get('name', ''))
            cleaned['sector'] = self._clean_text_field(company_info.get('sector', ''))
            cleaned['industry'] = self._clean_text_field(company_info.get('industry', ''))
            cleaned['market_cap'] = self._clean_numeric_field(company_info.get('market_cap', 0))
            cleaned['listing_date'] = company_info.get('listing_date', '')
            cleaned['exchange'] = self._clean_text_field(company_info.get('exchange', ''))
            
            # Additional fields that might be present
            for field in ['description', 'website', 'employees', 'founded']:
                if field in company_info:
                    if field == 'employees':
                        cleaned[field] = self._clean_numeric_field(company_info[field])
                    else:
                        cleaned[field] = self._clean_text_field(company_info[field])
                        
        except Exception as e:
            logger.warning(f"Error cleaning company info: {e}")
            cleaned['cleaning_error'] = str(e)
        
        return cleaned
    
    def _clean_financial_metrics(self, financial_metrics: Dict) -> Dict:
        """Clean financial metrics section"""
        cleaned = {}
        
        try:
            # Key financial metrics
            metric_fields = [
                'revenue', 'net_income', 'total_assets', 'total_equity', 'total_debt',
                'current_assets', 'current_liabilities', 'cash_and_equivalents',
                'operating_income', 'ebitda', 'free_cash_flow'
            ]
            
            for field in metric_fields:
                if field in financial_metrics:
                    cleaned[field] = self._clean_numeric_field(financial_metrics[field])
            
            # Handle any additional metrics
            for key, value in financial_metrics.items():
                if key not in cleaned:
                    cleaned[key] = self._clean_numeric_field(value)
                    
        except Exception as e:
            logger.warning(f"Error cleaning financial metrics: {e}")
            cleaned['cleaning_error'] = str(e)
        
        return cleaned
    
    def _clean_financial_statement(self, statement: Dict, statement_type: str) -> Dict:
        """Clean individual financial statement (balance sheet, P&L, cash flow)"""
        cleaned = {}
        
        try:
            for key, value in statement.items():
                # Clean the key name
                clean_key = self._standardize_field_name(key)
                
                # Clean the value
                cleaned_value = self._clean_numeric_field(value)
                
                cleaned[clean_key] = cleaned_value
                
        except Exception as e:
            logger.warning(f"Error cleaning {statement_type}: {e}")
            cleaned['cleaning_error'] = str(e)
        
        return cleaned
    
    def _clean_ratios(self, ratios: Dict) -> Dict:
        """Clean financial ratios with validation"""
        cleaned = {}
        
        try:
            ratio_fields = [
                'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_ratio',
                'return_on_equity', 'return_on_assets', 'profit_margin',
                'gross_margin', 'operating_margin', 'asset_turnover'
            ]
            
            for field in ratio_fields:
                if field in ratios:
                    ratio_value = self._clean_numeric_field(ratios[field])
                    
                    # Validate ratio ranges (basic sanity checks)
                    validated_value = self._validate_ratio(field, ratio_value)
                    cleaned[field] = validated_value
            
            # Handle additional ratios
            for key, value in ratios.items():
                if key not in cleaned:
                    cleaned[key] = self._clean_numeric_field(value)
                    
        except Exception as e:
            logger.warning(f"Error cleaning ratios: {e}")
            cleaned['cleaning_error'] = str(e)
        
        return cleaned
    
    def _clean_growth_metrics(self, growth_metrics: Dict) -> Dict:
        """Clean growth metrics"""
        cleaned = {}
        
        try:
            growth_fields = [
                'revenue_growth', 'profit_growth', 'asset_growth',
                'equity_growth', 'eps_growth'
            ]
            
            for field in growth_fields:
                if field in growth_metrics:
                    growth_value = self._clean_numeric_field(growth_metrics[field])
                    
                    # Validate growth rates (reasonable ranges)
                    if abs(growth_value) > 1000:  # > 1000% growth seems unrealistic
                        logger.warning(f"Unusual growth rate for {field}: {growth_value}%")
                        growth_value = min(max(growth_value, -100), 100)  # Cap at ±100%
                    
                    cleaned[field] = growth_value
            
            # Handle additional growth metrics
            for key, value in growth_metrics.items():
                if key not in cleaned:
                    cleaned[key] = self._clean_numeric_field(value)
                    
        except Exception as e:
            logger.warning(f"Error cleaning growth metrics: {e}")
            cleaned['cleaning_error'] = str(e)
        
        return cleaned
    
    def _clean_numeric_field(self, value) -> float:
        """Clean and convert numeric values"""
        if value is None or value == '':
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value) if not (isinstance(value, float) and (value != value)) else 0.0  # Handle NaN
        
        if isinstance(value, str):
            # Remove common formatting characters
            cleaned_str = (value.replace(',', '')
                              .replace('₹', '')
                              .replace('$', '')
                              .replace('%', '')
                              .replace('(', '-')
                              .replace(')', '')
                              .strip())
            
            # Handle special text values
            if cleaned_str.upper() in ['N/A', 'NA', 'NULL', 'NIL', '-', '']:
                return 0.0
            
            try:
                return float(cleaned_str)
            except ValueError:
                logger.warning(f"Could not convert '{value}' to numeric")
                return 0.0
        
        return 0.0
    
    def _clean_text_field(self, value) -> str:
        """Clean text fields"""
        if not value:
            return ''
        
        return str(value).strip().title() if isinstance(value, str) else str(value)
    
    def _standardize_field_name(self, field_name: str) -> str:
        """Standardize field names"""
        if not field_name:
            return 'unknown_field'
        
        # Convert to lowercase and replace spaces/special chars with underscores
        standardized = str(field_name).lower().replace(' ', '_').replace('-', '_')
        
        # Remove special characters
        standardized = ''.join(c for c in standardized if c.isalnum() or c == '_')
        
        return standardized
    
    def _validate_ratio(self, ratio_name: str, value: float) -> float:
        """Validate ratio values for reasonableness"""
        # Define reasonable ranges for common ratios
        validation_ranges = {
            'current_ratio': (0, 10),
            'quick_ratio': (0, 5),
            'debt_to_equity': (0, 10),
            'debt_ratio': (0, 1),
            'return_on_equity': (-100, 100),
            'return_on_assets': (-50, 50),
            'profit_margin': (-50, 50)
        }
        
        if ratio_name in validation_ranges:
            min_val, max_val = validation_ranges[ratio_name]
            if value < min_val or value > max_val:
                logger.warning(f"Ratio {ratio_name} value {value} outside expected range ({min_val}, {max_val})")
                # Don't modify the value, just log the warning
        
        return value
    
    def _assess_data_quality(self, raw_data: Dict, cleaned_data: Dict) -> Dict:
        """Assess the quality of cleaned data"""
        quality_assessment = {
            'completeness_score': 0,
            'missing_sections': [],
            'data_issues': [],
            'quality_grade': 'F'
        }
        
        try:
            # Check completeness of required sections
            total_sections = len(self.required_sections)
            present_sections = 0
            missing_sections = []
            
            for section in self.required_sections:
                if section in cleaned_data and cleaned_data[section]:
                    # Check if section has meaningful data
                    section_data = cleaned_data[section]
                    if isinstance(section_data, dict) and len(section_data) > 0:
                        # Remove error fields from count
                        meaningful_fields = {k: v for k, v in section_data.items() 
                                           if not k.endswith('_error')}
                        if meaningful_fields:
                            present_sections += 1
                        else:
                            missing_sections.append(section)
                    else:
                        missing_sections.append(section)
                else:
                    missing_sections.append(section)
            
            # Calculate completeness score
            completeness_score = (present_sections / total_sections) * 100
            quality_assessment['completeness_score'] = completeness_score
            quality_assessment['missing_sections'] = missing_sections
            
            # Identify data quality issues
            data_issues = []
            
            # Check for cleaning errors in any section
            for section_name, section_data in cleaned_data.items():
                if isinstance(section_data, dict) and 'cleaning_error' in section_data:
                    data_issues.append(f"Cleaning error in {section_name}")
            
            # Check for suspicious values
            financial_metrics = cleaned_data.get('financial_metrics', {})
            if all(v == 0 for v in financial_metrics.values() if isinstance(v, (int, float))):
                data_issues.append("All financial metrics are zero")
            
            # Check company info completeness
            company_info = cleaned_data.get('company_info', {})
            if not company_info.get('name'):
                data_issues.append("Missing company name")
            if not company_info.get('sector'):
                data_issues.append("Missing sector information")
            
            quality_assessment['data_issues'] = data_issues
            
            # Assign quality grade
            if completeness_score >= 90 and len(data_issues) == 0:
                grade = 'A'
            elif completeness_score >= 80 and len(data_issues) <= 1:
                grade = 'B'
            elif completeness_score >= 70 and len(data_issues) <= 2:
                grade = 'C'
            elif completeness_score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            
            quality_assessment['quality_grade'] = grade
            
            # Log quality assessment
            if len(data_issues) > 0:
                self.cleaning_stats['quality_issues'] += 1
                logger.warning(f"Data quality issues found: {data_issues}")
            
        except Exception as e:
            logger.error(f"Error in data quality assessment: {e}")
            quality_assessment['assessment_error'] = str(e)
        
        return quality_assessment
    
    def _create_empty_result(self, error_message: str) -> Dict:
        """Create empty result structure for failed processing"""
        return {
            'company_id': 'UNKNOWN',
            'processing_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'company_info': {},
            'financial_metrics': {},
            'balance_sheet': {},
            'profit_loss': {},
            'cash_flow': {},
            'ratios': {},
            'growth_metrics': {},
            'data_quality': {
                'completeness_score': 0,
                'missing_sections': self.required_sections,
                'data_issues': ['Processing failed'],
                'quality_grade': 'F'
            }
        }
    
    def get_cleaning_statistics(self) -> Dict:
        """Get statistics about data cleaning operations"""
        return {
            'total_processed': self.cleaning_stats['total_processed'],
            'cleaning_errors': self.cleaning_stats['cleaning_errors'],
            'quality_issues': self.cleaning_stats['quality_issues'],
            'success_rate': (
                (self.cleaning_stats['total_processed'] - self.cleaning_stats['cleaning_errors']) 
                / max(1, self.cleaning_stats['total_processed'])
            ) * 100
        }
    
    def reset_statistics(self):
        """Reset cleaning statistics"""
        self.cleaning_stats = {
            'total_processed': 0,
            'cleaning_errors': 0,
            'quality_issues': 0
        }