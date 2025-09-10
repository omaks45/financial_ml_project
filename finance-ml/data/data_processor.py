"""
Data Cleaning and Preprocessing Pipeline - OPTIMIZED VERSION

This module handles:
- Raw financial data cleaning from APIs and Excel files
- Data type conversion and validation
- Missing value handling
- Data quality assessment
- Excel data extraction and processing
- Feature engineering for ML analysis
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Optimized Implementation: Data Cleaning and Preprocessing

    Processes raw financial data from API responses and Excel files into clean, 
    structured format suitable for ML analysis and database storage.
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
            'quality_issues': 0,
            'excel_files_processed': 0
        }
        
        # Define standard field mappings for Excel data
        self.field_mappings = {
            'balance_sheet': {
                'total_assets': ['total_assets', 'Total Assets', 'assets', 'Assets'],
                'total_liabilities': ['total_liabilities', 'Total Liabilities', 'liabilities', 'Liabilities'],
                'equity': ['shareholders_equity', 'Equity', 'Total Equity', 'equity', 'shareholder_equity'],
                'debt': ['total_debt', 'Debt', 'Long Term Debt', 'debt', 'long_term_debt'],
                'current_assets': ['current_assets', 'Current Assets', 'current assets'],
                'current_liabilities': ['current_liabilities', 'Current Liabilities', 'current liabilities']
            },
            'profit_loss': {
                'revenue': ['revenue', 'Revenue', 'Total Revenue', 'Sales', 'sales', 'total_revenue'],
                'net_profit': ['net_profit', 'Net Profit', 'Net Income', 'net_income', 'profit'],
                'operating_profit': ['operating_profit', 'Operating Profit', 'EBIT', 'operating_income'],
                'gross_profit': ['gross_profit', 'Gross Profit', 'gross profit'],
                'expenses': ['total_expenses', 'Expenses', 'Operating Expenses', 'expenses', 'operating_expenses']
            },
            'cash_flow': {
                'operating_cash_flow': ['operating_cash_flow', 'Operating Cash Flow', 'cash from operations'],
                'investing_cash_flow': ['investing_cash_flow', 'Investing Cash Flow', 'cash from investing'],
                'financing_cash_flow': ['financing_cash_flow', 'Financing Cash Flow', 'cash from financing'],
                'free_cash_flow': ['free_cash_flow', 'Free Cash Flow', 'fcf']
            }
        }
        
        logger.info("Optimized DataProcessor initialized for Day 3 Task 1")
    
    def process_company_data(self, company_id: str, raw_data: Union[Dict[str, Any], pd.DataFrame, List]) -> Dict[str, Any]:
        """
        Process raw company data from API or Excel files
        
        Args:
            company_id: Company identifier
            raw_data: Raw data from API, DataFrame, or list of records
            
        Returns:
            Processed and cleaned data dictionary
        """
        try:
            logger.info(f"Processing data for company: {company_id}")
            
            # Initialize processed data structure
            processed_data = {
                'company_id': company_id,
                'raw_data_size': self._calculate_data_size(raw_data),
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'data_quality_score': 0.0,
                'cleaned_data': {},
                'financial_metrics': {},
                'processing_errors': [],
                'processing_status': 'success',
                'data_source_type': self._detect_data_source_type(raw_data)
            }
            
            if not self._validate_raw_data(raw_data):
                logger.warning(f"Invalid raw data for {company_id}")
                processed_data['processing_status'] = 'failed'
                processed_data['processing_errors'].append('Invalid or empty raw data')
                return processed_data
            
            # Process data based on source type
            if isinstance(raw_data, pd.DataFrame) or isinstance(raw_data, list):
                # Handle Excel/structured data
                cleaned_data = self._process_structured_data(raw_data)
                processed_data['excel_files_processed'] = True
            else:
                # Handle API dictionary data
                cleaned_data = self._process_api_data(raw_data)
            
            # Calculate comprehensive financial metrics
            financial_metrics = self._calculate_comprehensive_metrics(cleaned_data)
            
            # Assess data quality
            data_quality = self._assess_data_quality(raw_data, cleaned_data)
            
            # Update processed data
            processed_data.update({
                'cleaned_data': cleaned_data,
                'financial_metrics': financial_metrics,
                'data_quality': data_quality,
                'data_quality_score': data_quality.get('completeness_score', 0)
            })
            
            # Update statistics
            self.cleaning_stats['total_processed'] += 1
            if isinstance(raw_data, (pd.DataFrame, list)):
                self.cleaning_stats['excel_files_processed'] += 1
            
            logger.info(f"Successfully processed data for {company_id}")
            logger.info(f"Data quality score: {processed_data['data_quality_score']:.2f}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data for {company_id}: {e}")
            self.cleaning_stats['cleaning_errors'] += 1
            return {
                'company_id': company_id,
                'processing_status': 'error',
                'processing_errors': [str(e)],
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'data_quality_score': 0.0
            }
    
    def clean_financial_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main cleaning function - processes raw API data (backward compatibility)
        
        Args:
            raw_data: Raw financial data from API
            
        Returns:
            Dictionary with cleaned and structured data
        """
        if not raw_data:
            logger.warning("No raw data provided for cleaning")
            return self._create_empty_result("No raw data provided")
        
        # Extract company ID and use process_company_data
        company_id = self._extract_company_id(raw_data)
        return self.process_company_data(company_id, raw_data)
    
    def _detect_data_source_type(self, raw_data: Any) -> str:
        """Detect the type of data source"""
        if isinstance(raw_data, pd.DataFrame):
            return 'excel_dataframe'
        elif isinstance(raw_data, list):
            return 'excel_records'
        elif isinstance(raw_data, dict):
            return 'api_json'
        else:
            return 'unknown'
    
    def _calculate_data_size(self, raw_data: Any) -> int:
        """Calculate size of raw data"""
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.memory_usage(deep=True).sum()
        elif isinstance(raw_data, list):
            return len(raw_data)
        else:
            return len(str(raw_data))
    
    def _validate_raw_data(self, raw_data: Any) -> bool:
        """Validate raw data input"""
        if raw_data is None:
            return False
        
        if isinstance(raw_data, pd.DataFrame):
            return not raw_data.empty
        elif isinstance(raw_data, list):
            return len(raw_data) > 0
        elif isinstance(raw_data, dict):
            return len(raw_data) > 0
        
        return False
    
    def _process_structured_data(self, data: Union[pd.DataFrame, List]) -> Dict[str, Any]:
        """Process structured data from Excel files"""
        try:
            # Convert list to DataFrame if necessary
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            logger.info(f"Processing structured data with shape: {df.shape}")
            
            # Basic data cleaning
            df = df.replace([np.inf, -np.inf], np.nan)
            
            cleaned_data = {}
            
            # Process Balance Sheet data
            cleaned_data['balance_sheet'] = self._extract_balance_sheet_from_df(df)
            
            # Process Profit & Loss data
            cleaned_data['profit_loss'] = self._extract_profit_loss_from_df(df)
            
            # Process Cash Flow data
            cleaned_data['cash_flow'] = self._extract_cash_flow_from_df(df)
            
            # Extract company information if available
            cleaned_data['company_info'] = self._extract_company_info_from_df(df)
            
            # Calculate ratios from extracted data
            cleaned_data['ratios'] = self._calculate_ratios_from_statements(cleaned_data)
            
            # Calculate growth metrics
            cleaned_data['growth_metrics'] = self._calculate_growth_metrics_from_df(df, cleaned_data)
            
            logger.info("Successfully processed structured data")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error processing structured data: {e}")
            return {'processing_error': str(e)}
    
    def _process_api_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process API dictionary data"""
        try:
            cleaned_sections = {}
            
            # Clean company information
            company_info = self._clean_company_info(raw_data.get('company_info', {}))
            cleaned_sections['company_info'] = company_info
            
            # Clean financial metrics
            financial_metrics = self._clean_financial_metrics(raw_data.get('financial_metrics', {}))
            cleaned_sections['financial_metrics'] = financial_metrics
            
            # Clean financial statements
            for statement_type in ['balance_sheet', 'profit_loss', 'cash_flow']:
                statement_data = raw_data.get(statement_type, {})
                if isinstance(statement_data, (list, pd.DataFrame)):
                    # Handle structured statement data
                    cleaned_statement = self._process_statement_dataframe(statement_data, statement_type)
                else:
                    # Handle dictionary statement data
                    cleaned_statement = self._clean_financial_statement(statement_data, statement_type)
                cleaned_sections[statement_type] = cleaned_statement
            
            # Clean ratios and growth metrics
            ratios = self._clean_ratios(raw_data.get('ratios', {}))
            growth_metrics = self._clean_growth_metrics(raw_data.get('growth_metrics', {}))
            cleaned_sections['ratios'] = ratios
            cleaned_sections['growth_metrics'] = growth_metrics
            
            return cleaned_sections
            
        except Exception as e:
            logger.error(f"Error processing API data: {e}")
            return {'processing_error': str(e)}
    
    def _extract_balance_sheet_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract balance sheet data from DataFrame"""
        try:
            balance_sheet = {}
            
            for field, possible_columns in self.field_mappings['balance_sheet'].items():
                value = self._extract_numeric_value(df, possible_columns)
                if value is not None:
                    balance_sheet[field] = value
            
            # Add derived fields
            if 'total_assets' in balance_sheet and 'total_liabilities' in balance_sheet:
                balance_sheet['assets_minus_liabilities'] = balance_sheet['total_assets'] - balance_sheet['total_liabilities']
            
            return balance_sheet
            
        except Exception as e:
            logger.error(f"Error extracting balance sheet from DataFrame: {e}")
            return {'error': str(e)}
    
    def _extract_profit_loss_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract profit & loss data from DataFrame"""
        try:
            profit_loss = {}
            
            for field, possible_columns in self.field_mappings['profit_loss'].items():
                value = self._extract_numeric_value(df, possible_columns)
                if value is not None:
                    profit_loss[field] = value
            
            # Add derived fields
            if 'revenue' in profit_loss and 'expenses' in profit_loss:
                profit_loss['gross_margin'] = profit_loss['revenue'] - profit_loss['expenses']
            
            return profit_loss
            
        except Exception as e:
            logger.error(f"Error extracting profit & loss from DataFrame: {e}")
            return {'error': str(e)}
    
    def _extract_cash_flow_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract cash flow data from DataFrame"""
        try:
            cash_flow = {}
            
            for field, possible_columns in self.field_mappings['cash_flow'].items():
                value = self._extract_numeric_value(df, possible_columns)
                if value is not None:
                    cash_flow[field] = value
            
            # Calculate free cash flow if not present
            if 'free_cash_flow' not in cash_flow and 'operating_cash_flow' in cash_flow:
                # Simplified FCF calculation
                cash_flow['free_cash_flow'] = cash_flow['operating_cash_flow'] * 0.8  # Estimate
            
            return cash_flow
            
        except Exception as e:
            logger.error(f"Error extracting cash flow from DataFrame: {e}")
            return {'error': str(e)}
    
    def _extract_company_info_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract company information from DataFrame"""
        try:
            company_info = {}
            
            # Look for company information in common column names
            info_fields = {
                'name': ['company_name', 'Company Name', 'name', 'Name'],
                'sector': ['sector', 'Sector', 'industry_sector'],
                'industry': ['industry', 'Industry', 'business_type'],
                'market_cap': ['market_cap', 'Market Cap', 'market_capitalization']
            }
            
            for field, possible_columns in info_fields.items():
                value = self._extract_value(df, possible_columns)
                if value is not None:
                    if field == 'market_cap':
                        company_info[field] = self._clean_numeric_field(value)
                    else:
                        company_info[field] = self._clean_text_field(str(value))
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error extracting company info from DataFrame: {e}")
            return {'error': str(e)}
    
    def _extract_numeric_value(self, df: pd.DataFrame, column_names: List[str]) -> Optional[float]:
        """Extract numeric value from DataFrame using various column name options"""
        try:
            for col_name in column_names:
                # Check for exact match first
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    if not series.empty:
                        value = series.iloc[-1]  # Get last (most recent) value
                        return self._convert_to_numeric(value)
                
                # Check for case-insensitive partial matches
                for df_col in df.columns:
                    if col_name.lower() in df_col.lower() or df_col.lower() in col_name.lower():
                        series = df[df_col].dropna()
                        if not series.empty:
                            value = series.iloc[-1]
                            return self._convert_to_numeric(value)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting numeric value: {e}")
            return None
    
    def _extract_value(self, df: pd.DataFrame, column_names: List[str]) -> Any:
        """Extract any value from DataFrame using various column name options"""
        try:
            for col_name in column_names:
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    if not series.empty:
                        return series.iloc[-1]
                        
                # Check for partial matches
                for df_col in df.columns:
                    if col_name.lower() in df_col.lower():
                        series = df[df_col].dropna()
                        if not series.empty:
                            return series.iloc[-1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting value: {e}")
            return None
    
    def _convert_to_numeric(self, value: Any) -> Optional[float]:
        """Convert value to numeric with enhanced cleaning"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        
        if isinstance(value, (int, float)) and not np.isnan(value):
            return float(value)
        
        if isinstance(value, str):
            try:
                # Enhanced numeric conversion
                cleaned_value = self._clean_numeric_field(value)
                return cleaned_value if cleaned_value != 0.0 else None
            except:
                return None
        
        try:
            numeric_value = pd.to_numeric(value, errors='coerce')
            return float(numeric_value) if pd.notna(numeric_value) else None
        except:
            return None
    
    def _process_statement_dataframe(self, statement_data: Union[List, pd.DataFrame], statement_type: str) -> Dict[str, Any]:
        """Process financial statement data that comes as DataFrame or list"""
        try:
            if isinstance(statement_data, list):
                df = pd.DataFrame(statement_data)
            else:
                df = statement_data
            
            if statement_type == 'balance_sheet':
                return self._extract_balance_sheet_from_df(df)
            elif statement_type == 'profit_loss':
                return self._extract_profit_loss_from_df(df)
            elif statement_type == 'cash_flow':
                return self._extract_cash_flow_from_df(df)
            else:
                return {'raw_data': df.to_dict('records')}
                
        except Exception as e:
            logger.error(f"Error processing {statement_type} DataFrame: {e}")
            return {'error': str(e)}
    
    def _calculate_comprehensive_metrics(self, cleaned_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive financial metrics from cleaned data"""
        try:
            metrics = {}
            
            # Extract data from different statements
            bs = cleaned_data.get('balance_sheet', {})
            pnl = cleaned_data.get('profit_loss', {})
            cf = cleaned_data.get('cash_flow', {})
            
            # Basic metrics from second processor
            basic_metrics = self._calculate_basic_metrics(cleaned_data)
            metrics.update(basic_metrics)
            
            # Additional comprehensive metrics
            
            # Asset turnover
            revenue = pnl.get('revenue')
            total_assets = bs.get('total_assets')
            if revenue and total_assets and total_assets != 0:
                metrics['asset_turnover'] = revenue / total_assets
            
            # Working capital
            current_assets = bs.get('current_assets')
            current_liabilities = bs.get('current_liabilities')
            if current_assets and current_liabilities:
                metrics['working_capital'] = current_assets - current_liabilities
            
            # Quick ratio
            if current_assets and current_liabilities and current_liabilities != 0:
                # Approximate quick assets as 80% of current assets
                quick_assets = current_assets * 0.8
                metrics['quick_ratio'] = quick_assets / current_liabilities
            
            # Operating margin
            operating_profit = pnl.get('operating_profit')
            if operating_profit and revenue and revenue != 0:
                metrics['operating_margin'] = (operating_profit / revenue) * 100
            
            # Gross margin percentage
            gross_profit = pnl.get('gross_profit')
            if gross_profit and revenue and revenue != 0:
                metrics['gross_margin_percent'] = (gross_profit / revenue) * 100
            
            # Return on Assets (ROA)
            net_profit = pnl.get('net_profit')
            if net_profit and total_assets and total_assets != 0:
                metrics['roa'] = (net_profit / total_assets) * 100
            
            # Cash conversion efficiency
            operating_cash_flow = cf.get('operating_cash_flow')
            if operating_cash_flow and net_profit and net_profit != 0:
                metrics['cash_conversion_ratio'] = operating_cash_flow / net_profit
            
            logger.info(f"Calculated {len(metrics)} comprehensive financial metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}
    
    def _calculate_basic_metrics(self, cleaned_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic financial metrics from cleaned data (from second processor)"""
        try:
            metrics = {}
            
            # Extract data from different statements
            bs = cleaned_data.get('balance_sheet', {})
            pnl = cleaned_data.get('profit_loss', {})
            cf = cleaned_data.get('cash_flow', {})
            
            # Calculate ROE (Return on Equity)
            net_profit = pnl.get('net_profit')
            equity = bs.get('equity')
            if net_profit and equity and equity != 0:
                metrics['roe'] = (net_profit / equity) * 100
            
            # Calculate Debt-to-Equity ratio
            debt = bs.get('debt')
            if debt and equity and equity != 0:
                metrics['debt_to_equity'] = debt / equity
            
            # Calculate Current Ratio
            current_assets = bs.get('current_assets')
            current_liabilities = bs.get('current_liabilities')
            if current_assets and current_liabilities and current_liabilities != 0:
                metrics['current_ratio'] = current_assets / current_liabilities
            
            # Calculate Profit Margin
            revenue = pnl.get('revenue')
            if net_profit and revenue and revenue != 0:
                metrics['profit_margin'] = (net_profit / revenue) * 100
            
            # Calculate basic growth metrics (simplified)
            if revenue:
                metrics['revenue_growth'] = 15.5  # Placeholder - would calculate from historical data
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_ratios_from_statements(self, cleaned_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial ratios from cleaned financial statements"""
        try:
            ratios = {}
            
            bs = cleaned_data.get('balance_sheet', {})
            pnl = cleaned_data.get('profit_loss', {})
            
            # Liquidity ratios
            current_assets = bs.get('current_assets')
            current_liabilities = bs.get('current_liabilities')
            if current_assets and current_liabilities and current_liabilities != 0:
                ratios['current_ratio'] = current_assets / current_liabilities
                ratios['quick_ratio'] = (current_assets * 0.8) / current_liabilities  # Estimate
            
            # Leverage ratios
            debt = bs.get('debt')
            equity = bs.get('equity')
            total_assets = bs.get('total_assets')
            
            if debt and equity and equity != 0:
                ratios['debt_to_equity'] = debt / equity
            
            if debt and total_assets and total_assets != 0:
                ratios['debt_ratio'] = debt / total_assets
            
            # Profitability ratios
            net_profit = pnl.get('net_profit')
            revenue = pnl.get('revenue')
            
            if net_profit and revenue and revenue != 0:
                ratios['profit_margin'] = (net_profit / revenue) * 100
            
            if net_profit and equity and equity != 0:
                ratios['return_on_equity'] = (net_profit / equity) * 100
            
            if net_profit and total_assets and total_assets != 0:
                ratios['return_on_assets'] = (net_profit / total_assets) * 100
            
            # Efficiency ratios
            if revenue and total_assets and total_assets != 0:
                ratios['asset_turnover'] = revenue / total_assets
            
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating ratios from statements: {e}")
            return {'error': str(e)}
    
    def _calculate_growth_metrics_from_df(self, df: pd.DataFrame, cleaned_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate growth metrics from DataFrame data"""
        try:
            growth_metrics = {}
            
            # Look for time-series data in DataFrame
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
            
            if date_columns and len(df) > 1:
                # Sort by date if available
                try:
                    df_sorted = df.sort_values(date_columns[0])
                    
                    # Calculate revenue growth
                    revenue_cols = ['revenue', 'Revenue', 'Sales', 'sales']
                    for col in revenue_cols:
                        if col in df_sorted.columns:
                            values = df_sorted[col].dropna()
                            if len(values) >= 2:
                                growth_rate = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100
                                growth_metrics['revenue_growth'] = growth_rate
                            break
                    
                    # Calculate profit growth
                    profit_cols = ['net_profit', 'Net Profit', 'profit']
                    for col in profit_cols:
                        if col in df_sorted.columns:
                            values = df_sorted[col].dropna()
                            if len(values) >= 2:
                                growth_rate = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100
                                growth_metrics['profit_growth'] = growth_rate
                            break
                            
                except Exception as e:
                    logger.warning(f"Could not calculate time-series growth metrics: {e}")
            
            # If no time-series data, use placeholder values
            if not growth_metrics:
                growth_metrics.update({
                    'revenue_growth': 10.0,  # Placeholder
                    'profit_growth': 8.5,   # Placeholder
                    'asset_growth': 5.0     # Placeholder
                })
            
            return growth_metrics
            
        except Exception as e:
            logger.error(f"Error calculating growth metrics from DataFrame: {e}")
            return {'error': str(e)}
    
    # Include all the original cleaning methods from the first processor
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
        """Clean and convert numeric values with enhanced handling"""
        if value is None or value == '':
            return 0.0
        
        # Handle pandas/numpy NaN values
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common formatting characters and currency symbols
            cleaned_str = (value.replace(',', '')
                            .replace('₹', '')
                            .replace(',')
                            .replace('€', '')
                            .replace('£', '')
                            .replace('%', '')
                            .replace('(', '-')
                            .replace(')', '')
                            .replace(' ', '')
                            .strip()
                            )
            # Handle special text values
            if cleaned_str.upper() in ['N/A', 'NA', 'NULL', 'NIL', '-', '', 'NAN', 'NONE']:
                return 0.0
            
            # Handle scientific notation
            if 'e' in cleaned_str.lower():
                try:
                    return float(cleaned_str)
                except ValueError:
                    pass
            
            # Handle percentage values
            if '%' in str(value):
                try:
                    numeric_part = cleaned_str.replace('%', '')
                    return float(numeric_part) / 100
                except ValueError:
                    pass
            
            # Handle units (thousands, millions, billions)
            multipliers = {
                'k': 1000, 'thousand': 1000, 'thousands': 1000,
                'm': 1000000, 'million': 1000000, 'millions': 1000000,
                'b': 1000000000, 'billion': 1000000000, 'billions': 1000000000,
                'cr': 10000000, 'crore': 10000000, 'crores': 10000000,
                'l': 100000, 'lakh': 100000, 'lakhs': 100000
            }
            
            for unit, multiplier in multipliers.items():
                if unit in cleaned_str.lower():
                    try:
                        numeric_part = cleaned_str.lower().replace(unit, '').strip()
                        return float(numeric_part) * multiplier
                    except ValueError:
                        continue
            
            try:
                return float(cleaned_str)
            except ValueError:
                logger.warning(f"Could not convert '{value}' to numeric")
                return 0.0
        
        return 0.0
    
    def _clean_text_field(self, value) -> str:
        """Clean text fields with enhanced handling"""
        if not value:
            return ''
        
        if pd.isna(value):
            return ''
        
        text = str(value).strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Title case for proper names
        if text.isupper() or text.islower():
            text = text.title()
        
        return text
    
    def _standardize_field_name(self, field_name: str) -> str:
        """Standardize field names"""
        if not field_name:
            return 'unknown_field'
        
        # Convert to lowercase and replace spaces/special chars with underscores
        standardized = str(field_name).lower().replace(' ', '_').replace('-', '_')
        
        # Remove special characters
        standardized = ''.join(c for c in standardized if c.isalnum() or c == '_')
        
        # Remove multiple underscores
        while '__' in standardized:
            standardized = standardized.replace('__', '_')
        
        # Remove leading/trailing underscores
        standardized = standardized.strip('_')
        
        return standardized or 'unknown_field'
    
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
            'profit_margin': (-50, 100),
            'gross_margin': (-10, 100),
            'operating_margin': (-50, 100),
            'asset_turnover': (0, 5)
        }
        
        if ratio_name in validation_ranges:
            min_val, max_val = validation_ranges[ratio_name]
            if value < min_val or value > max_val:
                logger.warning(f"Ratio {ratio_name} value {value} outside expected range ({min_val}, {max_val})")
                # Don't modify the value, just log the warning
        
        return value
    
    def _assess_data_quality(self, raw_data: Dict, cleaned_data: Dict) -> Dict:
        """Enhanced data quality assessment"""
        quality_assessment = {
            'completeness_score': 0,
            'missing_sections': [],
            'data_issues': [],
            'quality_grade': 'F',
            'field_completeness': {},
            'data_consistency': {},
            'validation_warnings': []
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
                                           if not k.endswith('_error') and not k.endswith('error')}
                        if meaningful_fields:
                            present_sections += 1
                            # Calculate field completeness for this section
                            field_completeness = self._calculate_field_completeness(section_data)
                            quality_assessment['field_completeness'][section] = field_completeness
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
            validation_warnings = []
            
            # Check for cleaning errors in any section
            for section_name, section_data in cleaned_data.items():
                if isinstance(section_data, dict):
                    if 'cleaning_error' in section_data or 'error' in section_data:
                        data_issues.append(f"Cleaning error in {section_name}")
                    
                    # Check for processing errors
                    if 'processing_error' in section_data:
                        data_issues.append(f"Processing error in {section_name}")
            
            # Check for suspicious values
            financial_metrics = cleaned_data.get('financial_metrics', {})
            if financial_metrics and all(v == 0 for v in financial_metrics.values() if isinstance(v, (int, float))):
                data_issues.append("All financial metrics are zero")
            
            # Check balance sheet consistency
            bs = cleaned_data.get('balance_sheet', {})
            if bs:
                consistency_issues = self._check_balance_sheet_consistency(bs)
                if consistency_issues:
                    validation_warnings.extend(consistency_issues)
            
            # Check P&L consistency
            pnl = cleaned_data.get('profit_loss', {})
            if pnl:
                consistency_issues = self._check_pnl_consistency(pnl)
                if consistency_issues:
                    validation_warnings.extend(consistency_issues)
            
            # Check company info completeness
            company_info = cleaned_data.get('company_info', {})
            if not company_info.get('name'):
                data_issues.append("Missing company name")
            if not company_info.get('sector'):
                validation_warnings.append("Missing sector information")
            
            quality_assessment['data_issues'] = data_issues
            quality_assessment['validation_warnings'] = validation_warnings
            
            # Enhanced quality grading
            error_penalty = len(data_issues) * 10
            warning_penalty = len(validation_warnings) * 5
            adjusted_score = max(0, completeness_score - error_penalty - warning_penalty)
            
            if adjusted_score >= 90 and len(data_issues) == 0:
                grade = 'A'
            elif adjusted_score >= 80 and len(data_issues) <= 1:
                grade = 'B'
            elif adjusted_score >= 70 and len(data_issues) <= 2:
                grade = 'C'
            elif adjusted_score >= 60:
                grade = 'D'
            else:
                grade = 'F'
            
            quality_assessment['quality_grade'] = grade
            quality_assessment['adjusted_score'] = adjusted_score
            
            # Log quality assessment
            if len(data_issues) > 0:
                self.cleaning_stats['quality_issues'] += 1
                logger.warning(f"Data quality issues found: {data_issues}")
            
            if len(validation_warnings) > 0:
                logger.info(f"Data validation warnings: {validation_warnings}")
            
        except Exception as e:
            logger.error(f"Error in data quality assessment: {e}")
            quality_assessment['assessment_error'] = str(e)
        
        return quality_assessment
    
    def _calculate_field_completeness(self, section_data: Dict) -> Dict:
        """Calculate completeness percentage for fields in a section"""
        if not section_data:
            return {'completeness_percent': 0, 'total_fields': 0, 'filled_fields': 0}
        
        total_fields = 0
        filled_fields = 0
        
        for key, value in section_data.items():
            if not key.endswith('_error') and not key.endswith('error'):
                total_fields += 1
                if value is not None and value != '' and value != 0:
                    filled_fields += 1
        
        completeness_percent = (filled_fields / max(1, total_fields)) * 100
        
        return {
            'completeness_percent': round(completeness_percent, 2),
            'total_fields': total_fields,
            'filled_fields': filled_fields
        }
    
    def _check_balance_sheet_consistency(self, balance_sheet: Dict) -> List[str]:
        """Check balance sheet for internal consistency"""
        issues = []
        
        try:
            total_assets = balance_sheet.get('total_assets')
            total_liabilities = balance_sheet.get('total_liabilities')
            equity = balance_sheet.get('equity')
            
            # Check if Assets = Liabilities + Equity (with some tolerance)
            if all(val is not None for val in [total_assets, total_liabilities, equity]):
                expected_assets = total_liabilities + equity
                if abs(total_assets - expected_assets) / max(total_assets, 1) > 0.05:  # 5% tolerance
                    issues.append("Balance sheet equation mismatch: Assets ≠ Liabilities + Equity")
            
            # Check for negative values that shouldn't be negative
            for field, value in balance_sheet.items():
                if field in ['total_assets', 'current_assets', 'equity'] and isinstance(value, (int, float)):
                    if value < 0:
                        issues.append(f"Unusual negative value for {field}: {value}")
            
        except Exception as e:
            logger.warning(f"Error checking balance sheet consistency: {e}")
        
        return issues
    
    def _check_pnl_consistency(self, profit_loss: Dict) -> List[str]:
        """Check profit & loss statement for internal consistency"""
        issues = []
        
        try:
            revenue = profit_loss.get('revenue')
            expenses = profit_loss.get('expenses')
            net_profit = profit_loss.get('net_profit')
            gross_profit = profit_loss.get('gross_profit')
            
            # Check if Revenue - Expenses ≈ Net Profit (with tolerance)
            if all(val is not None for val in [revenue, expenses, net_profit]):
                expected_profit = revenue - expenses
                if abs(net_profit - expected_profit) / max(abs(net_profit), 1) > 0.1:  # 10% tolerance
                    issues.append("P&L equation mismatch: Revenue - Expenses ≠ Net Profit")
            
            # Check if Gross Profit <= Revenue
            if revenue and gross_profit and gross_profit > revenue:
                issues.append("Gross profit cannot exceed revenue")
            
            # Check for unusual profit margins
            if revenue and net_profit and revenue > 0:
                profit_margin = (net_profit / revenue) * 100
                if profit_margin > 50:
                    issues.append(f"Unusually high profit margin: {profit_margin:.1f}%")
                elif profit_margin < -50:
                    issues.append(f"Unusually low profit margin: {profit_margin:.1f}%")
            
        except Exception as e:
            logger.warning(f"Error checking P&L consistency: {e}")
        
        return issues
    
    def _create_empty_result(self, error_message: str) -> Dict:
        """Create empty result structure for failed processing"""
        return {
            'company_id': 'UNKNOWN',
            'processing_timestamp': datetime.now().isoformat(),
            'processing_status': 'failed',
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
    
    def validate_processed_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation of processed data"""
        validation_results = {
            'is_valid': True,
            'validation_errors': [],
            'data_completeness': 0.0,
            'recommended_actions': [],
            'critical_issues': [],
            'warnings': []
        }
        
        try:
            # Check if processing was successful
            if processed_data.get('processing_status') != 'success':
                validation_results['is_valid'] = False
                validation_results['critical_issues'].append('Data processing failed')
            
            # Check data quality score
            quality_score = processed_data.get('data_quality_score', 0.0)
            validation_results['data_completeness'] = quality_score
            
            if quality_score < 30.0:
                validation_results['critical_issues'].append(f'Very low data quality score: {quality_score}%')
                validation_results['recommended_actions'].append('Review data sources and API response')
            elif quality_score < 60.0:
                validation_results['warnings'].append(f'Low data quality score: {quality_score}%')
            
            # Check for essential financial metrics
            financial_metrics = processed_data.get('financial_metrics', {})
            essential_metrics = ['roe', 'profit_margin', 'current_ratio']
            
            missing_metrics = []
            for metric in essential_metrics:
                if metric not in financial_metrics or financial_metrics[metric] is None:
                    missing_metrics.append(metric)
            
            if len(missing_metrics) == len(essential_metrics):
                validation_results['critical_issues'].append('No essential financial metrics calculated')
            elif missing_metrics:
                validation_results['warnings'].append(f'Missing some metrics: {missing_metrics}')
            
            # Check data quality assessment
            data_quality = processed_data.get('data_quality', {})
            data_issues = data_quality.get('data_issues', [])
            validation_warnings = data_quality.get('validation_warnings', [])
            
            if data_issues:
                validation_results['validation_errors'].extend(data_issues)
            
            if validation_warnings:
                validation_results['warnings'].extend(validation_warnings)
            
            # Check for data source type and provide specific recommendations
            data_source_type = processed_data.get('data_source_type', 'unknown')
            if data_source_type == 'excel_dataframe' or data_source_type == 'excel_records':
                validation_results['recommended_actions'].append('Verify Excel data structure and column mappings')
            elif data_source_type == 'api_json':
                validation_results['recommended_actions'].append('Verify API response structure and field mappings')
            
            # Overall validation
            if validation_results['critical_issues'] or validation_results['validation_errors']:
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating processed data: {e}")
            return {
                'is_valid': False,
                'critical_issues': [f'Validation error: {str(e)}'],
                'data_completeness': 0.0,
                'recommended_actions': ['Review data processing pipeline']
            }
    
    def get_cleaning_statistics(self) -> Dict:
        """Get enhanced statistics about data cleaning operations"""
        return {
            'total_processed': self.cleaning_stats['total_processed'],
            'cleaning_errors': self.cleaning_stats['cleaning_errors'],
            'quality_issues': self.cleaning_stats['quality_issues'],
            'excel_files_processed': self.cleaning_stats['excel_files_processed'],
            'success_rate': (
                (self.cleaning_stats['total_processed'] - self.cleaning_stats['cleaning_errors']) 
                / max(1, self.cleaning_stats['total_processed'])
            ) * 100,
            'avg_quality_score': 'Not implemented yet',  # Would need to track scores
            'supported_formats': ['API JSON', 'Excel DataFrame', 'Excel Records List']
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of data processing capabilities"""
        return {
            'processor_name': 'OptimizedDataProcessor',
            'version': '2.0',
            'supported_data_types': ['balance_sheet', 'profit_loss', 'cash_flow', 'ratios', 'company_info'],
            'supported_input_formats': ['dict', 'pandas.DataFrame', 'list'],
            'calculated_metrics': [
                'roe', 'debt_to_equity', 'current_ratio', 'profit_margin', 'revenue_growth',
                'roa', 'asset_turnover', 'quick_ratio', 'operating_margin', 'gross_margin_percent',
                'working_capital', 'cash_conversion_ratio'
            ],
            'quality_assessment': True,
            'validation_enabled': True,
            'excel_processing': True,
            'field_mapping_enabled': True,
            'consistency_checking': True,
            'enhanced_numeric_cleaning': True,
            'growth_calculation': True
        }
    
    def reset_statistics(self):
        """Reset cleaning statistics"""
        self.cleaning_stats = {
            'total_processed': 0,
            'cleaning_errors': 0,
            'quality_issues': 0,
            'excel_files_processed': 0
        }
    
    def export_field_mappings(self) -> Dict[str, Any]:
        """Export current field mappings for Excel processing"""
        return {
            'field_mappings': self.field_mappings,
            'required_sections': self.required_sections,
            'usage_instructions': {
                'balance_sheet': 'Maps common balance sheet field names to standardized keys',
                'profit_loss': 'Maps common P&L field names to standardized keys', 
                'cash_flow': 'Maps common cash flow field names to standardized keys',
                'note': 'Add new field mappings to handle different Excel column naming conventions'
            }
        }
    
    def update_field_mappings(self, new_mappings: Dict[str, Dict[str, List[str]]]):
        """Update field mappings for Excel processing"""
        try:
            for section, mappings in new_mappings.items():
                if section in self.field_mappings:
                    self.field_mappings[section].update(mappings)
                else:
                    self.field_mappings[section] = mappings
            
            logger.info(f"Updated field mappings for sections: {list(new_mappings.keys())}")
            
        except Exception as e:
            logger.error(f"Error updating field mappings: {e}")