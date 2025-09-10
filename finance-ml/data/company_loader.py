"""
data/company_loader.py - FIXED VERSION
Company Data Loading from Excel Files

This module handles:
- Loading company IDs from Excel files in data/ directory
- Auto-detection of Excel files and column formats
- Fallback to sample companies for testing
"""

import os
import logging
import pandas as pd
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class CompanyDataLoader:
    """
    Handles loading company IDs from Excel files
    Supports various Excel file formats and column names
    """
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize company data loader
        
        Args:
            data_directory: Directory containing Excel files
        """
        self.data_directory = data_directory
        # FIXED: Changed from 'supported_filename' to 'supported_filenames' (plural)
        self.supported_filenames = [
            "Nifty100Companies.xlsx",
            "company_id.xlsx",
            "companies.xlsx",
            "nifty100.xlsx",
            "company_list.xlsx"
        ]
        
        self.supported_column_names = [
            'company_id', 'Company ID', 'ID', 'Symbol', 'SYMBOL',
            'company_symbol', 'ticker', 'Ticker', 'Code', 'code',
            'CompanyID', 'Stock_Symbol', 'stock_symbol'
        ]
        
        logger.info(f"CompanyDataLoader initialized with data directory: {data_directory}")
    
    def load_companies_from_excel(self, filename: Optional[str] = None) -> List[str]:
        """
        Load company IDs from Excel file
        
        Args:
            filename: Specific Excel filename to load, or None for auto-detection
            
        Returns:
            List of company ID strings
        """
        logger.info("Starting company data loading from Excel...")
        
        try:
            # Find Excel file
            excel_file_path = self._find_excel_file(filename)
            
            if not excel_file_path:
                logger.warning("No Excel file found in data directory")
                logger.info("Available files in data directory:")
                self._list_data_directory_contents()
                logger.info("Creating sample Excel file for testing...")
                sample_file = self.save_sample_excel_file()
                if sample_file:
                    logger.info(f"Using created sample file: {sample_file}")
                    excel_file_path = sample_file
                else:
                    logger.info("Using hardcoded sample companies")
                    return self._get_sample_companies()
            
            # Load Excel file
            logger.info(f"Loading Excel file: {excel_file_path}")
            company_ids = self._load_excel_file(excel_file_path)
            
            if not company_ids:
                logger.warning("No valid company IDs found in Excel file")
                logger.info("Using sample companies as fallback")
                return self._get_sample_companies()
            
            logger.info(f"Successfully loaded {len(company_ids)} company IDs from Excel")
            logger.info(f"Sample companies: {company_ids[:5]}{'...' if len(company_ids) > 5 else ''}")
            
            return company_ids
            
        except Exception as e:
            logger.error(f"Error loading companies from Excel: {e}")
            logger.info("Using sample companies as fallback")
            return self._get_sample_companies()
    
    def _find_excel_file(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Find Excel file in data directory
        
        Args:
            filename: Specific filename to look for
            
        Returns:
            Full path to Excel file or None if not found
        """
        # Check if data directory exists
        if not os.path.exists(self.data_directory):
            logger.warning(f"Data directory '{self.data_directory}' does not exist")
            logger.info("Creating data directory...")
            try:
                os.makedirs(self.data_directory, exist_ok=True)
                logger.info(f"Created data directory: {self.data_directory}")
            except Exception as e:
                logger.error(f"Failed to create data directory: {e}")
            return None
        
        # If specific filename provided, check for it
        if filename:
            filepath = os.path.join(self.data_directory, filename)
            if os.path.exists(filepath) and filename.endswith(('.xlsx', '.xls')):
                logger.info(f"Found specified Excel file: {filepath}")
                return filepath
            else:
                logger.warning(f"Specified file not found: {filepath}")
        
        # Auto-detect supported filenames
        for supported_filename in self.supported_filenames:
            filepath = os.path.join(self.data_directory, supported_filename)
            if os.path.exists(filepath):
                logger.info(f"Auto-detected Excel file: {filepath}")
                return filepath
        
        # Look for any .xlsx or .xls files
        try:
            for file in os.listdir(self.data_directory):
                if file.endswith(('.xlsx', '.xls')):
                    filepath = os.path.join(self.data_directory, file)
                    logger.info(f"Found Excel file: {filepath}")
                    return filepath
        except Exception as e:
            logger.error(f"Error scanning data directory: {e}")
        
        return None
    
    def _load_excel_file(self, filepath: str) -> List[str]:
        """
        Load company IDs from Excel file
        
        Args:
            filepath: Full path to Excel file
            
        Returns:
            List of company ID strings
        """
        try:
            # Read Excel file - try different sheet options
            logger.info(f"Reading Excel file: {filepath}")
            
            # First try to read the first sheet
            df = pd.read_excel(filepath, sheet_name=0)
            
            logger.info(f"Excel file loaded successfully")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Display first few rows for debugging
            logger.info("First 5 rows of data:")
            logger.info(f"\n{df.head().to_string()}")
            
            # Extract company IDs
            company_ids = self._extract_company_ids_from_dataframe(df)
            
            return company_ids
            
        except Exception as e:
            logger.error(f"Error reading Excel file {filepath}: {e}")
            # Try reading with different parameters
            try:
                logger.info("Trying to read Excel with different parameters...")
                df = pd.read_excel(filepath, header=0)
                company_ids = self._extract_company_ids_from_dataframe(df)
                return company_ids
            except Exception as e2:
                logger.error(f"Failed with alternative parameters: {e2}")
                return []
    
    def _extract_company_ids_from_dataframe(self, df: pd.DataFrame) -> List[str]:
        """
        Extract company IDs from DataFrame
        
        Args:
            df: Pandas DataFrame loaded from Excel
            
        Returns:
            List of company ID strings
        """
        company_ids = []
        
        try:
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Try each supported column name
            for col_name in self.supported_column_names:
                if col_name in df.columns:
                    logger.info(f"Found company ID column: '{col_name}'")
                    
                    # Extract and clean company IDs
                    ids_series = df[col_name].dropna()  # Remove NaN values
                    logger.info(f"Non-null values in column '{col_name}': {len(ids_series)}")
                    
                    # Convert to string and clean
                    company_ids = []
                    for idx, id_val in enumerate(ids_series):
                        if pd.notna(id_val):  # Additional NaN check
                            # Convert to string and clean
                            clean_id = str(id_val).strip().upper()
                            
                            # Skip empty or invalid entries
                            if clean_id and clean_id not in ['NAN', 'NULL', '', 'NONE']:
                                company_ids.append(clean_id)
                                if idx < 10:  # Log first 10 for debugging
                                    logger.debug(f"  Row {idx}: '{id_val}' -> '{clean_id}'")
                    
                    if company_ids:
                        logger.info(f"Extracted {len(company_ids)} company IDs from column '{col_name}'")
                        logger.info(f"First 10 company IDs: {company_ids[:10]}")
                        return company_ids
            
            # If no recognized column found, try the first column
            if not company_ids and len(df.columns) > 0:
                first_col = df.columns[0]
                logger.info(f"No recognized column found, trying first column: '{first_col}'")
                
                ids_series = df[first_col].dropna()
                logger.info(f"Non-null values in first column: {len(ids_series)}")
                
                for idx, id_val in enumerate(ids_series):
                    if pd.notna(id_val):
                        clean_id = str(id_val).strip().upper()
                        if clean_id and clean_id not in ['NAN', 'NULL', '', 'NONE']:
                            company_ids.append(clean_id)
                            if idx < 10:  # Log first 10 for debugging
                                logger.debug(f"  Row {idx}: '{id_val}' -> '{clean_id}'")
                
                if company_ids:
                    logger.info(f"Extracted {len(company_ids)} company IDs from first column")
                    logger.info(f"First 10 company IDs: {company_ids[:10]}")
                    return company_ids
            
            # If still no company IDs found, show available data for debugging
            if not company_ids:
                logger.warning("No valid company IDs found in any column")
                logger.info("Available columns and sample data:")
                for col in df.columns[:5]:  # Show first 5 columns
                    sample_data = df[col].dropna().head(5).tolist()
                    logger.info(f"  Column '{col}' sample: {sample_data}")
            
        except Exception as e:
            logger.error(f"Error extracting company IDs from DataFrame: {e}")
        
        return company_ids
    
    def _list_data_directory_contents(self):
        """List contents of data directory for debugging"""
        try:
            if os.path.exists(self.data_directory):
                files = os.listdir(self.data_directory)
                if files:
                    logger.info(f"Files in '{self.data_directory}' directory:")
                    for file in files:
                        file_path = os.path.join(self.data_directory, file)
                        file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                        logger.info(f"  {file} ({file_size} bytes)")
                else:
                    logger.info(f"  Data directory '{self.data_directory}' is empty")
            else:
                logger.info(f"  Data directory '{self.data_directory}' does not exist")
        except Exception as e:
            logger.error(f"Error listing data directory: {e}")
    
    def _get_sample_companies(self) -> List[str]:
        """
        Get sample company IDs for testing when Excel file is not available
        
        Returns:
            List of sample company ID strings
        """
        # Extended sample with more companies for better testing
        sample_companies = [
            "TCS", "HDFCBANK", "DMART", "INFY", "RELIANCE", 
            "WIPRO", "BAJFINANCE", "AXISBANK", "ICICIBANK", "SBIN",
            "LT", "ASIANPAINT", "NESTLEIND", "KOTAKBANK", "BHARTIARTL",
            "MARUTI", "TITAN", "HINDUNILVR", "ITC", "POWERGRID"
        ]
        
        logger.info(f"Using sample companies: {sample_companies}")
        return sample_companies
    
    def save_sample_excel_file(self) -> str:
        """
        Create a sample Excel file with Nifty 100 companies for testing purposes
        
        Returns:
            Path to created sample file
        """
        try:
            # Ensure data directory exists
            os.makedirs(self.data_directory, exist_ok=True)
            
            # Create sample data with more companies (representative of Nifty 100)
            sample_data = {
                'Symbol': [  # Using 'Symbol' as it's a common column name
                    'TCS', 'HDFCBANK', 'DMART', 'INFY', 'RELIANCE',
                    'WIPRO', 'BAJFINANCE', 'AXISBANK', 'ICICIBANK', 'SBIN',
                    'LT', 'ASIANPAINT', 'NESTLEIND', 'KOTAKBANK', 'BHARTIARTL',
                    'MARUTI', 'TITAN', 'HINDUNILVR', 'ITC', 'POWERGRID',
                    'SUNPHARMA', 'ULTRACEMCO', 'ONGC', 'NTPC', 'ADANIPORTS',
                    'JSWSTEEL', 'TATASTEEL', 'COALINDIA', 'INDUSINDBK', 'GRASIM',
                    'TECHM', 'HCLTECH', 'DRREDDY', 'CIPLA', 'DIVISLAB',
                    'BPCL', 'IOC', 'HINDALCO', 'BAJAJ-AUTO', 'HEROMOTOCO',
                    'EICHERMOT', 'BAJAJFINSV', 'BRITANNIA', 'DABUR', 'GODREJCP',
                    'MARICO', 'UPL', 'PIDILITIND', 'BERGEPAINT', 'SBILIFE'
                ],
                'Company_Name': [
                    'Tata Consultancy Services', 'HDFC Bank', 'Avenue Supermarts', 'Infosys', 'Reliance Industries',
                    'Wipro', 'Bajaj Finance', 'Axis Bank', 'ICICI Bank', 'State Bank of India',
                    'Larsen & Toubro', 'Asian Paints', 'Nestle India', 'Kotak Mahindra Bank', 'Bharti Airtel',
                    'Maruti Suzuki', 'Titan Company', 'Hindustan Unilever', 'ITC', 'Power Grid Corporation',
                    'Sun Pharmaceutical', 'UltraTech Cement', 'Oil & Natural Gas Corporation', 'NTPC', 'Adani Ports',
                    'JSW Steel', 'Tata Steel', 'Coal India', 'IndusInd Bank', 'Grasim Industries',
                    'Tech Mahindra', 'HCL Technologies', 'Dr. Reddys Laboratories', 'Cipla', 'Divi\'s Laboratories',
                    'Bharat Petroleum Corporation', 'Indian Oil Corporation', 'Hindalco Industries', 'Bajaj Auto', 'Hero MotoCorp',
                    'Eicher Motors', 'Bajaj Finserv', 'Britannia Industries', 'Dabur India', 'Godrej Consumer Products',
                    'Marico', 'UPL', 'Pidilite Industries', 'Berger Paints', 'SBI Life Insurance'
                ],
                'Sector': [
                    'IT', 'Banking', 'Retail', 'IT', 'Oil & Gas',
                    'IT', 'Financial Services', 'Banking', 'Banking', 'Banking',
                    'Construction', 'Paints', 'FMCG', 'Banking', 'Telecom',
                    'Automobile', 'Consumer Goods', 'FMCG', 'FMCG', 'Utilities',
                    'Pharmaceuticals', 'Cement', 'Oil & Gas', 'Utilities', 'Ports',
                    'Metals', 'Metals', 'Mining', 'Banking', 'Textiles',
                    'IT', 'IT', 'Pharmaceuticals', 'Pharmaceuticals', 'Pharmaceuticals',
                    'Oil & Gas', 'Oil & Gas', 'Metals', 'Automobile', 'Automobile',
                    'Automobile', 'Financial Services', 'FMCG', 'FMCG', 'FMCG',
                    'FMCG', 'Chemicals', 'Chemicals', 'Paints', 'Insurance'
                ]
            }
            
            # Create DataFrame and save to Excel
            df = pd.DataFrame(sample_data)
            sample_file_path = os.path.join(self.data_directory, 'Nifty100Companies.xlsx')
            df.to_excel(sample_file_path, index=False)
            
            logger.info(f"Created sample Excel file: {sample_file_path}")
            logger.info(f"File contains {len(sample_data['Symbol'])} companies")
            
            return sample_file_path
            
        except Exception as e:
            logger.error(f"Error creating sample Excel file: {e}")
            return ""
    
    def validate_company_ids(self, company_ids: List[str]) -> Dict[str, Any]:
        """
        Validate loaded company IDs
        
        Args:
            company_ids: List of company IDs to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'total_ids': len(company_ids),
            'valid_ids': [],
            'invalid_ids': [],
            'duplicate_ids': [],
            'validation_summary': {}
        }
        
        try:
            seen_ids = set()
            valid_ids = []
            invalid_ids = []
            duplicate_ids = []
            
            for company_id in company_ids:
                # Check for duplicates
                if company_id in seen_ids:
                    duplicate_ids.append(company_id)
                    continue
                
                seen_ids.add(company_id)
                
                # Basic validation rules
                if self._is_valid_company_id(company_id):
                    valid_ids.append(company_id)
                else:
                    invalid_ids.append(company_id)
            
            validation_results.update({
                'valid_ids': valid_ids,
                'invalid_ids': invalid_ids,
                'duplicate_ids': duplicate_ids,
                'validation_summary': {
                    'valid_count': len(valid_ids),
                    'invalid_count': len(invalid_ids),
                    'duplicate_count': len(duplicate_ids),
                    'validation_rate': (len(valid_ids) / max(1, len(company_ids))) * 100
                }
            })
            
            # Log validation results
            logger.info(f"Company ID validation completed:")
            logger.info(f"  Valid IDs: {len(valid_ids)}")
            logger.info(f"  Invalid IDs: {len(invalid_ids)}")
            logger.info(f"  Duplicate IDs: {len(duplicate_ids)}")
            logger.info(f"  Validation rate: {validation_results['validation_summary']['validation_rate']:.1f}%")
            
            if invalid_ids:
                logger.warning(f"Invalid company IDs found: {invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}")
            
        except Exception as e:
            logger.error(f"Error validating company IDs: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _is_valid_company_id(self, company_id: str) -> bool:
        """
        Check if company ID is valid
        
        Args:
            company_id: Company ID string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation rules
            if not company_id or not isinstance(company_id, str):
                return False
            
            # Check length (typically 1-15 characters)
            if len(company_id) < 1 or len(company_id) > 15:
                return False
            
            # Check for valid characters (alphanumeric, some special chars allowed)
            valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.')
            if not all(c in valid_chars for c in company_id.upper()):
                return False
            
            # Check for common invalid patterns
            invalid_patterns = ['NULL', 'NONE', 'EMPTY', 'N/A', 'NA']
            if company_id.upper() in invalid_patterns:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_loading_statistics(self) -> Dict[str, Any]:
        """Get statistics about company loading operations"""
        return {
            'supported_filenames': self.supported_filenames,
            'supported_columns': self.supported_column_names,
            'data_directory': self.data_directory
        }