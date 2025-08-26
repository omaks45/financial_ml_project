"""
data/company_loader.py
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
        self.supported_filename = [
            "company_id.xlsx",
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
                logger.info("Using sample companies for testing")
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
            # Read Excel file
            logger.info(f"Reading Excel file: {filepath}")
            df = pd.read_excel(filepath)
            
            logger.info(f"Excel file loaded successfully")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Display first few rows for debugging
            logger.info("First 3 rows of data:")
            logger.info(f"{df.head(3).to_string()}")
            
            # Extract company IDs
            company_ids = self._extract_company_ids_from_dataframe(df)
            
            return company_ids
            
        except Exception as e:
            logger.error(f"Error reading Excel file {filepath}: {e}")
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
            # Try each supported column name
            for col_name in self.supported_column_names:
                if col_name in df.columns:
                    logger.info(f"Found company ID column: '{col_name}'")
                    
                    # Extract and clean company IDs
                    ids_series = df[col_name].dropna()  # Remove NaN values
                    
                    # Convert to string and clean
                    company_ids = []
                    for id_val in ids_series:
                        if pd.notna(id_val):  # Additional NaN check
                            # Convert to string and clean
                            clean_id = str(id_val).strip().upper()
                            
                            # Skip empty or invalid entries
                            if clean_id and clean_id not in ['NAN', 'NULL', '', 'NONE']:
                                company_ids.append(clean_id)
                    
                    if company_ids:
                        logger.info(f"Extracted {len(company_ids)} company IDs from column '{col_name}'")
                        return company_ids
            
            # If no recognized column found, try the first column
            if not company_ids and len(df.columns) > 0:
                first_col = df.columns[0]
                logger.info(f"No recognized column found, trying first column: '{first_col}'")
                
                ids_series = df[first_col].dropna()
                for id_val in ids_series:
                    if pd.notna(id_val):
                        clean_id = str(id_val).strip().upper()
                        if clean_id and clean_id not in ['NAN', 'NULL', '', 'NONE']:
                            company_ids.append(clean_id)
                
                if company_ids:
                    logger.info(f"Extracted {len(company_ids)} company IDs from first column")
                    return company_ids
            
            # If still no company IDs found, show available data for debugging
            if not company_ids:
                logger.warning("No valid company IDs found in any column")
                logger.info("Available columns and sample data:")
                for col in df.columns[:5]:  # Show first 5 columns
                    sample_data = df[col].dropna().head(3).tolist()
                    logger.info(f"  {col}: {sample_data}")
            
        except Exception as e:
            logger.error(f"Error extracting company IDs from DataFrame: {e}")
        
        return company_ids
    
    def _list_data_directory_contents(self):
        """List contents of data directory for debugging"""
        try:
            if os.path.exists(self.data_directory):
                files = os.listdir(self.data_directory)
                if files:
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
        sample_companies = [
            "TCS", "HDFCBANK", "DMART", "INFY", "RELIANCE", 
            "WIPRO", "BAJFINANCE", "AXISBANK", "ICICIBANK", "SBIN"
        ]
        
        logger.info(f"Using sample companies: {sample_companies}")
        return sample_companies
    
    def save_sample_excel_file(self) -> str:
        """
        Create a sample Excel file for testing purposes
        
        Returns:
            Path to created sample file
        """
        try:
            # Ensure data directory exists
            os.makedirs(self.data_directory, exist_ok=True)
            
            # Create sample data
            sample_data = {
                'company_id': [
                    'TCS', 'HDFCBANK', 'DMART', 'INFY', 'RELIANCE',
                    'WIPRO', 'BAJFINANCE', 'AXISBANK', 'ICICIBANK', 'SBIN',
                    'LT', 'ASIANPAINT', 'NESTLEIND', 'KOTAKBANK', 'BHARTIARTL'
                ],
                'company_name': [
                    'Tata Consultancy Services', 'HDFC Bank', 'D-Mart', 'Infosys', 'Reliance Industries',
                    'Wipro', 'Bajaj Finance', 'Axis Bank', 'ICICI Bank', 'State Bank of India',
                    'Larsen & Toubro', 'Asian Paints', 'Nestle India', 'Kotak Mahindra Bank', 'Bharti Airtel'
                ],
                'sector': [
                    'IT', 'Banking', 'Retail', 'IT', 'Oil & Gas',
                    'IT', 'Financial Services', 'Banking', 'Banking', 'Banking',
                    'Construction', 'Paints', 'FMCG', 'Banking', 'Telecom'
                ]
            }
            
            # Create DataFrame and save to Excel
            df = pd.DataFrame(sample_data)
            sample_file_path = os.path.join(self.data_directory, 'Nifty100Companies.xlsx')
            df.to_excel(sample_file_path, index=False)
            
            logger.info(f"Created sample Excel file: {sample_file_path}")
            logger.info(f"File contains {len(sample_data['company_id'])} companies")
            
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
        # This would be expanded to track loading statistics over time
        return {
            'supported_filenames': self.supported_filenames,
            'supported_columns': self.supported_column_names,
            'data_directory': self.data_directory
        }