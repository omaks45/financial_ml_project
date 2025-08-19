"""
Financial Data API Client
Handles fetching financial data from the StockTicker API
"""

import requests
import json
from typing import Dict, Any, Optional, List
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataAPI:
    """
    API client for fetching financial data from StockTicker API
    """
    
    def __init__(self):
        """Initialize the API client with base configuration"""
        self.base_url = "https://bluemutualfund.in/server/api/company.php"
        self.api_key = "ghfkffu6378382826hhdjgk"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Financial-ML-Analysis/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # seconds between requests
        
    def _make_request(self, company_id: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Make API request with error handling and rate limiting
        
        Args:
            company_id: Company identifier (e.g., 'TCS', 'HDFCBANK')
            retries: Number of retry attempts
            
        Returns:
            Dictionary containing API response or None if failed
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        params = {
            'id': company_id,
            'api_key': self.api_key
        }
        
        for attempt in range(retries):
            try:
                logger.info(f"Fetching data for {company_id} (attempt {attempt + 1}/{retries})")
                
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=30
                )
                
                self.last_request_time = time.time()
                
                # Check if request was successful
                response.raise_for_status()
                
                # Try to parse JSON
                try:
                    data = response.json()
                    logger.info(f"Successfully fetched data for {company_id}")
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {company_id}: {e}")
                    logger.error(f"Response content: {response.text[:200]}...")
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {company_id} (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {company_id}")
                    
        return None
    
    def fetch_company_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive financial data for a single company
        
        Args:
            company_id: Company identifier
            
        Returns:
            Dictionary with structured financial data or None if failed
        """
        raw_data = self._make_request(company_id)
        
        if not raw_data:
            return None
            
        # Structure the data for easier processing
        structured_data = self._structure_company_data(company_id, raw_data)
        return structured_data
    
    def _structure_company_data(self, company_id: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure raw API response into organized format
        
        Args:
            company_id: Company identifier
            raw_data: Raw API response
            
        Returns:
            Structured financial data dictionary
        """
        structured = {
            'company_id': company_id,
            'company_info': {},
            'financial_metrics': {},
            'balance_sheet': {},
            'profit_loss': {},
            'cash_flow': {},
            'ratios': {},
            'growth_metrics': {},
            'raw_data': raw_data  # Keep raw data for debugging
        }
        
        # Extract company information
        if isinstance(raw_data, dict):
            # Basic company info
            structured['company_info'] = {
                'name': raw_data.get('company_name', 'N/A'),
                'symbol': company_id,
                'sector': raw_data.get('sector', 'N/A'),
                'industry': raw_data.get('industry', 'N/A')
            }
            
            # Extract financial metrics that are commonly used for ML analysis
            self._extract_financial_metrics(raw_data, structured)
            
        return structured
    
    def _extract_financial_metrics(self, raw_data: Dict[str, Any], structured: Dict[str, Any]):
        """
        Extract key financial metrics from raw data
        
        Args:
            raw_data: Raw API response
            structured: Structured data dictionary to update
        """
        # Common financial metrics for ML analysis
        metrics_mapping = {
            'roe': ['roe_percentage', 'roe', 'return_on_equity'],
            'roce': ['roce_percentage', 'roce', 'return_on_capital_employed'],
            'debt_to_equity': ['debt_to_equity_ratio', 'debt_equity', 'de_ratio'],
            'current_ratio': ['current_ratio', 'cr'],
            'book_value': ['book_value_per_share', 'book_value', 'bvps'],
            'eps': ['earnings_per_share', 'eps'],
            'pe_ratio': ['pe_ratio', 'pe', 'price_earnings'],
            'dividend_yield': ['dividend_yield', 'dy'],
            'sales_growth': ['sales_growth', 'revenue_growth'],
            'profit_growth': ['profit_growth', 'net_income_growth']
        }
        
        # Extract metrics
        for metric, possible_keys in metrics_mapping.items():
            value = None
            for key in possible_keys:
                if key in raw_data:
                    value = raw_data[key]
                    break
            structured['financial_metrics'][metric] = value
    
    def fetch_multiple_companies(self, company_ids: List[str], 
                                batch_size: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Fetch data for multiple companies with batch processing
        
        Args:
            company_ids: List of company identifiers
            batch_size: Number of companies to process at once
            
        Returns:
            Dictionary mapping company_id to financial data
        """
        results = {}
        
        logger.info(f"Starting batch fetch for {len(company_ids)} companies")
        
        for i in range(0, len(company_ids), batch_size):
            batch = company_ids[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            for company_id in batch:
                data = self.fetch_company_data(company_id)
                if data:
                    results[company_id] = data
                    logger.info(f"✓ Successfully fetched: {company_id}")
                else:
                    logger.error(f"✗ Failed to fetch: {company_id}")
                    results[company_id] = None
            
            # Small delay between batches
            if i + batch_size < len(company_ids):
                logger.info("Pausing between batches...")
                time.sleep(2)
        
        logger.info(f"Batch fetch completed. Success: {sum(1 for v in results.values() if v is not None)}/{len(company_ids)}")
        return results
    
    def test_connectivity(self, test_company: str = 'TCS') -> bool:
        """
        Test API connectivity with a sample company
        
        Args:
            test_company: Company ID to test with
            
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Testing API connectivity with {test_company}...")
        
        try:
            data = self.fetch_company_data(test_company)
            if data:
                logger.info("✓ API connectivity test PASSED")
                logger.info(f"Sample data keys: {list(data.keys())}")
                return True
            else:
                logger.error("✗ API connectivity test FAILED - No data returned")
                return False
        except Exception as e:
            logger.error(f"✗ API connectivity test FAILED - Exception: {e}")
            return False
    
    def get_sample_companies(self) -> List[str]:
        """
        Get a list of sample company IDs for testing
        
        Returns:
            List of company identifiers
        """
        # Based on the Nifty100 companies from the documentation
        sample_companies = [
            'TCS', 'HDFCBANK', 'DMART', 'INFY', 'WIPRO', 
            'SBILIFE', 'RELIANCE', 'ASIANPAINT', 'ICICIBANK', 
            'MARUTI', 'BHARTIARTL', 'KOTAKBANK', 'ITC', 
            'HINDUNILVR', 'BAJFINANCE', 'TITAN', 'SUNPHARMA',
            'NESTLEIND', 'ULTRACEMCO', 'TATASTEEL'
        ]
        return sample_companies
    
    def save_data_to_file(self, data: Dict[str, Any], filename: str):
        """
        Save fetched data to JSON file for backup/debugging
        
        Args:
            data: Data to save
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data to {filename}: {e}")


# Utility function for easy import
def create_api_client() -> FinancialDataAPI:
    """
    Factory function to create and return API client instance
    
    Returns:
        FinancialDataAPI instance
    """
    return FinancialDataAPI()


# Example usage and testing code
if __name__ == "__main__":
    # Example usage
    api = create_api_client()
    
    # Test connectivity
    if api.test_connectivity():
        print("API is working correctly!")
        
        # Fetch sample data
        sample_companies = ['TCS', 'HDFCBANK', 'DMART']
        results = api.fetch_multiple_companies(sample_companies)
        
        # Display results
        for company_id, data in results.items():
            if data:
                print(f"\n=== {company_id} ===")
                print(f"Company Name: {data['company_info'].get('name', 'N/A')}")
                print("Financial Metrics:")
                for metric, value in data['financial_metrics'].items():
                    if value is not None:
                        print(f"  {metric}: {value}")
            else:
                print(f"\n=== {company_id} === FAILED")
    else:
        print("API connectivity test failed!")