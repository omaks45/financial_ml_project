"""
Main Processing Script for Financial ML Analysis
Orchestrates the entire pipeline: API → ML Analysis → Database Storage
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import your project modules
from data.api_client import create_api_client
from database.operations import create_enhanced_database_operations
from config.database import DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_ml_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FinancialMLPipeline:
    """
    Main pipeline class that orchestrates the entire process
    """
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing Financial ML Pipeline...")
        
        # Initialize API client
        self.api_client = create_api_client()
        logger.info("API client initialized")
        
        # Initialize database operations
        self.db_ops = create_enhanced_database_operations()
        logger.info("Database operations initialized")
        
        # Test connections
        self._test_connections()
        
        # Processing statistics
        self.stats = {
            'total_companies': 0,
            'processed_successfully': 0,
            'failed_processing': 0,
            'skipped_existing': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _test_connections(self):
        """Test API and database connections"""
        logger.info("Testing connections...")
        
        # Test API connection
        if not self.api_client.test_connectivity():
            raise Exception("API connectivity test failed!")
        logger.info("✓ API connection successful")
        
        # Test database connection (by getting stats)
        db_stats = self.db_ops.get_database_stats()
        if 'error' in db_stats:
            raise Exception(f"Database connection failed: {db_stats['error']}")
        logger.info("Database connection successful")
        logger.info(f"Current database stats: {db_stats}")
    
    def load_company_list(self, source: str = 'database') -> List[str]:
        """
        Load list of companies to process
        
        Args:
            source: 'database', 'file', or 'sample'
            
        Returns:
            List of company IDs
        """
        if source == 'database':
            # Load from your existing companies table
            companies = self.db_ops.get_company_list()
            company_ids = [comp['id'] for comp in companies if comp.get('id')]
            logger.info(f"Loaded {len(company_ids)} companies from database")
            return company_ids
            
        elif source == 'file':
            # Load from Excel file (you can implement this)
            # For now, return sample list
            logger.info("File loading not implemented yet, using sample list")
            return self.api_client.get_sample_companies()
            
        elif source == 'sample':
            # Use sample companies for testing
            company_ids = self.api_client.get_sample_companies()
            logger.info(f"Using sample companies: {company_ids}")
            return company_ids
            
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def process_single_company(self, company_id: str, skip_existing: bool = True) -> bool:
        """
        Process a single company through the entire pipeline
        
        Args:
            company_id: Company identifier (e.g., 'TCS')
            skip_existing: Skip if analysis already exists
            
        Returns:
            True if successful, False if failed
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Company: {company_id}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Step 1: Check if already processed
            if skip_existing and self.db_ops.check_existing_analysis(company_id):
                logger.info(f"Skipping {company_id} - analysis already exists")
                self.stats['skipped_existing'] += 1
                return True
            
            # Step 2: Log processing start
            self.db_ops.log_processing_step(
                company_id, 'pipeline_start', 'started',
                message="Starting complete pipeline processing"
            )
            
            # Step 3: Fetch data from API
            logger.info(f"Fetching data for {company_id}...")
            self.db_ops.log_processing_step(company_id, 'api_fetch', 'started')
            
            api_data = self.api_client.fetch_company_data(company_id)
            
            if not api_data:
                error_msg = f"Failed to fetch data from API for {company_id}"
                logger.error(f"{error_msg}")
                self.db_ops.log_processing_step(
                    company_id, 'api_fetch', 'failed', 
                    error_details=error_msg
                )
                return False
            
            fetch_time = time.time() - start_time
            self.db_ops.log_processing_step(
                company_id, 'api_fetch', 'completed',
                message=f"API data fetched successfully",
                processing_time=fetch_time
            )
            logger.info(f"Data fetched in {fetch_time:.2f} seconds")
            
            # Step 4: Save financial data to database
            logger.info(f"Saving financial data for {company_id}...")
            financial_data = api_data.get('financial_metrics', {})
            financial_data['raw_data'] = api_data.get('raw_data', {})
            
            if self.db_ops.save_financial_data(company_id, financial_data):
                logger.info(f"Financial data saved for {company_id}")
            else:
                logger.warning(f"Failed to save financial data for {company_id}")
            
            # Step 5: Perform ML Analysis (simplified for now)
            logger.info(f"Performing ML analysis for {company_id}...")
            self.db_ops.log_processing_step(company_id, 'ml_analysis', 'started')
            
            ml_results = self.perform_ml_analysis(company_id, financial_data)
            
            if not ml_results:
                error_msg = f"ML analysis failed for {company_id}"
                logger.error(f"{error_msg}")
                self.db_ops.log_processing_step(
                    company_id, 'ml_analysis', 'failed',
                    error_details=error_msg
                )
                return False
            
            # Step 6: Save ML results to database
            logger.info(f"Saving ML results for {company_id}...")

            # Save to your existing analysis table
            analysis_data = {
                'sales_growth': financial_data.get('sales_growth'),
                'profit_growth': financial_data.get('profit_growth'),
                'stock_cagr': financial_data.get('stock_cagr'),
                'roe': financial_data.get('roe'),
                'debt_to_equity': financial_data.get('debt_to_equity'),
                'current_ratio': financial_data.get('current_ratio'),
                'overall_score': ml_results.get('overall_score', 0)
            }
            
            success1 = self.db_ops.insert_analysis_data(company_id, analysis_data)
            success2 = self.db_ops.insert_pros_cons(
                company_id, 
                ml_results.get('pros', []), 
                ml_results.get('cons', [])
            )
            
            if success1 and success2:
                total_time = time.time() - start_time
                self.db_ops.log_processing_step(
                    company_id, 'ml_analysis', 'completed',
                    message=f"Complete pipeline finished successfully",
                    processing_time=total_time
                )
                
                logger.info(f"Successfully processed {company_id} in {total_time:.2f} seconds")
                logger.info(f"Overall Score: {ml_results.get('overall_score', 0)}")
                logger.info(f"Pros: {len(ml_results.get('pros', []))}")
                logger.info(f"Cons: {len(ml_results.get('cons', []))}")

                self.stats['processed_successfully'] += 1
                return True
            else:
                logger.error(f"Failed to save results for {company_id}")
                return False
                
        except Exception as e:
            error_msg = f"Unexpected error processing {company_id}: {str(e)}"
            logger.error(f"{error_msg}")
            self.db_ops.log_processing_step(
                company_id, 'pipeline_error', 'failed',
                error_details=error_msg
            )
            self.stats['failed_processing'] += 1
            return False
    
    def perform_ml_analysis(self, company_id: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform ML analysis on financial data
        (Simplified version for now - you'll expand this in Day 4)
        
        Args:
            company_id: Company identifier
            financial_data: Financial metrics from API
            
        Returns:
            Dictionary with ML analysis results
        """
        logger.info(f"Analyzing financial data for {company_id}")
        
        try:
            # Simple analysis logic (you'll replace this with actual ML in Day 4)
            roe = financial_data.get('roe', 0) or 0
            debt_to_equity = financial_data.get('debt_to_equity', 0) or 0
            sales_growth = financial_data.get('sales_growth', 0) or 0
            profit_growth = financial_data.get('profit_growth', 0) or 0
            
            # Convert to float if they're strings
            try:
                roe = float(roe) if roe != 'N/A' else 0
                debt_to_equity = float(debt_to_equity) if debt_to_equity != 'N/A' else 0
                sales_growth = float(sales_growth) if sales_growth != 'N/A' else 0
                profit_growth = float(profit_growth) if profit_growth != 'N/A' else 0
            except (ValueError, TypeError):
                logger.warning(f"Some financial data couldn't be converted to numbers for {company_id}")
            
            # Simple pros/cons logic (as per your requirements)
            pros = []
            cons = []
            
            # Pros criteria (values > 10%)
            if roe > 15:
                pros.append(f"Strong return on equity of {roe:.1f}%")
            if debt_to_equity < 0.5:
                pros.append(f"Company is almost debt-free with D/E ratio of {debt_to_equity:.2f}")
            if sales_growth > 10:
                pros.append(f"Good sales growth of {sales_growth:.1f}%")
            if profit_growth > 10:
                pros.append(f"Strong profit growth of {profit_growth:.1f}%")
            
            # Cons criteria (values < 10%)
            if roe < 10:
                cons.append(f"Low return on equity of {roe:.1f}%")
            if sales_growth < 5:
                cons.append(f"Poor sales growth of {sales_growth:.1f}%")
            if profit_growth < 5:
                cons.append(f"Weak profit growth of {profit_growth:.1f}%")
            if debt_to_equity > 1.0:
                cons.append(f"High debt-to-equity ratio of {debt_to_equity:.2f}")
            
            # Limit to 3 pros and 3 cons as per requirements
            pros = pros[:3]
            cons = cons[:3]
            
            # Calculate overall score (simple average)
            scores = [
                min(roe * 2, 40),  # ROE score (max 40)
                max(0, 30 - debt_to_equity * 20),  # Debt score (max 30)
                min(sales_growth, 15),  # Sales growth score (max 15)
                min(profit_growth, 15)   # Profit growth score (max 15)
            ]
            overall_score = sum(scores)
            
            results = {
                'overall_score': round(overall_score, 2),
                'pros': pros,
                'cons': cons,
                'analysis_title': f"{company_id} Financial Analysis - Score: {overall_score:.1f}/100"
            }
            
            logger.info(f"ML analysis completed for {company_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ML analysis for {company_id}: {e}")
            return None
    
    def process_multiple_companies(self, company_ids: List[str], 
                                batch_size: int = 5, 
                                skip_existing: bool = True) -> Dict[str, Any]:
        """
        Process multiple companies one by one
        
        Args:
            company_ids: List of company identifiers
            batch_size: Number of companies to process before pausing
            skip_existing: Skip companies that already have analysis
            
        Returns:
            Processing results summary
        """
        logger.info(f"\nStarting batch processing of {len(company_ids)} companies")
        logger.info(f"Batch size: {batch_size}, Skip existing: {skip_existing}")
        
        self.stats['total_companies'] = len(company_ids)
        self.stats['start_time'] = datetime.now()
        
        # Process companies one by one
        for i, company_id in enumerate(company_ids, 1):
            logger.info(f"\nProgress: {i}/{len(company_ids)} ({i/len(company_ids)*100:.1f}%)")
            
            # Process single company
            success = self.process_single_company(company_id, skip_existing)
            
            # Small pause between companies to be respectful to API
            if i < len(company_ids):
                time.sleep(1)
            
            # Longer pause between batches
            if i % batch_size == 0 and i < len(company_ids):
                logger.info(f"Batch completed. Pausing for 3 seconds...")
                time.sleep(3)
                
                # Show progress
                progress = {
                    'processed': i,
                    'remaining': len(company_ids) - i,
                    'success_rate': f"{self.stats['processed_successfully']/i*100:.1f}%"
                }
                logger.info(f"Batch Progress: {progress}")
        
        self.stats['end_time'] = datetime.now()
        return self._get_processing_summary()
    
    def _get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
        
        summary = {
            'total_companies': self.stats['total_companies'],
            'processed_successfully': self.stats['processed_successfully'],
            'failed_processing': self.stats['failed_processing'],
            'skipped_existing': self.stats['skipped_existing'],
            'success_rate': f"{self.stats['processed_successfully']/max(self.stats['total_companies'], 1)*100:.1f}%",
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time'],
            'total_duration': str(duration) if duration else None
        }
        
        return summary
    
    def print_summary(self):
        """Print processing summary"""
        summary = self._get_processing_summary()
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Total Companies: {summary['total_companies']}")
        print(f"Processed Successfully: {summary['processed_successfully']}")
        print(f"Failed Processing: {summary['failed_processing']}")
        print(f"Skipped (Existing): {summary['skipped_existing']}")
        print(f"Success Rate: {summary['success_rate']}")
        print(f"Total Duration: {summary['total_duration']}")
        print(f"{'='*60}")

def main():
    """
    Main execution function
    """
    print("Financial ML Analysis Pipeline Starting...")
    
    try:
        # Initialize pipeline
        pipeline = FinancialMLPipeline()
        
        # Load company list (you can change source here)
        company_ids = pipeline.load_company_list(source='sample')  # Change to 'database' for full list
        
        if not company_ids:
            logger.error(" No companies to process!")
            return
        
        # Process companies
        results = pipeline.process_multiple_companies(
            company_ids=company_ids,
            batch_size=5,  # Process 5 companies at a time
            skip_existing=True  # Skip companies already analyzed
        )
        
        # Show final summary
        pipeline.print_summary()
        
        # Show database stats
        db_stats = pipeline.db_ops.get_database_stats()
        print(f"\nFinal Database Stats: {db_stats}")
        
        print("\nProcessing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()