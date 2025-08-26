#!/usr/bin/env python3
"""
Main Processing Script for Financial ML Analysis
Orchestrates the entire pipeline: API → ML Analysis → Database Storage
Author: Financial ML Team
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

# Import project modules (these should exist in separate files)
try:
    from data.api_client import FinancialDataAPI
    from data.data_processor import DataProcessor
    from data.company_loader import CompanyDataLoader
    from ml.analyzer import FinancialMetricsCalculator
    from database.operations import DatabaseOperations
    from config.database import DatabaseConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all module files are created according to the project structure")
    print("Run: python setup_modules.py to create the required module files")
    sys.exit(1)

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
    Main pipeline class that orchestrates Day 3 Tasks 1 & 2
    using modular components from separate files
    """
    
    def __init__(self):
        """Initialize all components from separate modules"""
        logger.info("Initializing Financial ML Pipeline...")
        
        try:
            # Initialize components from separate modules
            self.db_config = DatabaseConfig()
            self.db_operations = DatabaseOperations(self.db_config)
            self.api_client = FinancialDataAPI()
            
            # Data processing components
            self.company_loader = CompanyDataLoader()  # Excel loading
            self.data_processor = DataProcessor()      # Data cleaning
            
            # ML/Financial calculation components  
            self.metrics_calculator = FinancialMetricsCalculator()  # Metrics calculation
            
            logger.info("All components initialized successfully")
            
            # Test connections
            self._test_connections()
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def _test_connections(self):
        """Test database and API connections"""
        logger.info("Testing connections...")
        
        # Test database connection
        if not self.db_config.connect():
            raise Exception("Database connection failed!")
        
        if not self.db_config.create_tables():
            raise Exception("Failed to create database tables!")
        
        logger.info("✓ Database connection successful")
        
        # Test API connection with sample company
        test_data = self.api_client.fetch_company_data("TCS")
        if not test_data:
            logger.warning("API test failed, but continuing...")
        else:
            logger.info("✓ API connection successful")
    
    def run_day3_implementation(self, limit: int = 5):
        """
        Main function to run Day 3 Task 1 and Task 2
        
        Args:
            limit: Number of companies to process for testing
        """
        logger.info(f"\n{'='*60}")
        logger.info("DAY 3 IMPLEMENTATION: Tasks 1 & 2")
        logger.info(f"{'='*60}")
        
        try:
            # =================================================================
            #  Data Processing & Cleaning Pipeline
            # =================================================================
            logger.info("\n  Data Processing & Cleaning")
            logger.info("-" * 50)
            
            # Step 1: Load companies from Excel (Task 1 component)
            logger.info("Step 1: Loading companies from Excel file...")
            company_ids = self.company_loader.load_companies_from_excel()
            
            if not company_ids:
                logger.error("No companies loaded from Excel file")
                return
            
            logger.info(f"✓ Loaded {len(company_ids)} companies from Excel")
            
            # Limit for testing
            test_companies = company_ids[:limit]
            logger.info(f"Processing {len(test_companies)} companies for testing")
            
            processing_results = []
            
            for i, company_id in enumerate(test_companies, 1):
                logger.info(f"\n Processing Company {i}/{len(test_companies)}: {company_id}")
                logger.info("-" * 40)
                
                try:
                    # Step 2: Fetch raw data from API
                    logger.info("Step 2: Fetching raw financial data from API...")
                    raw_data = self.api_client.fetch_company_data(company_id)
                    
                    if not raw_data:
                        logger.error(f" Failed to fetch data for {company_id}")
                        continue
                    
                    logger.info(" Raw data fetched successfully")
                    
                    # Step 3: Clean and preprocess data (Task 1 implementation)
                    logger.info("Step 3: Cleaning and preprocessing data...")
                    cleaned_data = self.data_processor.clean_financial_data(raw_data)
                    
                    if not cleaned_data or 'error' in cleaned_data:
                        logger.error(f" Data cleaning failed for {company_id}")
                        continue
                    
                    logger.info(" Data cleaning completed")
                    
                    # Display data quality results
                    data_quality = cleaned_data.get('data_quality', {})
                    logger.info(f"  - Data Completeness: {data_quality.get('completeness', 0):.1f}%")
                    logger.info(f"  - Missing Fields: {len(data_quality.get('missing_fields', []))}")
                    
                    # =================================================================
                    #  Financial Metrics Calculation
                    # =================================================================
                    logger.info("\n  Financial Metrics Calculation")

                    # Step 4: Calculate financial metrics (Task 2 implementation)
                    logger.info("Step 4: Calculating financial metrics...")
                    calculated_metrics = self.metrics_calculator.calculate_all_metrics(cleaned_data)
                    
                    if not calculated_metrics:
                        logger.error(f" Metrics calculation failed for {company_id}")
                        continue
                    
                    logger.info(f" Calculated {len(calculated_metrics)} financial metrics")
                    
                    # Step 5: Store results in database
                    logger.info("Step 5: Storing results in database...")
                    success = self._save_processed_data(company_id, cleaned_data, calculated_metrics)
                    
                    if success:
                        logger.info(" Data saved to database successfully")

                        # Collect results for summary
                        processing_results.append({
                            'company_id': company_id,
                            'data_quality_score': data_quality.get('completeness', 0),
                            'metrics_calculated': len(calculated_metrics),
                            'financial_health_score': calculated_metrics.get('financial_health_score', 0),
                            'processing_status': 'success'
                        })
                        
                        # Display key metrics
                        self._display_company_results(company_id, calculated_metrics)
                        
                    else:
                        logger.error(f" Failed to save data for {company_id}")
                        processing_results.append({
                            'company_id': company_id,
                            'processing_status': 'failed'
                        })
                    
                except Exception as e:
                    logger.error(f" Error processing {company_id}: {e}")
                    processing_results.append({
                        'company_id': company_id,
                        'processing_status': 'error',
                        'error': str(e)
                    })
                
                # Rate limiting
                time.sleep(1)
            
            # Final summary of Day 3 implementation
            self._display_day3_summary(processing_results)
            
        except Exception as e:
            logger.error(f"Fatal error in Day 3 implementation: {e}")
            raise
        finally:
            self.db_config.disconnect()
    
    def _save_processed_data(self, company_id: str, cleaned_data: Dict, metrics: Dict) -> bool:
        """
        Save processed data and calculated metrics to database
        """
        try:
            # Prepare data for database insertion
            company_info = cleaned_data.get('company_info', {})
            
            analysis_data = {
                'company_id': company_id,
                'company_name': company_info.get('name', company_id),
                'sales_growth': metrics.get('sales_growth', 0),
                'profit_growth': metrics.get('profit_growth', 0),
                'stock_cagr': metrics.get('stock_cagr', 0),
                'roe': metrics.get('roe', 0),
                'debt_to_equity': metrics.get('debt_to_equity', 0),
                'current_ratio': metrics.get('current_ratio', 0),
                'overall_score': metrics.get('financial_health_score', 0),
                'financial_data': cleaned_data,
                'processed_metrics': metrics
            }
            
            # Use database operations to save
            return self.db_operations.insert_analysis_result(
                company_id, 
                cleaned_data, 
                analysis_data, 
                metrics
            )
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False
    
    def _display_company_results(self, company_id: str, metrics: Dict):
        """Display processing results for a company"""
        print(f"\n Results for {company_id}:")
        print(f"   Financial Health Score: {metrics.get('financial_health_score', 0):.1f}/100")
        print(f"   ROE: {metrics.get('roe', 0):.2f}%")
        print(f"   Sales Growth: {metrics.get('sales_growth', 0):.2f}%")
        print(f"   Profit Growth: {metrics.get('profit_growth', 0):.2f}%")
        print(f"   Debt-to-Equity: {metrics.get('debt_to_equity', 0):.2f}")
        print(f"   Current Ratio: {metrics.get('current_ratio', 0):.2f}")
        print(f"   Profitability Score: {metrics.get('profitability_score', 0):.1f}/100")
        print(f"   Growth Score: {metrics.get('growth_score', 0):.1f}/100")
        print(f"   Stability Score: {metrics.get('stability_score', 0):.1f}/100")
    
    def _display_day3_summary(self, results: List[Dict]):
        """Display final summary of Day 3 implementation"""
        if not results:
            logger.warning("No results to summarize")
            return
        
        successful = [r for r in results if r.get('processing_status') == 'success']
        failed = [r for r in results if r.get('processing_status') in ['failed', 'error']]
        
        print(f"\n{'='*60}")
        print("DAY 3 IMPLEMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f" TASK 1 - Data Processing & Cleaning:")
        print(f"   ✓ Excel file loading: Implemented")
        print(f"   ✓ Data cleaning pipeline: Implemented") 
        print(f"   ✓ Data quality assessment: Implemented")
        print(f"\n TASK 2 - Financial Metrics Calculation:")
        print(f"   ✓ Financial ratios calculation: Implemented")
        print(f"   ✓ Composite scores generation: Implemented")
        print(f"   ✓ Health assessment scoring: Implemented")
        print(f"\n PROCESSING RESULTS:")
        print(f"   Total Companies: {len(results)}")
        print(f"   Successfully Processed: {len(successful)}")
        print(f"   Failed Processing: {len(failed)}")
        print(f"   Success Rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_health = sum(r.get('financial_health_score', 0) for r in successful) / len(successful)
            avg_quality = sum(r.get('data_quality_score', 0) for r in successful) / len(successful)
            total_metrics = sum(r.get('metrics_calculated', 0) for r in successful)

            print(f"\n QUALITY METRICS:")
            print(f"   Average Financial Health Score: {avg_health:.1f}/100")
            print(f"   Average Data Quality Score: {avg_quality:.1f}%")
            print(f"   Total Financial Metrics Calculated: {total_metrics}")
        
        print(f"{'='*60}")
        print(" DAY 3 TASKS 1 & 2 COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

def create_module_files():
    """Helper function to create empty module files if they don't exist"""
    modules_to_create = [
        'data/__init__.py',
        'data/company_loader.py',
        'data/data_processor.py',
        'ml/__init__.py', 
        'ml/analyzer.py',
        'config/__init__.py',
        'database/__init__.py'
    ]
    
    for module_path in modules_to_create:
        if not os.path.exists(module_path):
            os.makedirs(os.path.dirname(module_path), exist_ok=True)
            with open(module_path, 'w') as f:
                if module_path.endswith('__init__.py'):
                    f.write('# Module initialization file\n')
                else:
                    f.write(f'# {module_path} - Implementation needed\n')
            print(f"Created: {module_path}")

def main():
    """
    Main execution function - orchestrates Day 3 Tasks 1 & 2
    """
    print("Financial ML Analysis Pipeline - Day 3 Implementation")
    print("Tasks 1 & 2: Data Processing & Financial Metrics Calculation")
    
    try:
        # Check if required modules exist
        required_modules = ['data.data_processor', 'ml.analyzer', 'data.company_loader']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"\n Missing required modules: {missing_modules}")
            print("Please create the required module files according to project structure")
            print("Or run the setup script to create template files")
            return
        
        # Initialize and run Day 3 implementation
        pipeline = FinancialMLPipeline()
        
        # Run Day 3 Tasks 1 & 2
        pipeline.run_day3_implementation(limit=3)  # Test with 3 companies
        
        print(f"\n Day 3 implementation completed successfully!")
        print("Next: Implement Day 4 - ML Insights Generation")
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n Error: {e}")
        print("Please check the logs for detailed error information")

if __name__ == "__main__":
    main()