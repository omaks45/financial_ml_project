"""
ML Results Database Schema Creation and Testing Script
Design ML results table structure and test data operations

This script creates the necessary database tables for storing ML analysis results
and provides comprehensive testing for data insertion and retrieval operations.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pymysql
from datetime import datetime
import json
import logging
from typing import Dict, List, Any, Optional

# Now import from config
try:
    from config.database import DatabaseConfig
except ImportError:
    # Alternative import method if the above fails
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config.database import DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLDatabaseSchema:
    """
    Handles ML results database schema creation and operations
    """
    
    def __init__(self):
        self.config = DatabaseConfig()
    
    def create_ml_tables(self) -> bool:
        """
        Create all necessary tables for ML analysis results
        Task 3: Design ML results table structure
        """
        try:
            conn = self.config.get_connection()
            if not conn:
                logger.error("Failed to establish database connection")
                return False
            
            cursor = conn.cursor()
            
            # 1. Companies table (if not exists)
            companies_table = """
            CREATE TABLE IF NOT EXISTS companies (
                id INT AUTO_INCREMENT PRIMARY KEY,
                company_id VARCHAR(20) UNIQUE NOT NULL,
                company_name VARCHAR(200) NULL,
                sector VARCHAR(100) NULL,
                industry VARCHAR(100) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_company_id (company_id),
                INDEX idx_sector (sector)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 2. Financial Data table
            financial_data_table = """
            CREATE TABLE IF NOT EXISTS financial_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                company_id VARCHAR(20) NOT NULL,
                
                -- Core Financial Metrics for ML Analysis
                roe DECIMAL(10,2) NULL COMMENT 'Return on Equity %',
                roce DECIMAL(10,2) NULL COMMENT 'Return on Capital Employed %',
                debt_to_equity DECIMAL(10,2) NULL COMMENT 'Debt to Equity Ratio',
                current_ratio DECIMAL(10,2) NULL COMMENT 'Current Ratio',
                book_value DECIMAL(15,2) NULL COMMENT 'Book Value per Share',
                eps DECIMAL(10,2) NULL COMMENT 'Earnings per Share',
                pe_ratio DECIMAL(10,2) NULL COMMENT 'Price to Earnings Ratio',
                dividend_yield DECIMAL(10,2) NULL COMMENT 'Dividend Yield %',
                
                -- Growth Metrics
                sales_growth DECIMAL(10,2) NULL COMMENT 'Sales Growth %',
                profit_growth DECIMAL(10,2) NULL COMMENT 'Profit Growth %',
                
                -- Additional Financial Data
                revenue DECIMAL(15,2) NULL,
                net_income DECIMAL(15,2) NULL,
                total_assets DECIMAL(15,2) NULL,
                total_debt DECIMAL(15,2) NULL,
                market_cap DECIMAL(15,2) NULL,
                
                -- Data Quality & Metadata
                data_completeness_score DECIMAL(5,2) NULL COMMENT 'Data completeness 0-100%',
                raw_api_data JSON NULL COMMENT 'Complete API response',
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                UNIQUE KEY unique_company_financial (company_id),
                INDEX idx_roe (roe),
                INDEX idx_created (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 3. ML Results table (Main table for ML analysis output)
            ml_results_table = """
            CREATE TABLE IF NOT EXISTS ml_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                company_id VARCHAR(20) NOT NULL,
                
                -- ML Analysis Output
                analysis_title VARCHAR(500) NULL COMMENT 'Dynamic analysis title',
                
                -- Pros and Cons (JSON arrays to store up to 3 each)
                pros JSON NULL COMMENT 'Array of up to 3 pros',
                cons JSON NULL COMMENT 'Array of up to 3 cons',
                
                -- ML Scoring System
                overall_score DECIMAL(5,2) NULL COMMENT 'Overall financial health score 0-100',
                growth_score DECIMAL(5,2) NULL COMMENT 'Growth potential score 0-100',
                profitability_score DECIMAL(5,2) NULL COMMENT 'Profitability score 0-100',
                financial_stability_score DECIMAL(5,2) NULL COMMENT 'Financial stability score 0-100',
                
                -- Analysis Metadata
                analysis_version VARCHAR(10) DEFAULT '1.0' COMMENT 'ML model version',
                confidence_score DECIMAL(5,2) NULL COMMENT 'Analysis confidence 0-100',
                
                -- Processing Status
                status ENUM('processing', 'completed', 'failed') DEFAULT 'processing',
                error_message TEXT NULL,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                UNIQUE KEY unique_company_ml (company_id),
                INDEX idx_overall_score (overall_score),
                INDEX idx_status (status),
                INDEX idx_created (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 4. Processing Logs table (for monitoring and debugging)
            processing_logs_table = """
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                company_id VARCHAR(20) NOT NULL,
                
                -- Processing Stage Information
                stage VARCHAR(50) NOT NULL COMMENT 'api_fetch, ml_analysis, database_save',
                status VARCHAR(20) NOT NULL COMMENT 'started, completed, failed',
                
                -- Log Details
                message TEXT NULL,
                error_details TEXT NULL,
                processing_time DECIMAL(10,3) NULL COMMENT 'Processing time in seconds',
                
                -- Batch Processing Support
                batch_id VARCHAR(50) NULL,
                batch_position INT NULL,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_company_stage (company_id, stage, created_at),
                INDEX idx_batch (batch_id, batch_position),
                INDEX idx_status_created (status, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Execute table creation
            tables = [
                ("companies", companies_table),
                ("financial_data", financial_data_table),
                ("ml_results", ml_results_table),
                ("processing_logs", processing_logs_table)
            ]
            
            for table_name, table_sql in tables:
                try:
                    cursor.execute(table_sql)
                    logger.info(f"✓ Created/verified table: {table_name}")
                except Exception as e:
                    logger.error(f"✗ Error creating table {table_name}: {e}")
                    return False
            
            conn.close()
            logger.info("🎉 All ML database tables created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ML database schema: {e}")
            return False
    
    def insert_test_data(self) -> bool:
        """
        Task 4: Test data insertion and retrieval
        Insert comprehensive test data for ML analysis
        """
        try:
            conn = self.config.get_connection()
            cursor = conn.cursor()
            
            # Test companies data
            test_companies = [
                {
                    'company_id': 'TCS',
                    'company_name': 'Tata Consultancy Services',
                    'sector': 'Information Technology',
                    'industry': 'Software Services'
                },
                {
                    'company_id': 'HDFCBANK',
                    'company_name': 'HDFC Bank Limited',
                    'sector': 'Financial Services',
                    'industry': 'Private Banks'
                },
                {
                    'company_id': 'RELIANCE',
                    'company_name': 'Reliance Industries Limited',
                    'sector': 'Energy',
                    'industry': 'Oil & Gas'
                }
            ]
            
            # Insert companies
            for company in test_companies:
                insert_company = """
                INSERT INTO companies (company_id, company_name, sector, industry)
                VALUES (%(company_id)s, %(company_name)s, %(sector)s, %(industry)s)
                ON DUPLICATE KEY UPDATE
                    company_name = VALUES(company_name),
                    sector = VALUES(sector),
                    industry = VALUES(industry),
                    updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(insert_company, company)
                logger.info(f"✓ Inserted/updated company: {company['company_id']}")
            
            # Test financial data
            test_financial_data = [
                {
                    'company_id': 'TCS',
                    'roe': 25.40,
                    'roce': 30.20,
                    'debt_to_equity': 0.10,
                    'current_ratio': 2.50,
                    'book_value': 45.30,
                    'eps': 125.50,
                    'pe_ratio': 28.60,
                    'dividend_yield': 3.20,
                    'sales_growth': 15.30,
                    'profit_growth': 18.20,
                    'revenue': 1940000.00,
                    'net_income': 386000.00,
                    'market_cap': 12500000.00,
                    'data_completeness_score': 95.00,
                    'raw_api_data': json.dumps({"source": "test_data", "timestamp": str(datetime.now())})
                },
                {
                    'company_id': 'HDFCBANK',
                    'roe': 17.80,
                    'roce': 18.50,
                    'debt_to_equity': 5.20,
                    'current_ratio': 1.10,
                    'book_value': 1580.20,
                    'eps': 65.40,
                    'pe_ratio': 22.30,
                    'dividend_yield': 1.50,
                    'sales_growth': 12.80,
                    'profit_growth': 16.40,
                    'revenue': 1560000.00,
                    'net_income': 364000.00,
                    'market_cap': 8900000.00,
                    'data_completeness_score': 92.00,
                    'raw_api_data': json.dumps({"source": "test_data", "timestamp": str(datetime.now())})
                },
                {
                    'company_id': 'RELIANCE',
                    'roe': 12.30,
                    'roce': 11.80,
                    'debt_to_equity': 0.35,
                    'current_ratio': 1.80,
                    'book_value': 980.50,
                    'eps': 87.20,
                    'pe_ratio': 25.40,
                    'dividend_yield': 0.80,
                    'sales_growth': 8.60,
                    'profit_growth': 22.10,
                    'revenue': 6929000.00,
                    'net_income': 534000.00,
                    'market_cap': 15200000.00,
                    'data_completeness_score': 89.00,
                    'raw_api_data': json.dumps({"source": "test_data", "timestamp": str(datetime.now())})
                }
            ]
            
            # Insert financial data
            for financial in test_financial_data:
                insert_financial = """
                INSERT INTO financial_data (
                    company_id, roe, roce, debt_to_equity, current_ratio, book_value,
                    eps, pe_ratio, dividend_yield, sales_growth, profit_growth,
                    revenue, net_income, market_cap, data_completeness_score, raw_api_data
                ) VALUES (
                    %(company_id)s, %(roe)s, %(roce)s, %(debt_to_equity)s, %(current_ratio)s, %(book_value)s,
                    %(eps)s, %(pe_ratio)s, %(dividend_yield)s, %(sales_growth)s, %(profit_growth)s,
                    %(revenue)s, %(net_income)s, %(market_cap)s, %(data_completeness_score)s, %(raw_api_data)s
                ) ON DUPLICATE KEY UPDATE
                    roe = VALUES(roe),
                    roce = VALUES(roce),
                    debt_to_equity = VALUES(debt_to_equity),
                    current_ratio = VALUES(current_ratio),
                    book_value = VALUES(book_value),
                    eps = VALUES(eps),
                    pe_ratio = VALUES(pe_ratio),
                    dividend_yield = VALUES(dividend_yield),
                    sales_growth = VALUES(sales_growth),
                    profit_growth = VALUES(profit_growth),
                    revenue = VALUES(revenue),
                    net_income = VALUES(net_income),
                    market_cap = VALUES(market_cap),
                    data_completeness_score = VALUES(data_completeness_score),
                    raw_api_data = VALUES(raw_api_data),
                    updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(insert_financial, financial)
                logger.info(f"✓ Inserted/updated financial data: {financial['company_id']}")
            
            # Test ML results data
            test_ml_results = [
                {
                    'company_id': 'TCS',
                    'analysis_title': 'TCS demonstrates exceptional financial performance with strong profitability metrics',
                    'pros': json.dumps([
                        'Excellent ROE of 25.4% indicating strong profitability',
                        'Very low debt-to-equity ratio of 0.10 showing financial stability',
                        'Consistent profit growth of 18.2% year-over-year'
                    ]),
                    'cons': json.dumps([
                        'High PE ratio of 28.6 might indicate overvaluation',
                        'Dividend yield of 3.2% could be improved'
                    ]),
                    'overall_score': 87.50,
                    'growth_score': 82.00,
                    'profitability_score': 95.00,
                    'financial_stability_score': 88.00,
                    'confidence_score': 92.00,
                    'status': 'completed'
                },
                {
                    'company_id': 'HDFCBANK',
                    'analysis_title': 'HDFC Bank shows solid banking fundamentals with steady growth trajectory',
                    'pros': json.dumps([
                        'Strong ROE of 17.8% for banking sector',
                        'Consistent profit growth of 16.4%',
                        'Stable market position in private banking'
                    ]),
                    'cons': json.dumps([
                        'High debt-to-equity ratio typical of banking sector',
                        'Lower dividend yield of 1.5%',
                        'Current ratio of 1.1 indicates tight liquidity'
                    ]),
                    'overall_score': 78.30,
                    'growth_score': 75.00,
                    'profitability_score': 83.00,
                    'financial_stability_score': 72.00,
                    'confidence_score': 88.00,
                    'status': 'completed'
                },
                {
                    'company_id': 'RELIANCE',
                    'analysis_title': 'Reliance Industries balances diversified operations with growth initiatives',
                    'pros': json.dumps([
                        'Strong profit growth of 22.1% despite market challenges',
                        'Diversified business portfolio reducing risk',
                        'Good current ratio of 1.8 indicating liquidity strength'
                    ]),
                    'cons': json.dumps([
                        'ROE of 12.3% below industry leaders',
                        'Low dividend yield of 0.8%',
                        'Slower sales growth of 8.6% compared to profit growth'
                    ]),
                    'overall_score': 74.80,
                    'growth_score': 70.00,
                    'profitability_score': 68.00,
                    'financial_stability_score': 82.00,
                    'confidence_score': 85.00,
                    'status': 'completed'
                }
            ]
            
            # Insert ML results
            for ml_result in test_ml_results:
                insert_ml_result = """
                INSERT INTO ml_results (
                    company_id, analysis_title, pros, cons, overall_score,
                    growth_score, profitability_score, financial_stability_score,
                    confidence_score, status
                ) VALUES (
                    %(company_id)s, %(analysis_title)s, %(pros)s, %(cons)s, %(overall_score)s,
                    %(growth_score)s, %(profitability_score)s, %(financial_stability_score)s,
                    %(confidence_score)s, %(status)s
                ) ON DUPLICATE KEY UPDATE
                    analysis_title = VALUES(analysis_title),
                    pros = VALUES(pros),
                    cons = VALUES(cons),
                    overall_score = VALUES(overall_score),
                    growth_score = VALUES(growth_score),
                    profitability_score = VALUES(profitability_score),
                    financial_stability_score = VALUES(financial_stability_score),
                    confidence_score = VALUES(confidence_score),
                    status = VALUES(status),
                    updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(insert_ml_result, ml_result)
                logger.info(f"✓ Inserted/updated ML results: {ml_result['company_id']}")
            
            conn.close()
            logger.info(" All test data inserted successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting test data: {e}")
            return False
    
    def test_data_retrieval(self) -> bool:
        """
        Task 4: Test data retrieval operations
        Comprehensive testing of data retrieval with various queries
        """
        try:
            conn = self.config.get_connection()
            cursor = conn.cursor()
            
            logger.info("Starting data retrieval tests...")
            
            # Test 1: Retrieve all companies with basic info
            logger.info("\n--- Test 1: Company List Retrieval ---")
            cursor.execute("""
                SELECT company_id, company_name, sector, industry, created_at
                FROM companies
                ORDER BY company_name
            """)
            companies = cursor.fetchall()
            logger.info(f"✓ Retrieved {len(companies)} companies:")
            for company in companies:
                logger.info(f"  - {company['company_id']}: {company['company_name']} ({company['sector']})")
            
            # Test 2: Retrieve financial data with metrics
            logger.info("\n--- Test 2: Financial Data Retrieval ---")
            cursor.execute("""
                SELECT f.company_id, c.company_name, f.roe, f.roce, 
                    f.debt_to_equity, f.current_ratio
                FROM financial_data f
                JOIN companies c ON f.company_id = c.company_id
                ORDER BY f.roe DESC
            """)
            financial_data = cursor.fetchall()
            logger.info(f"✓ Retrieved financial data for {len(financial_data)} companies:")
            for data in financial_data:
                logger.info(f"  - {data['company_id']}: ROE={data['roe']}%, ROCE={data['roce']}%")
            
            # Test 3: Retrieve ML results with scores
            logger.info("\n--- Test 3: ML Results Retrieval ---")
            cursor.execute("""
                SELECT ml.company_id, c.company_name, ml.analysis_title,
                    ml.overall_score, ml.growth_score, ml.profitability_score,
                    ml.financial_stability_score, ml.status
                FROM ml_results ml
                JOIN companies c ON ml.company_id = c.company_id
                ORDER BY ml.overall_score DESC
            """)
            ml_results = cursor.fetchall()
            logger.info(f"✓ Retrieved ML results for {len(ml_results)} companies:")
            for result in ml_results:
                logger.info(f"  - {result['company_id']}: Score={result['overall_score']}, Status={result['status']}")
                logger.info(f"    Title: {result['analysis_title'][:80]}...")
            
            # Test 4: Retrieve pros and cons
            logger.info("\n--- Test 4: Pros and Cons Retrieval ---")
            cursor.execute("""
                SELECT company_id, pros, cons
                FROM ml_results
                WHERE status = 'completed'
            """)
            pros_cons_data = cursor.fetchall()
            logger.info(f"✓ Retrieved pros/cons for {len(pros_cons_data)} companies:")
            for data in pros_cons_data:
                pros = json.loads(data['pros']) if data['pros'] else []
                cons = json.loads(data['cons']) if data['cons'] else []
                logger.info(f"  - {data['company_id']}: {len(pros)} pros, {len(cons)} cons")
                if pros:
                    logger.info(f"    First Pro: {pros[0][:60]}...")
                if cons:
                    logger.info(f"    First Con: {cons[0][:60]}...")
            
            # Test 5: Data quality and completeness check
            logger.info("\n--- Test 5: Data Quality Assessment ---")
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_companies,
                    SUM(CASE WHEN f.company_id IS NOT NULL THEN 1 ELSE 0 END) as with_financial_data,
                    SUM(CASE WHEN ml.company_id IS NOT NULL THEN 1 ELSE 0 END) as with_ml_results,
                    SUM(CASE WHEN ml.status = 'completed' THEN 1 ELSE 0 END) as completed_analysis,
                    AVG(f.data_completeness_score) as avg_data_completeness
                FROM companies c
                LEFT JOIN financial_data f ON c.company_id = f.company_id
                LEFT JOIN ml_results ml ON c.company_id = ml.company_id
            """)
            quality_stats = cursor.fetchone()
            logger.info("✓ Data Quality Statistics:")
            for key, value in quality_stats.items():
                if isinstance(value, float) and value is not None:
                    logger.info(f"  - {key}: {value:.2f}")
                else:
                    logger.info(f"  - {key}: {value}")
            
            conn.close()
            logger.info("All data retrieval tests completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in data retrieval tests: {e}")
            return False
    
    def run_complete_test_suite(self) -> bool:
        """
        Run complete test suite for Day 2 Tasks 3 & 4
        """
        logger.info("Starting ML Database Schema Test Suite for Day 2")
        logger.info("="*60)
        
        # Step 1: Test database connection
        logger.info("Step 1: Testing database connection...")
        if not self.config.test_connection():
            logger.error("Database connection failed!")
            return False
        
        # Step 2: Create ML database schema
        logger.info("\nStep 2: Creating ML database schema...")
        if not self.create_ml_tables():
            logger.error(" Schema creation failed!")
            return False
        
        # Step 3: Insert test data
        logger.info("\nStep 3: Inserting test data...")
        if not self.insert_test_data():
            logger.error("Test data insertion failed!")
            return False
        
        # Step 4: Test data retrieval
        logger.info("\nStep 4: Testing data retrieval operations...")
        if not self.test_data_retrieval():
            logger.error("Data retrieval tests failed!")
            return False
        
        logger.info("\n" + "="*60)
        logger.info("ML Database Schema Test Suite COMPLETED SUCCESSFULLY!")
        logger.info("ML results table structure designed and implemented")
        logger.info("Data insertion and retrieval operations tested")
        logger.info("="*60)
        
        return True

# Utility functions for easy usage
def create_ml_schema_tester():
    """Factory function to create ML schema tester"""
    return MLDatabaseSchema()

def run_day2_tasks():
    """Run Day 2 Tasks 3 & 4"""
    tester = create_ml_schema_tester()
    return tester.run_complete_test_suite()

# Main execution
if __name__ == "__main__":
    #print("Starting Day 2 Tasks 3 & 4...")
    print("   Design ML results table structure")
    print("   Test data insertion and retrieval operations")
    print("-" * 60)
    
    # Run Day 2 Tasks 3 & 4
    success = run_day2_tasks()
    
    print("\n" + "="*60)
    if success:
        print("completed successfully!")
        print("ML results table structure designed")
        print("Data insertion and retrieval tested")
        print("Database ready for ML pipeline integration")
    else:
        print("failed. Please check the logs for details.")
    print("="*60)