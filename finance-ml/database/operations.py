"""
Enhanced Database Operations - Hybrid Approach

"""

import mysql.connector
from datetime import datetime
from config.database import DatabaseConfig
import json
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDatabaseOperations:
    """
    Enhanced version of your existing DatabaseOperations class
    Keeps your existing table structure but adds improvements
    """
    
    def __init__(self):
        self.config = DatabaseConfig()
        # Add connection pooling settings
        self.pool_config = {
            'pool_name': 'financial_ml_pool',
            'pool_size': 10,
            'pool_reset_session': True,
            'autocommit': True
        }
    
    @contextmanager
    def get_connection(self):
        """
        Enhanced connection method with proper error handling
        Uses context manager for automatic cleanup
        """
        conn = None
        try:
            conn = self.config.get_connection()
            if not conn:
                raise Exception("Failed to establish database connection")
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def insert_pros_cons(self, company_id: str, pros_list: List[str], cons_list: List[str]) -> bool:
        """
        Enhanced pros_cons insertion
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                conn.start_transaction()
                
                try:
                    # Clear existing entries for this company
                    delete_query = "DELETE FROM prosandcons WHERE company_id = %s"
                    cursor.execute(delete_query, (company_id,))
                    logger.info(f"Cleared existing pros/cons for {company_id}")
                    
                    # Insert new pros (limit to 3 as per requirements)
                    for i, pro in enumerate(pros_list[:3]):  # Limit to 3
                        insert_query = """
                        INSERT INTO prosandcons (company_id, pros, cons, created_at) 
                        VALUES (%s, %s, NULL, %s)
                        """
                        cursor.execute(insert_query, (company_id, pro, datetime.now()))
                    
                    # Insert new cons (limit to 3 as per requirements)
                    for i, con in enumerate(cons_list[:3]):  # Limit to 3
                        insert_query = """
                        INSERT INTO prosandcons (company_id, pros, cons, created_at) 
                        VALUES (%s, NULL, %s, %s)
                        """
                        cursor.execute(insert_query, (company_id, con, datetime.now()))
                    
                    # Commit transaction
                    conn.commit()
                    logger.info(f"✓ Successfully saved {len(pros_list[:3])} pros and {len(cons_list[:3])} cons for {company_id}")
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Transaction failed for {company_id}: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Database error for {company_id}: {e}")
            return False
    
    def insert_analysis_data(self, company_id: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Enhanced version of your existing analysis data insertion
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                conn.start_transaction()
                
                try:
                    # Clear existing entries for this company
                    delete_query = "DELETE FROM analysis WHERE company_id = %s"
                    cursor.execute(delete_query, (company_id,))
                    
                    # Insert new analysis data with more fields
                    insert_query = """
                    INSERT INTO analysis (
                        company_id, compounded_sales_growth, compounded_profit_growth, 
                        stock_price_cagr, roe, debt_to_equity, current_ratio, 
                        overall_score, analysis_date
                    ) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    values = (
                        company_id,
                        analysis_data.get('sales_growth', 'N/A'),
                        analysis_data.get('profit_growth', 'N/A'),
                        analysis_data.get('stock_cagr', 'N/A'),
                        analysis_data.get('roe', 'N/A'),
                        analysis_data.get('debt_to_equity', 'N/A'),
                        analysis_data.get('current_ratio', 'N/A'),
                        analysis_data.get('overall_score', 0),
                        datetime.now()
                    )
                    
                    cursor.execute(insert_query, values)
                    conn.commit()
                    logger.info(f" Analysis data saved for {company_id}")
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Analysis transaction failed for {company_id}: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Database error for {company_id}: {e}")
            return False
    
    def get_company_list(self) -> List[Dict[str, Any]]:
        """
        Enhanced company list method
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)  # Return as dictionary
                cursor.execute("""
                    SELECT id, company_name, sector, industry, created_at
                    FROM companies 
                    ORDER BY company_name
                """)
                companies = cursor.fetchall()
                
                logger.info(f"Retrieved {len(companies)} companies from database")
                return companies
                
        except Exception as e:
            logger.error(f"Error fetching company list: {e}")
            return []
    
    def check_existing_analysis(self, company_id: str) -> bool:
        """
        Enhanced analysis check
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check both prosandcons and analysis tables
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM prosandcons WHERE company_id = %s) as pros_cons_count,
                        (SELECT COUNT(*) FROM analysis WHERE company_id = %s) as analysis_count
                """, (company_id, company_id))
                
                result = cursor.fetchone()
                has_pros_cons = result[0] > 0
                has_analysis = result[1] > 0
                
                logger.info(f"Existing analysis check for {company_id}: pros_cons={has_pros_cons}, analysis={has_analysis}")
                return has_pros_cons and has_analysis
                
        except Exception as e:
            logger.error(f"Error checking existing analysis: {e}")
            return False
    
    
    def save_financial_data(self, company_id: str, financial_data: Dict[str, Any]) -> bool:
        """
        Save raw financial data from API (new method)
        Creates a financial_data table if it doesn't exist
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create financial_data table if it doesn't exist
                create_table_query = """
                CREATE TABLE IF NOT EXISTS financial_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company_id VARCHAR(20) NOT NULL,
                    roe DECIMAL(10,2) NULL,
                    roce DECIMAL(10,2) NULL,
                    debt_to_equity DECIMAL(10,2) NULL,
                    current_ratio DECIMAL(10,2) NULL,
                    eps DECIMAL(10,2) NULL,
                    pe_ratio DECIMAL(10,2) NULL,
                    dividend_yield DECIMAL(10,2) NULL,
                    sales_growth DECIMAL(10,2) NULL,
                    profit_growth DECIMAL(10,2) NULL,
                    raw_api_data JSON NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_company (company_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
                cursor.execute(create_table_query)
                
                # Insert or update financial data
                insert_query = """
                INSERT INTO financial_data (
                    company_id, roe, roce, debt_to_equity, current_ratio, 
                    eps, pe_ratio, dividend_yield, sales_growth, profit_growth, raw_api_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    roe = VALUES(roe),
                    roce = VALUES(roce),
                    debt_to_equity = VALUES(debt_to_equity),
                    current_ratio = VALUES(current_ratio),
                    eps = VALUES(eps),
                    pe_ratio = VALUES(pe_ratio),
                    dividend_yield = VALUES(dividend_yield),
                    sales_growth = VALUES(sales_growth),
                    profit_growth = VALUES(profit_growth),
                    raw_api_data = VALUES(raw_api_data),
                    updated_at = CURRENT_TIMESTAMP
                """
                
                values = (
                    company_id,
                    financial_data.get('roe'),
                    financial_data.get('roce'),
                    financial_data.get('debt_to_equity'),
                    financial_data.get('current_ratio'),
                    financial_data.get('eps'),
                    financial_data.get('pe_ratio'),
                    financial_data.get('dividend_yield'),
                    financial_data.get('sales_growth'),
                    financial_data.get('profit_growth'),
                    json.dumps(financial_data.get('raw_data', {}))
                )
                
                cursor.execute(insert_query, values)
                conn.commit()
                logger.info(f"✓ Financial data saved for {company_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving financial data for {company_id}: {e}")
            return False
    
    def log_processing_step(self, company_id: str, stage: str, status: str, 
                        message: str = None, error_details: str = None, 
                        processing_time: float = None) -> bool:
        """
        Log processing steps for monitoring (new method)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create processing_logs table if it doesn't exist
                create_table_query = """
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    company_id VARCHAR(20) NOT NULL,
                    stage VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    message TEXT NULL,
                    error_details TEXT NULL,
                    processing_time DECIMAL(10,3) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_company_stage (company_id, stage, created_at),
                    INDEX idx_status (status, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
                cursor.execute(create_table_query)
                
                # Insert log entry
                insert_query = """
                INSERT INTO processing_logs (
                    company_id, stage, status, message, error_details, processing_time
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    company_id, stage, status, message, error_details, processing_time
                ))
                conn.commit()
                
                logger.info(f"Logged {stage} {status} for {company_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error logging processing step: {e}")
            return False
    
    def get_processing_status(self, company_id: str) -> Dict[str, Any]:
        """
        Get processing status for a company (new method)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # Get latest log entries for this company
                cursor.execute("""
                    SELECT stage, status, message, created_at 
                    FROM processing_logs 
                    WHERE company_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """, (company_id,))
                
                logs = cursor.fetchall()
                
                # Process status information
                status_info = {
                    'company_id': company_id,
                    'overall_status': 'not_started',
                    'stages': {},
                    'last_updated': None,
                    'error_count': 0
                }
                
                for log in logs:
                    stage = log['stage']
                    if stage not in status_info['stages']:
                        status_info['stages'][stage] = {
                            'status': log['status'],
                            'message': log['message'],
                            'last_updated': log['created_at']
                        }
                    
                    if log['status'] == 'failed':
                        status_info['error_count'] += 1
                    
                    if not status_info['last_updated'] or log['created_at'] > status_info['last_updated']:
                        status_info['last_updated'] = log['created_at']
                
                # Determine overall status
                if 'ml_analysis' in status_info['stages'] and status_info['stages']['ml_analysis']['status'] == 'completed':
                    status_info['overall_status'] = 'completed'
                elif status_info['error_count'] > 0:
                    status_info['overall_status'] = 'failed'
                elif len(status_info['stages']) > 0:
                    status_info['overall_status'] = 'in_progress'
                
                return status_info
                
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {'company_id': company_id, 'overall_status': 'error', 'error': str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics (new method)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count companies
                cursor.execute("SELECT COUNT(*) FROM companies")
                stats['total_companies'] = cursor.fetchone()[0]
                
                # Count analyses
                cursor.execute("SELECT COUNT(DISTINCT company_id) FROM analysis")
                stats['companies_analyzed'] = cursor.fetchone()[0]
                
                # Count pros/cons
                cursor.execute("SELECT COUNT(DISTINCT company_id) FROM prosandcons")
                stats['companies_with_pros_cons'] = cursor.fetchone()[0]
                
                # Count processing logs
                cursor.execute("SELECT COUNT(*) FROM processing_logs")
                stats['total_processing_logs'] = cursor.fetchone()[0]
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM processing_logs 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                """)
                stats['activity_last_24h'] = cursor.fetchone()[0]
                
                logger.info(f"Database stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_logs(self, days_old: int = 30) -> int:
        """
        Clean up old processing logs (new method)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                delete_query = """
                DELETE FROM processing_logs 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
                """
                
                cursor.execute(delete_query, (days_old,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old log entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            return 0

# Factory function for easy import
def create_enhanced_database_operations() -> EnhancedDatabaseOperations:
    """
    Factory function to create EnhancedDatabaseOperations instance
    
    Returns:
        EnhancedDatabaseOperations instance
    """
    return EnhancedDatabaseOperations()

# Example usage
if __name__ == "__main__":
    # Example of using the enhanced database operations
    db_ops = create_enhanced_database_operations()
    
    # Test database stats
    stats = db_ops.get_database_stats()
    print(f"Database Statistics: {stats}")
    
    # Example of enhanced pros/cons insertion
    sample_pros = ["Strong ROE of 25%", "Low debt ratio", "Consistent growth"]
    sample_cons = ["High PE ratio", "Market volatility risk"]
    
    result = db_ops.insert_pros_cons("TCS", sample_pros, sample_cons)
    print(f"Pros/Cons insertion result: {result}")
    
    # Check processing status
    status = db_ops.get_processing_status("TCS")
    print(f"Processing status: {status}")

    # For backward compatibility
DatabaseOperations = EnhancedDatabaseOperations