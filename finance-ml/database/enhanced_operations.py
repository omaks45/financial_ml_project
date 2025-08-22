"""
Enhanced Database Operations for ML Financial Analysis Pipeline
Builds on your existing operations.py with ML-specific functionality
"""

import pymysql
from datetime import datetime
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from config.database import DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLDatabaseOperations:
    """
    Enhanced database operations specifically designed for ML pipeline
    Extends your existing functionality with ML-focused methods
    """
    
    def __init__(self):
        self.config = DatabaseConfig()
    
    @contextmanager
    def get_connection(self):
        """Enhanced connection method with proper error handling"""
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
    
    # =================== COMPANY MANAGEMENT ===================
    
    def insert_company(self, company_id: str, company_name: str = None, 
                    sector: str = None, industry: str = None) -> bool:
        """Insert or update company information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                insert_query = """
                INSERT INTO companies (company_id, company_name, sector, industry)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    company_name = COALESCE(VALUES(company_name), company_name),
                    sector = COALESCE(VALUES(sector), sector),
                    industry = COALESCE(VALUES(industry), industry),
                    updated_at = CURRENT_TIMESTAMP
                """
                
                cursor.execute(insert_query, (company_id, company_name, sector, industry))
                logger.info(f"✓ Company {company_id} inserted/updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error inserting company {company_id}: {e}")
            return False
    
    def get_all_companies(self) -> List[Dict[str, Any]]:
        """Get list of all companies"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT company_id, company_name, sector, industry, created_at
                    FROM companies 
                    ORDER BY company_name
                """)
                companies = cursor.fetchall()
                logger.info(f"Retrieved {len(companies)} companies")
                return companies
                
        except Exception as e:
            logger.error(f"Error fetching companies: {e}")
            return []
    
    # =================== FINANCIAL DATA MANAGEMENT ===================
    
    def save_financial_data(self, company_id: str, financial_metrics: Dict[str, Any]) -> bool:
        """
        Save comprehensive financial data from API
        Task 3: Store financial data for ML analysis
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate data completeness score
                total_fields = 15  # Key financial metrics
                available_fields = sum(1 for key in ['roe', 'roce', 'debt_to_equity', 'current_ratio', 
                                                'book_value', 'eps', 'pe_ratio', 'dividend_yield',
                                                'sales_growth', 'profit_growth', 'revenue', 'net_income',
                                                'total_assets', 'total_debt', 'market_cap'] 
                                    if financial_metrics.get(key) is not None)
                completeness_score = (available_fields / total_fields) * 100
                
                insert_query = """
                INSERT INTO financial_data (
                    company_id, roe, roce, debt_to_equity, current_ratio, book_value,
                    eps, pe_ratio, dividend_yield, sales_growth, profit_growth,
                    revenue, net_income, total_assets, total_debt, market_cap,
                    data_completeness_score, raw_api_data
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                    total_assets = VALUES(total_assets),
                    total_debt = VALUES(total_debt),
                    market_cap = VALUES(market_cap),
                    data_completeness_score = VALUES(data_completeness_score),
                    raw_api_data = VALUES(raw_api_data),
                    updated_at = CURRENT_TIMESTAMP
                """
                
                values = (
                    company_id,
                    financial_metrics.get('roe'),
                    financial_metrics.get('roce'),
                    financial_metrics.get('debt_to_equity'),
                    financial_metrics.get('current_ratio'),
                    financial_metrics.get('book_value'),
                    financial_metrics.get('eps'),
                    financial_metrics.get('pe_ratio'),
                    financial_metrics.get('dividend_yield'),
                    financial_metrics.get('sales_growth'),
                    financial_metrics.get('profit_growth'),
                    financial_metrics.get('revenue'),
                    financial_metrics.get('net_income'),
                    financial_metrics.get('total_assets'),
                    financial_metrics.get('total_debt'),
                    financial_metrics.get('market_cap'),
                    completeness_score,
                    json.dumps(financial_metrics.get('raw_data', {}))
                )
                
                cursor.execute(insert_query, values)
                logger.info(f"✓ Financial data saved for {company_id} (completeness: {completeness_score:.1f}%)")
                return True
                
        except Exception as e:
            logger.error(f"Error saving financial data for {company_id}: {e}")
            return False
    
    def get_financial_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve financial data for a company"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM financial_data WHERE company_id = %s
                """, (company_id,))
                
                result = cursor.fetchone()
                if result:
                    logger.info(f"✓ Financial data retrieved for {company_id}")
                return result
                
        except Exception as e:
            logger.error(f"Error retrieving financial data for {company_id}: {e}")
            return None
    
    # =================== ML RESULTS MANAGEMENT ===================
    
    def save_ml_results(self, company_id: str, analysis_title: str, 
                    pros_list: List[str], cons_list: List[str],
                    scores: Dict[str, float], status: str = 'completed') -> bool:
        """
        Save ML analysis results
        Task 4: Store ML analysis output with pros/cons and scores
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Limit pros and cons to 3 each as per requirements
                limited_pros = pros_list[:3] if pros_list else []
                limited_cons = cons_list[:3] if cons_list else []
                
                insert_query = """
                INSERT INTO ml_results (
                    company_id, analysis_title, pros, cons,
                    overall_score, growth_score, profitability_score, 
                    financial_stability_score, confidence_score, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                
                values = (
                    company_id,
                    analysis_title,
                    json.dumps(limited_pros),
                    json.dumps(limited_cons),
                    scores.get('overall_score', 0),
                    scores.get('growth_score', 0),
                    scores.get('profitability_score', 0),
                    scores.get('financial_stability_score', 0),
                    scores.get('confidence_score', 0),
                    status
                )
                
                cursor.execute(insert_query, values)
                logger.info(f"✓ ML results saved for {company_id}: {len(limited_pros)} pros, {len(limited_cons)} cons")
                return True
                
        except Exception as e:
            logger.error(f"Error saving ML results for {company_id}: {e}")
            return False
    
    def get_ml_results(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve ML analysis results for a company"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM ml_results WHERE company_id = %s
                """, (company_id,))
                
                result = cursor.fetchone()
                if result:
                    # Parse JSON fields
                    if result.get('pros'):
                        result['pros_list'] = json.loads(result['pros'])
                    if result.get('cons'):
                        result['cons_list'] = json.loads(result['cons'])
                    logger.info(f"✓ ML results retrieved for {company_id}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error retrieving ML results for {company_id}: {e}")
            return None
    
    def get_all_ml_results(self, status: str = None) -> List[Dict[str, Any]]:
        """Get all ML results, optionally filtered by status"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if status:
                    query = """
                    SELECT ml.*, c.company_name, c.sector
                    FROM ml_results ml
                    LEFT JOIN companies c ON ml.company_id = c.company_id
                    WHERE ml.status = %s
                    ORDER BY ml.overall_score DESC
                    """
                    cursor.execute(query, (status,))
                else:
                    query = """
                    SELECT ml.*, c.company_name, c.sector
                    FROM ml_results ml
                    LEFT JOIN companies c ON ml.company_id = c.company_id
                    ORDER BY ml.overall_score DESC
                    """
                    cursor.execute(query)
                
                results = cursor.fetchall()
                
                # Parse JSON fields for each result
                for result in results:
                    if result.get('pros'):
                        result['pros_list'] = json.loads(result['pros'])
                    if result.get('cons'):
                        result['cons_list'] = json.loads(result['cons'])
                
                logger.info(f"Retrieved {len(results)} ML results")
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving ML results: {e}")
            return []
    
    # =================== PROCESSING LOGS ===================
    
    def log_processing_step(self, company_id: str, stage: str, status: str, 
                        message: str = None, error_details: str = None,
                        processing_time: float = None, batch_id: str = None) -> bool:
        """Log processing steps for monitoring"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                insert_query = """
                INSERT INTO processing_logs (
                    company_id, stage, status, message, error_details, 
                    processing_time, batch_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    company_id, stage, status, message, error_details, 
                    processing_time, batch_id
                ))
                
                return True
                
        except Exception as e:
            logger.error(f"Error logging processing step: {e}")
            return False
    
    def get_processing_status(self, company_id: str = None) -> Dict[str, Any]:
        """Get processing status for companies"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if company_id:
                    # Get status for specific company
                    cursor.execute("""
                        SELECT stage, status, message, processing_time, created_at
                        FROM processing_logs
                        WHERE company_id = %s
                        ORDER BY created_at DESC
                        LIMIT 10
                    """, (company_id,))
                    
                    logs = cursor.fetchall()
                    return {'company_id': company_id, 'logs': logs}
                else:
                    # Get overall processing statistics
                    cursor.execute("""
                        SELECT 
                            COUNT(DISTINCT company_id) as total_companies_processed,
                            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_steps,
                            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_steps,
                            AVG(processing_time) as avg_processing_time
                        FROM processing_logs
                        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                    """)
                    
                    stats = cursor.fetchone()
                    return stats or {}
                    
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {}
    
    # =================== DATA ANALYSIS & REPORTING ===================
    
    def get_complete_company_analysis(self, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete analysis combining company info, financial data, and ML results
        Task 4: Comprehensive data retrieval for analysis
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    c.company_id,
                    c.company_name,
                    c.sector,
                    c.industry,
                    f.roe,
                    f.roce,
                    f.debt_to_equity,
                    f.current_ratio,
                    f.sales_growth,
                    f.profit_growth,
                    f.data_completeness_score,
                    ml.analysis_title,
                    ml.pros,
                    ml.cons,
                    ml.overall_score,
                    ml.growth_score,
                    ml.profitability_score,
                    ml.financial_stability_score,
                    ml.confidence_score,
                    ml.status as ml_status,
                    ml.updated_at as analysis_date
                FROM companies c
                LEFT JOIN financial_data f ON c.company_id = f.company_id
                LEFT JOIN ml_results ml ON c.company_id = ml.company_id
                WHERE c.company_id = %s
                """
                
                cursor.execute(query, (company_id,))
                result = cursor.fetchone()
                
                if result:
                    # Parse JSON fields
                    if result.get('pros'):
                        result['pros_list'] = json.loads(result['pros'])
                    if result.get('cons'):
                        result['cons_list'] = json.loads(result['cons'])
                    
                    logger.info(f"✓ Complete analysis retrieved for {company_id}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error retrieving complete analysis for {company_id}: {e}")
            return None
    
    def get_top_companies_by_score(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top companies by overall ML score"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    c.company_id,
                    c.company_name,
                    c.sector,
                    ml.overall_score,
                    ml.analysis_title,
                    ml.status
                FROM ml_results ml
                JOIN companies c ON ml.company_id = c.company_id
                WHERE ml.status = 'completed' AND ml.overall_score IS NOT NULL
                ORDER BY ml.overall_score DESC
                LIMIT %s
                """
                
                cursor.execute(query, (limit,))
                results = cursor.fetchall()
                
                logger.info(f"Retrieved top {len(results)} companies by ML score")
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving top companies: {e}")
            return []
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM companies")
                stats['total_companies'] = cursor.fetchone()['COUNT(*)']
                
                cursor.execute("SELECT COUNT(*) FROM financial_data")
                stats['companies_with_financial_data'] = cursor.fetchone()['COUNT(*)']
                
                cursor.execute("SELECT COUNT(*) FROM ml_results WHERE status = 'completed'")
                stats['completed_ml_analyses'] = cursor.fetchone()['COUNT(*)']
                
                # Data quality metrics
                cursor.execute("SELECT AVG(data_completeness_score) FROM financial_data")
                result = cursor.fetchone()
                stats['avg_data_completeness'] = float(result['AVG(data_completeness_score)']) if result['AVG(data_completeness_score)'] else 0
                
                # ML performance metrics
                cursor.execute("""
                    SELECT 
                        AVG(overall_score) as avg_overall_score,
                        MAX(overall_score) as max_overall_score,
                        MIN(overall_score) as min_overall_score
                    FROM ml_results 
                    WHERE status = 'completed'
                """)
                ml_stats = cursor.fetchone()
                if ml_stats:
                    stats.update(ml_stats)
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM processing_logs 
                    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                """)
                stats['processing_activity_24h'] = cursor.fetchone()['COUNT(*)']
                
                logger.info("Database statistics calculated")
                return stats
                
        except Exception as e:
            logger.error(f"Error calculating database statistics: {e}")
            return {}
    
    # =================== BATCH OPERATIONS ===================
    
    def batch_process_status(self, company_ids: List[str]) -> Dict[str, str]:
        """Check processing status for multiple companies"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create placeholder string for IN clause
                placeholders = ','.join(['%s'] * len(company_ids))
                
                query = f"""
                SELECT 
                    c.company_id,
                    CASE 
                        WHEN ml.status = 'completed' THEN 'completed'
                        WHEN ml.status = 'failed' THEN 'failed'
                        WHEN f.company_id IS NOT NULL THEN 'has_data'
                        ELSE 'no_data'
                    END as status
                FROM companies c
                LEFT JOIN financial_data f ON c.company_id = f.company_id
                LEFT JOIN ml_results ml ON c.company_id = ml.company_id
                WHERE c.company_id IN ({placeholders})
                """
                
                cursor.execute(query, company_ids)
                results = cursor.fetchall()
                
                status_map = {result['company_id']: result['status'] for result in results}
                
                # Add missing companies as 'not_found'
                for company_id in company_ids:
                    if company_id not in status_map:
                        status_map[company_id] = 'not_found'
                
                return status_map
                
        except Exception as e:
            logger.error(f"Error checking batch process status: {e}")
            return {company_id: 'error' for company_id in company_ids}
    
    def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old processing logs and temporary data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cleanup_results = {}
                
                # Clean old processing logs
                cursor.execute("""
                    DELETE FROM processing_logs 
                    WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
                """, (days_old,))
                cleanup_results['processing_logs_deleted'] = cursor.rowcount
                
                logger.info(f"Cleanup completed: {cleanup_results}")
                return cleanup_results
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {}

# Factory function for easy import
def create_enhanced_ml_database() -> EnhancedMLDatabaseOperations:
    """Factory function to create enhanced ML database operations instance"""
    return EnhancedMLDatabaseOperations()

# Example usage and testing
if __name__ == "__main__":
    # Create database instance
    db = create_enhanced_ml_database()
    
    # Example: Save sample financial data
    sample_financial_data = {
        'roe': 25.4,
        'roce': 30.2,
        'debt_to_equity': 0.1,
        'current_ratio': 2.5,
        'sales_growth': 15.3,
        'profit_growth': 18.2,
        'raw_data': {'source': 'test', 'timestamp': str(datetime.now())}
    }
    
    # Example: Save ML results
    sample_scores = {
        'overall_score': 87.5,
        'growth_score': 82.0,
        'profitability_score': 95.0,
        'financial_stability_score': 88.0,
        'confidence_score': 92.0
    }
    
    sample_pros = ['Strong ROE of 25.4%', 'Low debt ratio', 'Good growth trajectory']
    sample_cons = ['High PE ratio', 'Market volatility']
    
    # Test the operations
    print("Testing Enhanced ML Database Operations...")
    
    # Insert company
    db.insert_company('TEST', 'Test Company', 'Technology', 'Software')
    
    # Save financial data
    db.save_financial_data('TEST', sample_financial_data)
    
    # Save ML results
    db.save_ml_results('TEST', 'Test analysis title', sample_pros, sample_cons, sample_scores)
    
    # Retrieve complete analysis
    analysis = db.get_complete_company_analysis('TEST')
    if analysis:
        print(f"Complete analysis for TEST:")
        print(f"  Overall Score: {analysis.get('overall_score')}")
        print(f"  Pros: {analysis.get('pros_list', [])}")
        print(f"  Cons: {analysis.get('cons_list', [])}")
    
    # Get database statistics
    stats = db.get_database_statistics()
    print(f"Database Statistics: {stats}")