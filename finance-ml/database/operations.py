import mysql.connector
from datetime import datetime
from config.database import DatabaseConfig
import json

class DatabaseOperations:
    def __init__(self):
        self.config = DatabaseConfig()
    
    def get_connection(self):
        """Get database connection"""
        return self.config.get_connection()
    
    def insert_pros_cons(self, company_id, pros_list, cons_list):
        """Insert pros and cons into prosandcons table"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            # Clear existing entries for this company
            delete_query = "DELETE FROM prosandcons WHERE company_id = %s"
            cursor.execute(delete_query, (company_id,))
            
            # Insert new pros
            for pro in pros_list:
                insert_query = """
                INSERT INTO prosandcons (company_id, pros, cons) 
                VALUES (%s, %s, NULL)
                """
                cursor.execute(insert_query, (company_id, pro))
            
            # Insert new cons
            for con in cons_list:
                insert_query = """
                INSERT INTO prosandcons (company_id, pros, cons) 
                VALUES (%s, NULL, %s)
                """
                cursor.execute(insert_query, (company_id, con))
            
            conn.commit()
            print(f"✓ Pros and cons saved for {company_id}")
            return True
            
        except Exception as e:
            print(f"Database error for {company_id}: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def insert_analysis_data(self, company_id, analysis_data):
        """Insert analysis data into analysis table"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            
            # Clear existing entries for this company
            delete_query = "DELETE FROM analysis WHERE company_id = %s"
            cursor.execute(delete_query, (company_id,))
            
            # Insert new analysis data
            insert_query = """
            INSERT INTO analysis (company_id, compounded_sales_growth, compounded_profit_growth, stock_price_cagr, roe) 
            VALUES (%s, %s, %s, %s, %s)
            """
            
            values = (
                company_id,
                analysis_data.get('sales_growth', 'N/A'),
                analysis_data.get('profit_growth', 'N/A'),
                analysis_data.get('stock_cagr', 'N/A'),
                analysis_data.get('roe', 'N/A')
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            print(f"✓ Analysis data saved for {company_id}")
            return True
            
        except Exception as e:
            print(f"Database error for {company_id}: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_company_list(self):
        """Get list of all companies from companies table"""
        try:
            conn = self.get_connection()
            if not conn:
                return []
                
            cursor = conn.cursor()
            cursor.execute("SELECT id, company_name FROM companies")
            companies = cursor.fetchall()
            
            company_list = [{'id': comp[0], 'name': comp[1]} for comp in companies]
            return company_list
            
        except Exception as e:
            print(f"Error fetching company list: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def check_existing_analysis(self, company_id):
        """Check if analysis already exists for company"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
                
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prosandcons WHERE company_id = %s", (company_id,))
            count = cursor.fetchone()[0]
            
            return count > 0
            
        except Exception as e:
            print(f"Error checking existing analysis: {e}")
            return False
        finally:
            if conn:
                conn.close()