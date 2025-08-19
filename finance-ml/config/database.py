import os
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', 'your_password')
        self.database = os.getenv('DB_NAME', 'ml_financial_analysis')
        self.port = int(os.getenv('DB_PORT', '3306'))
    
    def get_connection(self):
        """Get MySQL database connection using PyMySQL"""
        try:
            connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                autocommit=True,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print(f"Successfully connected to {self.database} database")
            return connection
        except Exception as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def test_connection(self):
        """Test database connection"""
        conn = self.get_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT DATABASE();")
                    result = cursor.fetchone()
                    print(f"Connected to database: {result['DATABASE()']}")
                    
                    # Show available tables
                    cursor.execute("SHOW TABLES;")
                    tables = cursor.fetchall()
                    print("Available tables:")
                    for table in tables:
                        table_name = list(table.values())[0]
                        print(f"  - {table_name}")
                
                conn.close()
                return True
            except Exception as e:
                print(f"Error testing connection: {e}")
                conn.close()
                return False
        return False